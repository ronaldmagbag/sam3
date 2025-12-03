#!/usr/bin/env python
"""
GeoSeg to COCO Dataset Preparation Script

This script uses geoseg tools to:
1. Extract GeoJSON features from OSM data
2. Generate tiles covering the features
3. Download satellite images
4. Rasterize features to masks
5. Convert to COCO format

Usage:
    python full_pipeline.py --osm-file data.osm.pbf --output ./data/coco_dataset --zoom 19
"""

import os
import sys
import argparse
import json
import csv
import shutil
import subprocess
import collections
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from tiles import tiles_from_csv, tiles_from_slippy_map
from tools.extract import main as extract_main
from tools.cover import main as cover_main
from tools.download import main as download_main
from tools.rasterize import main as rasterize_main
from colors import make_palette, make_segmentation_palette, Mapbox, CLASS_COLORS, FEATURE_TYPE_TO_CLASS
GEOSEG_AVAILABLE = True

try:
    from dataset_builder.coco_exporter import COCOExporter
    from dataset_builder.splitter import split_dataset
    COCO_EXPORTER_AVAILABLE = True
except ImportError as e:
    COCO_EXPORTER_AVAILABLE = False
    print(f"Warning: COCO exporter not available: {e}")
    print("  Ensure dataset_builder module is in the current directory")

try:
    from label_preparation.sam_annotator import SAMAnnotator
    from label_preparation.sam2_annotator import SAM2Annotator
    from label_preparation.sam3_annotator import SAM3Annotator
    from label_preparation.mask_merger import MaskMerger
    SAM_AVAILABLE = True
except ImportError as e:
    SAM_AVAILABLE = False
    print(f"Note: SAM annotator not available: {e}")


class GeoSegToCOCO:
    """Convert GeoSeg workflow output to COCO format."""
    
    def __init__(
        self,
        output_dir: str,
        zoom: int = 19,
        tile_size: int = 512,
        feature_types: List[str] = None,
        test_tiles: Optional[int] = None
    ):
        """
        Initialize the converter.
        
        Args:
            output_dir: Base output directory
            zoom: Zoom level for tiles
            tile_size: Size of tiles in pixels
            feature_types: List of feature types to extract (building, road, parking)
            test_tiles: Limit number of tiles for testing (optional)
        """
        self.output_dir = Path(output_dir)
        self.zoom = zoom
        self.tile_size = tile_size
        # Handle "all" feature type
        if feature_types and "all" in feature_types:
            self.feature_types = ["building", "road", "parking"]
            self._use_all_handler = True
        else:
            self.feature_types = feature_types or ["building"]
            self._use_all_handler = False
        self.test_tiles = test_tiles
        
        # Create directory structure
        self.work_dir = self.output_dir / "geoseg_work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.geojson_dir = self.work_dir / "geojson"
        self.tiles_dir = self.work_dir / "tiles"
        self.images_dir = self.work_dir / "images"
        self.masks_dir = self.work_dir / "masks"
        self.sam_masks_dir = self.work_dir / "sam_masks"
        self.merged_masks_dir = self.work_dir / "merged_masks"
        self.coco_dir = self.output_dir / "coco_dataset"
        
        for d in [self.geojson_dir, self.tiles_dir, self.images_dir, self.masks_dir, self.sam_masks_dir, self.merged_masks_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _validate_osm_file(self, osm_file: str) -> str:
        """Validate OSM file exists and return absolute path."""
        osm_file_abs = os.path.abspath(osm_file)
        if not os.path.exists(osm_file_abs):
            suggestions = []
            current_dir = Path.cwd()
            for alt_path in [
                current_dir / "data" / "Essex" / "essex-latest.osm.pbf",
                current_dir / "data" / "essex-latest.osm.pbf",
                current_dir.parent / "data" / "essex-latest.osm.pbf",
            ]:
                if alt_path.exists():
                    suggestions.append(str(alt_path.relative_to(current_dir)))
            
            error_msg = f"OSM file not found: {osm_file_abs}"
            if suggestions:
                error_msg += f"\n\nDid you mean one of these?\n  - " + "\n  - ".join(suggestions)
            else:
                error_msg += f"\n\nTo download a test file, run:\n  wget https://download.geofabrik.de/europe/united-kingdom/england/essex-latest.osm.pbf -O data/essex-latest.osm.pbf"
            raise FileNotFoundError(error_msg)
        return osm_file_abs
    
    def extract_features_multi(self, osm_file: str, feature_types: List[str]) -> str:
        """Extract multiple feature types from OSM in a single pass."""
        if self._use_all_handler:
            print(f"\n[Step 1] Extracting all features (building, road, parking) from OSM...")
        else:
            print(f"\n[Step 1] Extracting {', '.join(feature_types)} features from OSM...")
        
        osm_file_abs = self._validate_osm_file(osm_file)
        geojson_file = self.geojson_dir / "merged_features.geojson"
        
        # Check if output file already exists
        if geojson_file.exists():
            print(f"  ⏭️  Skipping: GeoJSON file already exists ({geojson_file.stat().st_size:,} bytes)")
            return str(geojson_file)
        
        # Check for existing batch files (use first one if exists)
        batch_files = list(self.geojson_dir.glob("merged_features-*.geojson"))
        if batch_files:
            print(f"  ⏭️  Skipping: Found {len(batch_files)} batch file(s), using first one")
            return str(batch_files[0])
        
        # Extract using AllFeaturesHandler
        print(f"  Scanning OSM file (this may take a while for large files)...")
        class Args:
            # Use "all" if we're extracting all types, otherwise use the list
            types = ["all"] if self._use_all_handler else feature_types
            batch = 100000
            map = osm_file_abs
            out = str(geojson_file)
        extract_main(Args())
        
        # Check for batch files created by FeatureStorage (use first one)
        batch_files = list(self.geojson_dir.glob("merged_features-*.geojson"))
        if batch_files:
            print(f"  ✅ Extracted features to {len(batch_files)} batch file(s)")
            return str(batch_files[0])
        
        if geojson_file.exists():
            return str(geojson_file)
        
        raise FileNotFoundError(f"GeoJSON file not found after extraction: {self.geojson_dir}")
    
    def extract_features(self, osm_file: str, feature_type: str) -> str:
        """Extract single feature type from OSM (uses extract_features_multi internally)."""
        return self.extract_features_multi(osm_file, [feature_type])
    
    def generate_tiles(self, geojson_file: str) -> str:
        """Generate tiles covering GeoJSON features."""
        print(f"\n[Step 2] Generating tiles covering features...")
        
        tiles_file = self.tiles_dir / "tiles.csv"
        
        if tiles_file.exists() and tiles_file.stat().st_size > 0:
            # Count tiles to show progress
            with open(tiles_file, 'r') as f:
                tile_count = sum(1 for _ in f)
            print(f"  ⏭️  Skipping: Tiles file already exists with {tile_count} tiles: {tiles_file}")
            return str(tiles_file)
        
        class Args:
            zoom = self.zoom
            features = geojson_file
            out = str(tiles_file)
        
        cover_main(Args())
        
        # Count tiles (using CSV reader to skip empty lines)
        import csv
        tile_count = 0
        with open(tiles_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Only count non-empty rows
                    tile_count += 1
        print(f"  ✅ Generated {tile_count} tiles")
        
        # Limit tiles if test_tiles is set
        if self.test_tiles is not None and tile_count > self.test_tiles:
            self._limit_tiles(tiles_file, self.test_tiles)
            print(f"  🔬 Limited to {self.test_tiles} tiles for testing")
        
        return str(tiles_file)
    
    def download_images(self, tiles_file: str, mapbox_token: Optional[str] = None, url_template: Optional[str] = None) -> str:
        """Download satellite images for tiles."""
        print(f"\n[Step 3] Downloading satellite images...")
        
        zoom_dir = self.images_dir / str(self.zoom)
        if zoom_dir.exists() and any(zoom_dir.iterdir()):
            # Count downloaded images
            image_count = sum(1 for _ in zoom_dir.rglob("*.png")) + sum(1 for _ in zoom_dir.rglob("*.webp"))
            print(f"  ⏭️  Skipping: Images already downloaded ({image_count} images) in {zoom_dir}")
            return str(self.images_dir)
        
        # Default to Mapbox if token provided, otherwise use OpenStreetMap
        if url_template is None:
            if mapbox_token:
                url_template = f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.webp?access_token={mapbox_token}"
            else:
                # Use OpenStreetMap tile server (no token needed)
                url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        
        class Args:
            url = url_template
            ext = "png"
            rate = 10
            tiles = tiles_file
            out = str(self.images_dir)
        
        download_main(Args())
        
        print(f"  ✅ Downloaded images to {self.images_dir}")
        return str(self.images_dir)
    
    def _feature_to_mercator(self, feature):
        """Normalize feature and converts coords to 3857."""
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3857)
        
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            xys = (zip(*part) for part in geometry["coordinates"])
            xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
            yield {"coordinates": list(xys), "type": "Polygon"}
        elif geometry["type"] == "MultiPolygon":
            for component in geometry["coordinates"]:
                xys = (zip(*part) for part in component)
                xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
                yield {"coordinates": list(xys), "type": "Polygon"}
    
    def _rasterize_multiclass(self, geojson_file: str, tiles_file: str, dataset_config: Path):
        """Custom multi-class rasterization with different pixel values per feature type."""
        from tqdm import tqdm
        
        # Load dataset config
        try:
            import toml
            with open(dataset_config, 'r') as f:
                dataset = toml.load(f)
        except ImportError:
            # Fallback: parse manually
            dataset = {"common": {"classes": ["background"] + self.feature_types}}
        
        classes = dataset["common"]["classes"]
        # Get colors from config, or generate them if missing
        if "colors" in dataset["common"]:
            colors = dataset["common"]["colors"]
        else:
            # Generate colors if missing
            colors = ["denim"]  # background
            for class_name in classes[1:]:  # skip background
                colors.append(self._get_color_for_feature_type(class_name))
        
        # Create mapping from feature type to pixel value (class index)
        feature_type_to_class = {ft: idx + 1 for idx, ft in enumerate(self.feature_types)}
        # Background is class 0
        
        # Load GeoJSON
        with open(geojson_file, 'r') as f:
            fc = json.load(f)
        
        # Group features by type and tile
        feature_map = collections.defaultdict(lambda: collections.defaultdict(list))
        
        with tqdm(fc["features"], ascii=True, unit="feature") as pbar:
            for feature in pbar:
                if feature["geometry"]["type"] not in ["Polygon", "MultiPolygon"]:
                    continue
                
                props = feature.get("properties", {})
                # Support feature_types array (use first type) or fallback to feature_type
                feature_types = props.get("feature_types", None)
                if feature_types and isinstance(feature_types, list) and len(feature_types) > 0:
                    # Use first type for rasterization (can be extended to support all types)
                    feature_type = feature_types[0]
                else:
                    # Fallback to singular feature_type for backward compatibility
                    feature_type = props.get("feature_type", self.feature_types[0])
                
                if feature_type not in feature_type_to_class:
                    continue
                
                try:
                    for tile in burntiles.burn([feature], zoom=self.zoom):
                        tile_obj = mercantile.Tile(*tile)
                        feature_map[tile_obj][feature_type].append(feature)
                except ValueError:
                    continue
        
        # Rasterize each tile
        os.makedirs(self.masks_dir, exist_ok=True)
        
        with tqdm(list(tiles_from_csv(tiles_file)), ascii=True, unit="tile") as pbar:
            for tile in pbar:
                # Initialize mask with background (0)
                mask = np.zeros(shape=(self.tile_size, self.tile_size), dtype=np.uint8)
                
                if tile in feature_map:
                    # Rasterize each feature type with its class value
                    for feature_type, features in feature_map[tile].items():
                        class_value = feature_type_to_class[feature_type]
                        
                        # Convert features to mercator and rasterize
                        shapes = []
                        for feature in features:
                            for geometry in self._feature_to_mercator(feature):
                                shapes.append((geometry, class_value))
                        
                        if shapes:
                            bounds = mercantile.xy_bounds(tile)
                            transform = from_bounds(*bounds, self.tile_size, self.tile_size)
                            rasterized = rasterize(shapes, out_shape=(self.tile_size, self.tile_size), transform=transform)
                            # Use maximum to handle overlapping features (later features overwrite)
                            mask = np.maximum(mask, rasterized.astype(np.uint8))
                
                # Save mask
                out_dir = self.masks_dir / str(tile.z) / str(tile.x)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{tile.y}.png"
                
                # Convert to PIL Image with segmentation palette
                img = Image.fromarray(mask, mode="P")
                palette = make_segmentation_palette()
                img.putpalette(palette)
                img.save(out_path, optimize=True)
    
    def rasterize_masks(self, geojson_file: str, tiles_file: str, dataset_config: Optional[str] = None, use_multiclass: bool = True) -> str:
        """Rasterize GeoJSON features to masks with multi-class support."""
        print(f"\n[Step 4] Rasterizing features to masks...")
        
        zoom_mask_dir = self.masks_dir / str(self.zoom)
        # Check if masks actually exist (not just empty directories)
        mask_count = 0
        if zoom_mask_dir.exists():
            mask_count = sum(1 for _ in zoom_mask_dir.rglob("*.png"))
        
        if mask_count > 0:
            print(f"  ⏭️  Skipping: Masks already generated ({mask_count} masks) in {zoom_mask_dir}")
            return str(self.masks_dir)
        
        # Create dataset config
        if dataset_config is None:
            dataset_config = self.work_dir / "dataset.toml"
            self._create_dataset_config(dataset_config)
        else:
            dataset_config = Path(dataset_config)
        
        # Use geoseg rasterize (supports multi-class)
        if len(self.feature_types) > 1:
            print(f"  Using multi-class rasterization for {len(self.feature_types)} feature types...")
        class Args:
            features = geojson_file
            tiles = tiles_file
            out = str(self.masks_dir)
            dataset = str(dataset_config)
            zoom = self.zoom
            size = self.tile_size
        
        rasterize_main(Args())
        
        print(f"  ✅ Generated masks in {self.masks_dir}")
        return str(self.masks_dir)
    
    def _limit_tiles(self, tiles_file: Path, limit: int):
        """Limit tiles CSV to first N tiles."""
        import csv
        
        # Read all tiles using CSV reader to handle formatting properly
        tiles = []
        with open(tiles_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    tiles.append(row)
        
        # Limit to first N tiles
        if len(tiles) > limit:
            tiles = tiles[:limit]
            
            # Write back using CSV writer
            with open(tiles_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(tiles)
    
    def _get_color_for_feature_type(self, feature_type: str) -> str:
        """Get a color name for a feature type (matches SAM colors)."""
        color_map = {
            "building": "building_red",
            "road": "road_yellow",
            "parking": "parking_magenta",
            "water": "water_blue",
            "tree": "tree_green",
            "grass": "grass_lightgreen"
        }
        return color_map.get(feature_type, "building_red")
    
    def _create_dataset_config(self, config_path: Path):
        """Create dataset config with classes and colors for feature types."""
        classes = ["background"]
        colors = ["black"]
        
        for feature_type in self.feature_types:
            classes.append(feature_type)
            colors.append(self._get_color_for_feature_type(feature_type))
        
        try:
            import toml
            config = {
                "common": {
                    "classes": classes,
                    "colors": colors
                }
            }
            with open(config_path, 'w') as f:
                toml.dump(config, f)
        except ImportError:
            # Fallback: write TOML manually
            classes_str = "[" + ", ".join(f'"{c}"' for c in classes) + "]"
            colors_str = "[" + ", ".join(f'"{c}"' for c in colors) + "]"
            with open(config_path, 'w') as f:
                f.write(f"""[common]
classes = {classes_str}
colors = {colors_str}
""")
    
    def annotate_with_sam(self, model_type: str = "vit_b", device: str = "cuda", use_sam2: bool = False, use_sam3: bool = False) -> str:
        """
        Run SAM annotation on images to detect natural features (trees, grass).
        
        Args:
            model_type: SAM model type. For SAM1: 'vit_h', 'vit_l', 'vit_b'. For SAM2: 'tiny', 'small', 'base_plus', 'large'. Not used for SAM3.
            device: Device to run SAM on ('cuda' or 'cpu')
            use_sam2: If True, use SAM 2 instead of SAM 1
            use_sam3: If True, use SAM 3 (text-prompted segmentation)
        
        Returns:
            Path to SAM masks directory
        """
        if not SAM_AVAILABLE:
            print("\n[Step 5] SAM Annotation: Skipped (segment-anything not installed)")
            return str(self.masks_dir)
        
        # Determine version
        if use_sam3:
            sam_version = "SAM 3"
        elif use_sam2:
            sam_version = "SAM 2"
        else:
            sam_version = "SAM"
            
        print(f"\n[Step 5] Running {sam_version} annotation for trees and grass...")
        
        # Get all images from slippy map structure
        images_list = []
        zoom_dir = self.images_dir / str(self.zoom)
        if zoom_dir.exists():
            for x_dir in zoom_dir.iterdir():
                if x_dir.is_dir():
                    for img_file in x_dir.glob("*.*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                            images_list.append(img_file)
        
        if not images_list:
            print("  ⚠️  No images found for SAM annotation")
            return str(self.sam_masks_dir)
        
        # Filter images: only process those that have OSM mask but no SAM mask
        images_to_process = []
        skipped_count = 0
        for img_path in images_list:
            # Create corresponding paths
            rel_path = img_path.relative_to(self.images_dir)
            osm_mask_path = self.masks_dir / rel_path.with_suffix('.png')
            sam_mask_path = self.sam_masks_dir / rel_path.with_suffix('.png')
            
            # Only process if OSM mask exists and SAM mask doesn't exist
            if osm_mask_path.exists() and not sam_mask_path.exists():
                images_to_process.append(img_path)
            elif sam_mask_path.exists():
                skipped_count += 1
        
        # Report status
        existing_sam_count = sum(1 for _ in self.sam_masks_dir.rglob("*.png")) if self.sam_masks_dir.exists() else 0
        if existing_sam_count > 0:
            print(f"  Found {existing_sam_count} existing SAM masks")
        
        if not images_to_process:
            if skipped_count > 0:
                print(f"  ⏭️  All {len(images_list)} images already have SAM masks")
            else:
                print(f"  ⚠️  No images with OSM masks found to process")
            return str(self.sam_masks_dir)
        
        print(f"  Processing {len(images_to_process)} images (skipping {skipped_count} with existing SAM masks)")
        
        # Initialize appropriate annotator
        if use_sam3:
            # SAM3 uses text prompts, no model type selection
            annotator = SAM3Annotator(
                output_dir=str(self.sam_masks_dir),
                device=device,
                confidence_threshold=0.5
            )
        elif use_sam2:
            # Map SAM1 model types to SAM2 model sizes
            sam2_model_map = {
                "vit_b": "small",
                "vit_l": "base_plus", 
                "vit_h": "large",
                # Direct SAM2 sizes
                "tiny": "tiny",
                "small": "small",
                "base_plus": "base_plus",
                "large": "large"
            }
            model_size = sam2_model_map.get(model_type, "large")
            annotator = SAM2Annotator(
                model_size=model_size,
                output_dir=str(self.sam_masks_dir),
                device=device
            )
        else:
            annotator = SAMAnnotator(
                model_type=model_type,
                output_dir=str(self.sam_masks_dir),
                device=device
            )
        
        # Pre-load the model (fail fast if it can't load)
        try:
            annotator.load_model()
        except Exception as e:
            print(f"  ❌ Failed to load SAM model: {e}")
            print("  Skipping SAM annotation")
            return str(self.sam_masks_dir)
        
        # Process images with progress
        from tqdm import tqdm
        processed = 0
        with tqdm(images_to_process, ascii=True, unit="image") as pbar:
            for img_path in pbar:
                # Create corresponding output path maintaining structure
                rel_path = img_path.relative_to(self.images_dir)
                output_path = self.sam_masks_dir / rel_path.with_suffix('.png')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Find corresponding OSM mask (should exist since we filtered for it)
                osm_mask_path = self.masks_dir / rel_path.with_suffix('.png')
                existing_mask = str(osm_mask_path) if osm_mask_path.exists() else None
                
                # Double-check SAM mask doesn't exist (in case it was created between filtering and processing)
                if output_path.exists():
                    continue
                
                try:
                    annotator.annotate_image(
                        image_path=str(img_path),
                        existing_mask_path=existing_mask,
                        output_path=str(output_path)
                    )
                    processed += 1
                except Exception as e:
                    pbar.write(f"  Warning: Failed to process {img_path.name}: {e}")
        
        print(f"  ✅ SAM annotation complete: {processed} masks generated")
        return str(self.sam_masks_dir)
    
    def merge_masks(self, merge_strategy: str = "priority") -> str:
        """
        Merge OSM masks and SAM masks into final segmentation labels.
        
        Args:
            merge_strategy: Strategy for merging ('priority', 'max', 'union')
        
        Returns:
            Path to merged masks directory
        """
        if not SAM_AVAILABLE:
            print("\n[Step 6] Mask Merging: Skipped (using OSM masks only)")
            return str(self.masks_dir)
        
        print(f"\n[Step 6] Merging OSM and SAM masks...")
        
        # Check if merged masks already exist
        merged_mask_count = 0
        if self.merged_masks_dir.exists():
            merged_mask_count = sum(1 for _ in self.merged_masks_dir.rglob("*.png"))
        
        if merged_mask_count > 0:
            print(f"  ⏭️  Skipping: Merged masks already exist ({merged_mask_count} masks)")
            return str(self.merged_masks_dir)
        
        # Check if SAM masks exist
        sam_mask_count = sum(1 for _ in self.sam_masks_dir.rglob("*.png")) if self.sam_masks_dir.exists() else 0
        if sam_mask_count == 0:
            print("  ⚠️  No SAM masks found, using OSM masks only")
            return str(self.masks_dir)
        
        # Initialize merger
        merger = MaskMerger(output_dir=str(self.merged_masks_dir))
        
        # Get all OSM masks
        osm_masks = list(self.masks_dir.rglob("*.png"))
        
        if not osm_masks:
            print("  ⚠️  No OSM masks found")
            return str(self.masks_dir)
        
        print(f"  Found {len(osm_masks)} OSM masks and {sam_mask_count} SAM masks")
        
        # Process masks with progress
        from tqdm import tqdm
        merged_count = 0
        with tqdm(osm_masks, ascii=True, unit="mask") as pbar:
            for osm_mask_path in pbar:
                # Find corresponding SAM mask
                rel_path = osm_mask_path.relative_to(self.masks_dir)
                sam_mask_path = self.sam_masks_dir / rel_path
                
                # Create output path
                output_path = self.merged_masks_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if sam_mask_path.exists():
                    # Merge OSM and SAM masks
                    try:
                        merger.merge_masks(
                            mask_paths=[str(osm_mask_path), str(sam_mask_path)],
                            output_path=str(output_path),
                            merge_strategy=merge_strategy
                        )
                        merged_count += 1
                    except Exception as e:
                        pbar.write(f"  Warning: Failed to merge {osm_mask_path.name}: {e}")
                        # Copy OSM mask as fallback
                        shutil.copy(osm_mask_path, output_path)
                else:
                    # No SAM mask, just copy OSM mask
                    shutil.copy(osm_mask_path, output_path)
                    merged_count += 1
        
        print(f"  ✅ Mask merging complete: {merged_count} masks")
        return str(self.merged_masks_dir)
    
    def convert_to_coco(self, images_dir: str, masks_dir: str, split: bool = True, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Convert slippy map format to COCO format."""
        if not COCO_EXPORTER_AVAILABLE:
            print("Error: COCO exporter not available. Cannot convert to COCO format.")
            print("Please ensure geoseg project is available.")
            return
        
        print(f"\n[Step 5] Converting to COCO format...")
        
        # Check if COCO dataset already exists
        if self.coco_dir.exists():
            # Check if annotation files exist in split directories
            annotation_files = []
            for split_name in ["train", "val", "test"]:
                split_dir = self.coco_dir / split_name
                ann_file = split_dir / "_annotations.coco.json"
                if ann_file.exists():
                    annotation_files.append(ann_file)
            
            # Also check for non-split annotation file
            if not split:
                ann_file = self.coco_dir / "_annotations.coco.json"
                if ann_file.exists():
                    annotation_files.append(ann_file)
            
            if annotation_files:
                total_images = 0
                for split_name in ["train", "val", "test"]:
                    split_images_dir = self.coco_dir / split_name / "images"
                    if split_images_dir.exists():
                        total_images += sum(1 for _ in split_images_dir.glob("*.png"))
                # Also count images in root if no split
                if not split:
                    root_images_dir = self.coco_dir / "images"
                    if root_images_dir.exists():
                        total_images += sum(1 for _ in root_images_dir.glob("*.png"))
                
                if total_images > 0:
                    print(f"  ⏭️  Skipping: COCO dataset already exists ({len(annotation_files)} annotation files, {total_images} images) in {self.coco_dir}")
                    return
        
        # Convert slippy map to flat structure with images/ and masks/ subdirectories
        temp_flat_dir = self.work_dir / "flat"
        temp_flat_images = temp_flat_dir / "images"
        temp_flat_masks = temp_flat_dir / "masks"
        temp_flat_images.mkdir(parents=True, exist_ok=True)
        temp_flat_masks.mkdir(parents=True, exist_ok=True)
        
        print("  Converting slippy map to flat structure...")
        
        # Get all tiles from images
        image_tiles = list(tiles_from_slippy_map(images_dir))
        mask_tiles = list(tiles_from_slippy_map(masks_dir))
        
        # Create mapping from tile to paths
        mask_dict = {tile: path for tile, path in mask_tiles}
        
        copied = 0
        for tile, image_path in image_tiles:
            if tile not in mask_dict:
                continue
            
            # Create flat filenames: z_x_y.png
            flat_name = f"{tile.z}_{tile.x}_{tile.y}.png"
            flat_image_path = temp_flat_images / flat_name
            flat_mask_path = temp_flat_masks / flat_name
            
            # Copy files directly to final structure
            shutil.copy2(image_path, flat_image_path)
            shutil.copy2(mask_dict[tile], flat_mask_path)
            copied += 1
        
        print(f"  ✅ Copied {copied} image-mask pairs")
        
        if split:
            print("  Splitting dataset...")
            split_dataset(
                input_dir=str(temp_flat_dir),
                output_dir=str(self.coco_dir),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        else:
            # No split, just copy to coco_dir
            coco_images_dir = self.coco_dir / "images"
            coco_masks_dir = self.coco_dir / "masks"
            coco_images_dir.mkdir(parents=True, exist_ok=True)
            coco_masks_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copytree(temp_flat_images, coco_images_dir, dirs_exist_ok=True)
            shutil.copytree(temp_flat_masks, coco_masks_dir, dirs_exist_ok=True)
        
        # Export to COCO format
        exporter = COCOExporter()
        
        if split:
            for split_name in ["train", "val", "test"]:
                split_dir = self.coco_dir / split_name
                if split_dir.exists():
                    # Create annotation file inside the split directory
                    output_json = split_dir / "_annotations.coco.json"
                    exporter.export_split(str(split_dir), str(output_json), split_name)
        else:
            # No split: create annotation file in coco_dir
            output_json = self.coco_dir / "_annotations.coco.json"
            exporter.export_split(str(self.coco_dir), str(output_json), split_name="all")
        
        print(f"  ✅ COCO dataset created in {self.coco_dir}")
    
    def run_full_pipeline(self, osm_file: str, mapbox_token: Optional[str] = None, url_template: Optional[str] = None, skip_download: bool = False, skip_rasterize: bool = False, enable_sam: bool = False, sam_model: str = "vit_b", sam_device: str = "cuda", use_sam2: bool = False, use_sam3: bool = False):
        """Run full pipeline from OSM to COCO."""
        print("=" * 60)
        print("GeoSeg to COCO Dataset Preparation")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Work directory: {self.work_dir}")
        print(f"Feature types: {', '.join(self.feature_types)}")
        print(f"Zoom level: {self.zoom}")
        if self.test_tiles is not None:
            print(f"Test mode: Limited to {self.test_tiles} tiles")
        print()
        
        # Track what was processed vs skipped
        processed_steps = []
        skipped_steps = []
        
        # Step 1: Extract features
        if len(self.feature_types) > 1:
            # Use multi-type extraction (more efficient - single pass)
            existing_before = list(self.geojson_dir.glob("merged_features-*.geojson"))
            merged_geojson = self.extract_features_multi(osm_file, self.feature_types)
            existing_after = list(self.geojson_dir.glob("merged_features-*.geojson"))
            if existing_before and len(existing_after) == len(existing_before):
                skipped_steps.append(f"Extract {', '.join(self.feature_types)} (multi-type)")
            else:
                processed_steps.append(f"Extract {', '.join(self.feature_types)} (multi-type)")
            geojson_for_tiles = merged_geojson
            geojson_for_rasterize = merged_geojson
        else:
            # Single type extraction
            feature_type = self.feature_types[0]
            existing_before = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            geojson_file = self.extract_features(osm_file, feature_type)
            existing_after = list(self.geojson_dir.glob(f"{feature_type}-*.geojson"))
            if existing_before and len(existing_after) == len(existing_before):
                skipped_steps.append(f"Extract {feature_type}")
            else:
                processed_steps.append(f"Extract {feature_type}")
            geojson_for_tiles = geojson_file
            geojson_for_rasterize = geojson_file
        
        # Step 2: Generate tiles (use merged geojson to cover all features)
        tiles_file_before = self.tiles_dir / "tiles.csv"
        tiles_file = self.generate_tiles(geojson_for_tiles)
        if tiles_file_before.exists() and tiles_file_before.stat().st_size > 0:
            skipped_steps.append("Generate tiles")
        else:
            processed_steps.append("Generate tiles")
        
        # Step 3: Download images
        if not skip_download:
            zoom_dir_before = self.images_dir / str(self.zoom)
            had_images = zoom_dir_before.exists() and any(zoom_dir_before.iterdir())
            self.download_images(tiles_file, mapbox_token, url_template)
            if had_images:
                skipped_steps.append("Download images")
            else:
                processed_steps.append("Download images")
        else:
            skipped_steps.append("Download images (skipped by flag)")
        
        # Step 4: Rasterize masks (with multi-class support)
        if not skip_rasterize:
            zoom_mask_dir_before = self.masks_dir / str(self.zoom)
            had_masks = zoom_mask_dir_before.exists() and any(zoom_mask_dir_before.iterdir())
            # Use merged geojson for multi-class rasterization
            self.rasterize_masks(geojson_for_rasterize, tiles_file, use_multiclass=len(self.feature_types) > 1)
            if had_masks:
                skipped_steps.append("Rasterize masks")
            else:
                processed_steps.append("Rasterize masks")
        else:
            skipped_steps.append("Rasterize masks (skipped by flag)")
        
        # Determine which masks to use for COCO conversion
        final_masks_dir = self.masks_dir
        
        # Step 5: SAM Annotation (optional)
        if enable_sam and SAM_AVAILABLE:
            if use_sam3:
                sam_version = "SAM 3"
            elif use_sam2:
                sam_version = "SAM 2"
            else:
                sam_version = "SAM"
            sam_masks_before = sum(1 for _ in self.sam_masks_dir.rglob("*.png")) if self.sam_masks_dir.exists() else 0
            self.annotate_with_sam(model_type=sam_model, device=sam_device, use_sam2=use_sam2, use_sam3=use_sam3)
            sam_masks_after = sum(1 for _ in self.sam_masks_dir.rglob("*.png")) if self.sam_masks_dir.exists() else 0
            if sam_masks_before > 0 and sam_masks_after == sam_masks_before:
                skipped_steps.append(f"{sam_version} annotation")
            else:
                processed_steps.append(f"{sam_version} annotation")
            
            # Step 6: Merge masks
            merged_masks_before = sum(1 for _ in self.merged_masks_dir.rglob("*.png")) if self.merged_masks_dir.exists() else 0
            self.merge_masks()
            merged_masks_after = sum(1 for _ in self.merged_masks_dir.rglob("*.png")) if self.merged_masks_dir.exists() else 0
            if merged_masks_before > 0 and merged_masks_after == merged_masks_before:
                skipped_steps.append("Merge masks")
            else:
                processed_steps.append("Merge masks")
            
            # Use merged masks for COCO conversion
            final_masks_dir = self.merged_masks_dir
        elif enable_sam and not SAM_AVAILABLE:
            skipped_steps.append("SAM annotation (not installed)")
            skipped_steps.append("Merge masks (SAM not available)")
        
        # Step 7: Convert to COCO
        coco_existed = self.coco_dir.exists() and any(self.coco_dir.glob("annotations*.json"))
        self.convert_to_coco(str(self.images_dir), str(final_masks_dir))
        if coco_existed:
            skipped_steps.append("Convert to COCO")
        else:
            processed_steps.append("Convert to COCO")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ Pipeline completed!")
        print("=" * 60)
        if processed_steps:
            print(f"\n📝 Processed steps ({len(processed_steps)}):")
            for step in processed_steps:
                print(f"   ✓ {step}")
        if skipped_steps:
            print(f"\n⏭️  Skipped steps ({len(skipped_steps)}):")
            for step in skipped_steps:
                print(f"   ⊘ {step}")
        print(f"\nOutput directory: {self.coco_dir}")
        print(f"Work directory: {self.work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoSeg workflow to COCO format"
    )
    parser.add_argument(
        "--osm-file",
        type=str,
        required=True,
        help="Path to .osm.pbf file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for COCO dataset"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Zoom level for tiles (default: 19)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in pixels (default: 512)"
    )
    parser.add_argument(
        "--feature-types",
        dest="feature_types",
        type=str,
        nargs="+",
        default=["building"],
        choices=["building", "road", "parking", "all"],
        help="Feature types to extract (default: building). Use 'all' to extract all types. Can use --feature-types or --feature_types"
    )
    parser.add_argument(
        "--feature_types",
        dest="feature_types",
        type=str,
        nargs="+",
        choices=["building", "road", "parking", "all"],
        help=argparse.SUPPRESS  # Hide from help since it's an alias
    )
    parser.add_argument(
        "--mapbox-token",
        type=str,
        default=os.environ.get("MAPBOX_API_KEY"),
        help="Mapbox API token (optional, uses OSM tiles if not provided). Can also be set via MAPBOX_API_KEY environment variable."
    )
    parser.add_argument(
        "--url-template",
        type=str,
        default=None,
        help="Custom URL template for tile download"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip image download step"
    )
    parser.add_argument(
        "--skip-rasterize",
        action="store_true",
        help="Skip mask rasterization step"
    )
    parser.add_argument(
        "--no-split",
        dest="split",
        action="store_false",
        help="Don't split into train/val/test"
    )
    parser.add_argument(
        "--test-tiles",
        type=int,
        default=None,
        help="Limit number of tiles for quick testing (e.g., --test-tiles 20)"
    )
    parser.add_argument(
        "--enable-sam",
        action="store_true",
        help="Enable SAM annotation to detect trees and grass. Requires segment-anything or sam2 package."
    )
    parser.add_argument(
        "--use-sam2",
        action="store_true",
        help="Use SAM 2 instead of SAM 1 for better tree/grass detection. Requires sam2 package."
    )
    parser.add_argument(
        "--use-sam3",
        action="store_true",
        help="Use SAM 3 (text-prompted segmentation) for best tree/grass detection. Requires SAM3 to be installed."
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b", "tiny", "small", "base_plus", "large"],
        help="SAM model type. SAM1: vit_h/vit_l/vit_b. SAM2: tiny/small/base_plus/large. Not used for SAM3. Default: vit_b"
    )
    parser.add_argument(
        "--sam-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run SAM on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    converter = GeoSegToCOCO(
        output_dir=args.output,
        zoom=args.zoom,
        tile_size=args.tile_size,
        feature_types=args.feature_types,
        test_tiles=args.test_tiles
    )
    
    converter.run_full_pipeline(
        osm_file=args.osm_file,
        mapbox_token=args.mapbox_token,
        url_template=args.url_template,
        skip_download=args.skip_download,
        skip_rasterize=args.skip_rasterize,
        enable_sam=args.enable_sam,
        sam_model=args.sam_model,
        sam_device=args.sam_device,
        use_sam2=args.use_sam2,
        use_sam3=args.use_sam3
    )
    
    # Explicitly exit to ensure clean termination
    sys.exit(0)


if __name__ == "__main__":
    main()

