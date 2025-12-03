"""
SAM 2 (Segment Anything Model 2) annotator for detecting trees and grass.
Uses the improved SAM 2 architecture for better segmentation quality.

SAM 2 features:
- Improved mask quality
- Better handling of complex scenes
- More efficient architecture

Installation:
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict
import urllib.request

# Color mapping for SAM2 mask visualization (RGB values)
SAM2_CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    5: (0, 128, 0),      # Tree - Dark Green
    6: (144, 238, 144),  # Grass - Light Green
}

# Full color palette including OSM classes
FULL_CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building - Red (OSM)
    2: (255, 255, 0),    # Road - Yellow (OSM)
    3: (255, 0, 255),    # Parking - Magenta (OSM)
    4: (0, 0, 255),      # Water - Blue (OSM)
    5: (0, 128, 0),      # Tree - Dark Green (SAM2)
    6: (144, 238, 144),  # Grass - Light Green (SAM2)
}


class SAM2Annotator:
    """
    Uses SAM 2 to automatically detect and annotate trees and grass.
    
    SAM 2 provides better segmentation quality than the original SAM,
    with improved handling of complex natural scenes.
    """
    
    def __init__(
        self,
        model_size: str = "large",
        output_dir: str = "./data/processed/masks",
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize SAM 2 annotator.
        
        Args:
            model_size: SAM 2 model size ('tiny', 'small', 'base_plus', 'large')
            output_dir: Directory to save SAM masks
            device: Device to run SAM on ('cuda' or 'cpu')
            checkpoint_dir: Directory containing SAM 2 checkpoints
        """
        self.model_size = model_size
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "checkpoints", "sam2"
            )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.mask_generator = None
        self._model_loaded = False
    
    def load_model(self):
        """Load SAM 2 model (lazy loading)."""
        if self._model_loaded:
            return
        self._load_model()
        self._model_loaded = True
    
    def _download_checkpoint(self, url: str, filepath: str):
        """Download SAM 2 checkpoint with progress."""
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\r  Downloading: {percent:.1f}%", end='', flush=True)
        
        print(f"  📥 Downloading SAM 2 checkpoint...")
        print(f"     URL: {url}")
        print(f"     Save to: {filepath}")
        
        try:
            urllib.request.urlretrieve(url, filepath, show_progress)
            print(f"\n  ✅ SAM 2 checkpoint downloaded!")
        except Exception as e:
            print(f"\n  ❌ Error downloading SAM 2 checkpoint: {e}")
            raise
    
    def _load_model(self):
        """Load SAM 2 model with automatic mask generation."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with:\n"
                "  pip install torch torchvision"
            )
        
        # Try to import SAM 2
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM 2 not installed. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        # SAM 2 checkpoint mapping - try SAM 2.0 first (more widely installed)
        checkpoint_map = {
            "tiny": {
                "filename": "sam2_hiera_tiny.pt",
                "configs": ["sam2_hiera_t.yaml", "sam2.1_hiera_t.yaml"],
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
            },
            "small": {
                "filename": "sam2_hiera_small.pt",
                "configs": ["sam2_hiera_s.yaml", "sam2.1_hiera_s.yaml"],
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
            },
            "base_plus": {
                "filename": "sam2_hiera_base_plus.pt",
                "configs": ["sam2_hiera_b+.yaml", "sam2.1_hiera_b+.yaml"],
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
            },
            "large": {
                "filename": "sam2_hiera_large.pt",
                "configs": ["sam2_hiera_l.yaml", "sam2.1_hiera_l.yaml"],
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
            }
        }
        
        if self.model_size not in checkpoint_map:
            raise ValueError(f"Invalid model_size '{self.model_size}'. Choose from: {list(checkpoint_map.keys())}")
        
        checkpoint_info = checkpoint_map[self.model_size]
        checkpoint_filename = checkpoint_info["filename"]
        checkpoint_url = checkpoint_info["url"]
        config_options = checkpoint_info["configs"]
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  SAM 2 checkpoint not found: {checkpoint_path}")
            self._download_checkpoint(checkpoint_url, checkpoint_path)
        
        print(f"  🔄 Loading SAM 2 model: {self.model_size}")
        device_str = self.device if torch.cuda.is_available() else "cpu"
        
        # Try each config option until one works
        sam2_model = None
        last_error = None
        
        for config_name in config_options:
            try:
                sam2_model = build_sam2(
                    config_name, 
                    checkpoint_path, 
                    device=device_str,
                    apply_postprocessing=False
                )
                print(f"  ✅ Loaded with config: {config_name}")
                break
            except TypeError:
                # Older API without apply_postprocessing
                try:
                    sam2_model = build_sam2(config_name, checkpoint_path, device=device_str)
                    print(f"  ✅ Loaded with config: {config_name}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            except Exception as e:
                last_error = e
                continue
        
        if sam2_model is None:
            raise RuntimeError(
                f"Failed to load SAM 2 model with any config.\n"
                f"Tried configs: {config_options}\n"
                f"Last error: {last_error}\n"
                f"Checkpoint: {checkpoint_path}\n"
                "Try reinstalling SAM 2:\n"
                "  pip uninstall sam2 -y\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        # Create automatic mask generator optimized for vegetation
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            min_mask_region_area=100,
        )
        
        print(f"  ✅ SAM 2 model loaded on {device}")
    
    def annotate_image(
        self,
        image_path: str,
        existing_mask_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Detect and annotate trees and grass in an image.
        
        Args:
            image_path: Path to input image
            existing_mask_path: Path to existing OSM mask (to exclude labeled areas)
            output_path: Path to save SAM mask
        
        Returns:
            Path to saved SAM mask
        """
        # Lazy load model
        if not self._model_loaded:
            self.load_model()
        
        image = np.array(Image.open(image_path).convert('RGB'))
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_sam2_mask.png")
        
        mask = self._detect_vegetation(image, existing_mask_path)
        
        # Save mask as colored RGB image
        mask = mask.astype(np.uint8)
        height, width = mask.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, color in SAM2_CLASS_COLORS.items():
            mask_pixels = (mask == class_id)
            colored_image[mask_pixels] = color
        
        mask_image = Image.fromarray(colored_image, mode='RGB')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_image.save(output_path)
        
        return output_path
    
    def _classify_as_tree_or_grass(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mask_data: Dict
    ) -> int:
        """
        Classify a vegetation mask as tree (5) or grass (6).
        
        Uses multiple features:
        - Color intensity and variance
        - Texture analysis
        - Shape compactness
        - Mask area and aspect ratio
        
        Args:
            image: Original RGB image
            mask: Binary mask of the region
            mask_data: SAM mask metadata (area, bbox, etc.)
            
        Returns:
            Class ID: 5 for tree, 6 for grass
        """
        # Get pixels within the mask
        mask_pixels = image[mask]
        if len(mask_pixels) == 0:
            return 6  # Default to grass
        
        # Extract features
        avg_color = np.mean(mask_pixels, axis=0)  # [R, G, B]
        color_std = np.std(mask_pixels, axis=0)
        
        # Calculate brightness and green dominance
        brightness = np.mean(avg_color)
        green_dominance = avg_color[1] / (np.sum(avg_color) + 1e-6)
        
        # Texture variance (higher = more complex texture like leaves)
        texture_variance = np.mean(color_std)
        
        # Get mask properties from SAM data
        area = mask_data.get('area', np.sum(mask))
        bbox = mask_data.get('bbox', None)
        
        # Calculate compactness
        if bbox is not None:
            bbox_area = bbox[2] * bbox[3]  # width * height
            compactness = area / (bbox_area + 1e-6)
            aspect_ratio = max(bbox[2], bbox[3]) / (min(bbox[2], bbox[3]) + 1e-6)
        else:
            # Calculate from mask directly
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox_h = rmax - rmin + 1
                bbox_w = cmax - cmin + 1
                bbox_area = bbox_h * bbox_w
                compactness = area / (bbox_area + 1e-6)
                aspect_ratio = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-6)
            else:
                compactness = 0.5
                aspect_ratio = 1.0
        
        # Classification scoring
        tree_score = 0
        
        # Trees tend to be darker (shadows, dense foliage)
        if brightness < 90:
            tree_score += 3
        elif brightness < 110:
            tree_score += 2
        elif brightness < 130:
            tree_score += 1
        elif brightness > 160:
            tree_score -= 1  # Very bright = likely grass in sunlight
        
        # Trees have higher texture variance (leaves, branches)
        if texture_variance > 35:
            tree_score += 3
        elif texture_variance > 25:
            tree_score += 2
        elif texture_variance > 15:
            tree_score += 1
        elif texture_variance < 10:
            tree_score -= 1  # Very uniform = grass
        
        # Trees are more compact (round canopy)
        if compactness > 0.6:
            tree_score += 2
        elif compactness > 0.4:
            tree_score += 1
        elif compactness < 0.2:
            tree_score -= 1  # Very spread out = grass
        
        # Trees tend to be smaller individual areas
        if area < 3000:
            tree_score += 2
        elif area < 8000:
            tree_score += 1
        elif area > 30000:
            tree_score -= 2  # Very large areas = grass fields
        elif area > 15000:
            tree_score -= 1
        
        # Aspect ratio close to 1 = tree (round), elongated = grass
        if aspect_ratio < 1.5:
            tree_score += 1
        elif aspect_ratio > 3:
            tree_score -= 1
        
        # Final classification
        if tree_score >= 4:
            return 5  # Tree
        else:
            return 6  # Grass
    
    def _detect_vegetation(
        self,
        image: np.ndarray,
        existing_mask_path: Optional[str]
    ) -> np.ndarray:
        """
        Detect trees and grass in the image.
        
        Args:
            image: Input image array (RGB, uint8)
            existing_mask_path: Path to existing OSM mask (to exclude)
        
        Returns:
            Mask array with class 5 (Tree) and class 6 (Grass)
        """
        # Generate all masks using SAM 2
        masks = self.mask_generator.generate(image)
        
        # Load existing OSM mask to exclude already-labeled areas
        exclude_mask = None
        if existing_mask_path and os.path.exists(existing_mask_path):
            exclude_img = Image.open(existing_mask_path)
            exclude_arr = np.array(exclude_img)
            
            # Handle both RGB and grayscale masks
            if len(exclude_arr.shape) == 3:
                exclude_mask = np.any(exclude_arr > 10, axis=2)
            else:
                exclude_mask = exclude_arr > 0
            
            # Resize if needed
            if exclude_mask.shape != image.shape[:2]:
                exclude_img_resized = Image.fromarray(exclude_mask.astype(np.uint8) * 255).resize(
                    (image.shape[1], image.shape[0]),
                    Image.Resampling.NEAREST
                )
                exclude_mask = np.array(exclude_img_resized) > 0
        
        output_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        tree_count = 0
        grass_count = 0
        
        # Sort masks by area (larger first) for better overlap handling
        masks_sorted = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
        
        for mask_data in masks_sorted:
            mask = mask_data['segmentation']
            score = mask_data.get('predicted_iou', 0)
            stability = mask_data.get('stability_score', 0)
            
            # Quality filter
            if score < 0.65 or stability < 0.65:
                continue
            
            # Skip if overlaps too much with existing OSM mask
            if exclude_mask is not None:
                overlap = np.sum(mask & exclude_mask)
                total = np.sum(mask)
                if total > 0 and overlap / total > 0.3:
                    continue
            
            # Check if this region looks like vegetation
            mask_pixels = image[mask]
            if len(mask_pixels) == 0:
                continue
            
            avg_color = np.mean(mask_pixels, axis=0)
            
            # Check for greenish color
            green_ratio = avg_color[1] / (np.sum(avg_color) + 1e-6)
            
            # Also check that green is dominant over red and blue
            is_greenish = (
                avg_color[1] > avg_color[0] * 0.85 and  # Green > Red
                avg_color[1] > avg_color[2] * 0.85 and  # Green > Blue
                avg_color[1] > 40  # Minimum green intensity
            )
            
            # Accept as vegetation
            if green_ratio > 0.32 and is_greenish:
                # Classify as tree or grass
                class_id = self._classify_as_tree_or_grass(image, mask, mask_data)
                
                # Only fill background pixels (don't overwrite existing detections)
                output_mask[mask & (output_mask == 0)] = class_id
                
                if class_id == 5:
                    tree_count += 1
                else:
                    grass_count += 1
        
        # Post-processing: remove very small isolated regions
        self._remove_small_regions(output_mask, min_area=25)
        
        return output_mask
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int = 25):
        """Remove small isolated regions from the mask (in-place)."""
        try:
            from scipy import ndimage
            
            for class_id in [5, 6]:
                class_mask = (mask == class_id)
                if not np.any(class_mask):
                    continue
                
                labeled, num_features = ndimage.label(class_mask)
                for label_id in range(1, num_features + 1):
                    component = (labeled == label_id)
                    area = np.sum(component)
                    if area < min_area:
                        mask[component] = 0
        except ImportError:
            pass  # Skip if scipy not available
    
    def annotate_batch(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Annotate multiple images in batch.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing existing OSM masks (optional)
            output_dir: Output directory for SAM masks
            
        Returns:
            List of output mask paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        output_paths = []
        for img_file in image_files:
            image_path = os.path.join(image_dir, img_file)
            mask_path = None
            
            if mask_dir:
                base_name = os.path.splitext(img_file)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask = os.path.join(mask_dir, f"{base_name}{ext}")
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
            
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.png")
            result_path = self.annotate_image(image_path, mask_path, output_path=output_path)
            output_paths.append(result_path)
        
        return output_paths
