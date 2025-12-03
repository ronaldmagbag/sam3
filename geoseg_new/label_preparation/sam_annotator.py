"""
Apply Segment Anything Model (SAM) to annotate missing classes.
Detects trees and grass in satellite imagery.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import urllib.request

# Color mapping for SAM mask visualization (RGB values)
# Background matches OSM masks, Tree and Grass have distinct visible colors
SAM_CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black (same as OSM)
    5: (0, 128, 0),      # Tree - Dark Green
    6: (144, 238, 144),  # Grass - Light Green
}

# Full color palette including OSM classes (for reference/merging)
FULL_CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building - Red (OSM)
    2: (255, 255, 0),    # Road - Yellow (OSM)
    3: (255, 0, 255),    # Parking - Magenta (OSM)
    4: (0, 0, 255),      # Water - Blue (OSM)
    5: (0, 128, 0),      # Tree - Dark Green (SAM)
    6: (144, 238, 144),  # Grass - Light Green (SAM)
}


class SAMAnnotator:
    """Uses SAM to automatically annotate trees and grass in satellite imagery."""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        output_dir: str = "./data/processed/masks",
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize SAM annotator.
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            output_dir: Directory to save SAM masks
            device: Device to run SAM on ('cuda' or 'cpu')
            checkpoint_dir: Directory containing SAM checkpoints
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "checkpoints", "sam"
            )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.mask_generator = None
        self._model_loaded = False
    
    def load_model(self):
        """Load SAM model (lazy loading)."""
        if self._model_loaded:
            return
        self._load_model()
        self._model_loaded = True
    
    def _download_checkpoint(self, url: str, filepath: str):
        """Download checkpoint with progress."""
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\r  Downloading: {percent:.1f}%", end='', flush=True)
        
        print(f"  📥 Downloading SAM checkpoint...")
        print(f"     URL: {url}")
        print(f"     Save to: {filepath}")
        
        try:
            urllib.request.urlretrieve(url, filepath, show_progress)
            print(f"\n  ✅ Checkpoint downloaded!")
        except Exception as e:
            print(f"\n  ❌ Error downloading checkpoint: {e}")
            raise
    
    def _load_model(self):
        """Load SAM model with automatic mask generation."""
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            checkpoint_map = {
                "vit_h": {
                    "filename": "sam_vit_h_4b8939.pth",
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                },
                "vit_l": {
                    "filename": "sam_vit_l_0b3195.pth",
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
                },
                "vit_b": {
                    "filename": "sam_vit_b_01ec64.pth",
                    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                }
            }
            
            checkpoint_info = checkpoint_map.get(self.model_type, checkpoint_map["vit_h"])
            checkpoint_filename = checkpoint_info["filename"]
            checkpoint_url = checkpoint_info["url"]
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            if not os.path.exists(checkpoint_path):
                print(f"  ⚠️  SAM checkpoint not found: {checkpoint_path}")
                self._download_checkpoint(checkpoint_url, checkpoint_path)
            
            print(f"  🔄 Loading SAM model: {self.model_type}")
            device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            
            sam_model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam_model.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            print(f"  ✅ SAM model loaded on {device}")
            
        except ImportError:
            print("  ⚠️  segment-anything not installed, SAM features disabled")
            print("     Install with: pip install segment-anything")
            self.mask_generator = None
        except Exception as e:
            print(f"  ⚠️  Error loading SAM model: {e}")
            self.mask_generator = None
    
    def annotate_image(
        self,
        image_path: str,
        existing_mask_path: Optional[str] = None,
        target_classes: List[str] = ['tree', 'grass'],
        output_path: Optional[str] = None
    ) -> str:
        """
        Annotate trees and grass in an image using SAM.
        
        Args:
            image_path: Path to input image
            existing_mask_path: Path to existing OSM mask (to exclude labeled areas)
            target_classes: List of classes to annotate ('tree', 'grass')
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
            output_path = os.path.join(self.output_dir, f"{base_name}_sam_mask.png")
        
        if self.mask_generator is None:
            # Return empty mask if SAM not available
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = self._run_sam(image, existing_mask_path, target_classes)
        
        # Save mask as colored RGB image with distinct colors
        mask = mask.astype(np.uint8)
        height, width = mask.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colors for each class
        for class_id, color in SAM_CLASS_COLORS.items():
            mask_pixels = (mask == class_id)
            colored_image[mask_pixels] = color
        
        mask_image = Image.fromarray(colored_image, mode='RGB')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_image.save(output_path)
        
        return output_path
    
    def _classify_vegetation(self, image: np.ndarray, mask: np.ndarray) -> int:
        """
        Classify a vegetation mask as tree (5) or grass (6).
        
        Uses heuristics based on:
        - Color intensity (darker green = tree, lighter = grass)
        - Texture variance (higher variance = tree canopy)
        - Shape compactness (compact = tree, spread = grass)
        
        Args:
            image: Original RGB image
            mask: Binary mask of the region
            
        Returns:
            Class ID: 5 for tree, 6 for grass
        """
        # Get pixels within the mask
        mask_area = image[mask]
        if len(mask_area) == 0:
            return 6  # Default to grass
        
        # Calculate average color
        avg_color = np.mean(mask_area, axis=0)  # [R, G, B]
        
        # Calculate color variance (texture indicator)
        color_std = np.std(mask_area, axis=0)
        texture_variance = np.mean(color_std)
        
        # Calculate mask properties
        area = np.sum(mask)
        
        # Find bounding box for compactness
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 6
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        compactness = area / (bbox_area + 1e-6)
        
        # Classification heuristics:
        # Trees tend to be:
        # - Darker green (lower brightness)
        # - Higher texture variance (leaves, shadows)
        # - More compact shape
        
        # Grass tends to be:
        # - Lighter, more uniform green
        # - Lower texture variance
        # - Spread out, less compact
        
        brightness = np.mean(avg_color)
        green_dominance = avg_color[1] / (np.sum(avg_color) + 1e-6)
        
        # Score for tree classification
        tree_score = 0
        
        # Darker regions more likely to be trees
        if brightness < 100:
            tree_score += 2
        elif brightness < 130:
            tree_score += 1
        
        # High texture variance suggests tree canopy
        if texture_variance > 30:
            tree_score += 2
        elif texture_variance > 20:
            tree_score += 1
        
        # Compact shapes more likely to be trees
        if compactness > 0.5:
            tree_score += 1
        
        # Small to medium areas more likely to be trees
        if area < 5000:
            tree_score += 1
        elif area > 20000:
            tree_score -= 1  # Very large areas more likely grass
        
        # Classify based on score
        if tree_score >= 3:
            return 5  # Tree
        else:
            return 6  # Grass
    
    def _run_sam(
        self,
        image: np.ndarray,
        existing_mask_path: Optional[str],
        target_classes: List[str]
    ) -> np.ndarray:
        """
        Run SAM model on image to detect trees and grass.
        
        Args:
            image: Input image array (RGB, uint8)
            existing_mask_path: Path to existing OSM mask (to exclude)
            target_classes: Target classes ('tree', 'grass')
        
        Returns:
            Mask array with class 5 (Tree) and class 6 (Grass)
        """
        # Generate all masks using SAM
        masks = self.mask_generator.generate(image)
        
        # Load existing OSM mask to exclude already-labeled areas
        exclude_mask = None
        if existing_mask_path and os.path.exists(existing_mask_path):
            exclude_img = Image.open(existing_mask_path)
            exclude_arr = np.array(exclude_img)
            
            # Handle both RGB and grayscale masks
            if len(exclude_arr.shape) == 3:
                # RGB mask - check for non-black pixels
                exclude_mask = np.any(exclude_arr > 10, axis=2)
            else:
                # Grayscale/palette mask
                exclude_mask = exclude_arr > 0
            
            # Resize if needed
            if exclude_mask.shape != image.shape[:2]:
                exclude_img_resized = Image.fromarray(exclude_mask.astype(np.uint8) * 255).resize(
                    (image.shape[1], image.shape[0]),
                    Image.Resampling.NEAREST
                )
                exclude_mask = np.array(exclude_img_resized) > 0
        
        output_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Process each SAM mask
        tree_count = 0
        grass_count = 0
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            score = mask_data.get('predicted_iou', 0)
            stability = mask_data.get('stability_score', 0)
            
            # Filter by quality
            if score < 0.7 or stability < 0.7:
                continue
            
            # Skip if overlaps too much with existing OSM mask
            if exclude_mask is not None:
                overlap = np.sum(mask & exclude_mask)
                total = np.sum(mask)
                if total > 0 and overlap / total > 0.3:
                    continue
            
            # Check if mask looks like vegetation (greenish colors)
            mask_pixels = image[mask]
            if len(mask_pixels) == 0:
                continue
            
            avg_color = np.mean(mask_pixels, axis=0)
            
            # Green channel should be dominant for vegetation
            green_ratio = avg_color[1] / (np.sum(avg_color) + 1e-6)
            
            # Also check that it's actually greenish (not gray/brown)
            is_greenish = (avg_color[1] > avg_color[0] * 0.9 and 
                          avg_color[1] > avg_color[2] * 0.9 and
                          avg_color[1] > 50)
            
            if green_ratio > 0.33 and is_greenish:
                # Classify as tree or grass
                class_id = self._classify_vegetation(image, mask)
                
                # Only overwrite background (0) pixels
                output_mask[mask & (output_mask == 0)] = class_id
                
                if class_id == 5:
                    tree_count += 1
                else:
                    grass_count += 1
        
        # Remove very small isolated regions
        try:
            from scipy import ndimage
            
            for class_id in [5, 6]:
                class_mask = (output_mask == class_id)
                if not np.any(class_mask):
                    continue
                    
                labeled, num_features = ndimage.label(class_mask)
                for label_id in range(1, num_features + 1):
                    component = (labeled == label_id)
                    area = np.sum(component)
                    # Remove very small regions (noise)
                    if area < 30:
                        output_mask[component] = 0
        except ImportError:
            pass
        
        return output_mask
    
    def annotate_batch(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Annotate multiple images in batch."""
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
                # Try to find matching mask
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
