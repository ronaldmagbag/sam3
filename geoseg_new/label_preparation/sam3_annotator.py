"""
Apply Segment Anything Model 3 (SAM3) to annotate missing classes.
Detects trees and grass in satellite imagery using text prompts.
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict
from pathlib import Path

# Monkey-patch to filter out SAM2-specific keys from checkpoint loading warnings
try:
    from sam3.model_builder import _load_checkpoint as original_load_checkpoint
    from iopath.common.file_io import g_pathmgr
    import torch
    
    def patched_load_checkpoint(model, checkpoint_path):
        """Patched version that filters out known SAM2-specific keys from warnings."""
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        
        # Check if checkpoint uses HuggingFace format (has "detector." prefix)
        has_detector_prefix = any("detector" in k for k in ckpt.keys())
        
        if has_detector_prefix:
            # HuggingFace format: remove "detector." prefix
            sam3_image_ckpt = {
                k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
            }
            # Handle tracker keys if present
            if model.inst_interactive_predictor is not None:
                sam3_image_ckpt.update(
                    {
                        k.replace("tracker.", "inst_interactive_predictor.model."): v
                        for k, v in ckpt.items()
                        if "tracker" in k
                    }
                )
        else:
            # Training checkpoint format: use keys as-is (no prefix)
            sam3_image_ckpt = dict(ckpt)
            # Handle tracker keys if present
            if model.inst_interactive_predictor is not None:
                tracker_keys = {k: v for k, v in ckpt.items() if "tracker" in k}
                if tracker_keys:
                    sam3_image_ckpt.update(
                        {
                            k.replace("tracker.", "inst_interactive_predictor.model."): v
                            for k, v in tracker_keys.items()
                        }
                    )
        
        missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False)
        
        # Filter out known SAM2-specific keys (these are harmless and expected)
        sam2_key_patterns = [
            "sam2_convs",  # SAM2-specific convolutions
            "dconv_2x2",   # SAM2 dilated convolutions
        ]
        filtered_unexpected = [
            k for k in unexpected_keys 
            if not any(pattern in k for pattern in sam2_key_patterns)
        ]
        
        # Only print warnings for truly unexpected keys (not SAM2-specific)
        if len(missing_keys) > 0:
            print(
                f"loaded {checkpoint_path} and found "
                f"missing keys:\n{missing_keys=}"
            )
        if len(filtered_unexpected) > 0 and len(filtered_unexpected) < 50:
            print(f"unexpected_keys (not loaded): {filtered_unexpected}")
        elif len(unexpected_keys) > len(filtered_unexpected):
            # Only show count if we filtered out SAM2 keys
            sam2_filtered_count = len(unexpected_keys) - len(filtered_unexpected)
            if sam2_filtered_count > 0:
                print(f"Note: {sam2_filtered_count} SAM2-specific keys were filtered (expected, harmless)")
    
    # Apply monkey patch
    import sam3.model_builder
    sam3.model_builder._load_checkpoint = patched_load_checkpoint
except (ImportError, AttributeError):
    # If patching fails, continue without it (checkpoint loading will use default behavior)
    pass

# Color mapping for SAM3 mask visualization (RGB values)
# Background matches OSM masks, Tree and Grass have distinct visible colors
SAM3_CLASS_COLORS = {
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
    5: (0, 128, 0),      # Tree - Dark Green (SAM3)
    6: (144, 238, 144),  # Grass - Light Green (SAM3)
}


class SAM3Annotator:
    """Uses SAM3 text-prompted segmentation to automatically annotate trees and grass."""
    
    def __init__(
        self,
        output_dir: str = "./data/processed/masks",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        sam3_path: Optional[str] = None
    ):
        """
        Initialize SAM3 annotator.
        
        Args:
            output_dir: Directory to save SAM3 masks
            device: Device to run SAM3 on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
            sam3_path: Optional path to SAM3 project directory (only needed if not installed via pip)
        """
        self.output_dir = output_dir
        self.device = device
        self.confidence_threshold = confidence_threshold
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to import SAM3 directly (if installed via pip)
        # Only add to path if direct import fails and sam3_path is provided
        self._setup_sam3_path(sam3_path)
        
        self.sam3_model = None
        self.sam3_processor = None
        self._model_loaded = False
        
        # Class-specific thresholds for better detection
        self.class_thresholds = {
            "tree": 0.3,   # Lower threshold for trees
            "grass": 0.3,  # Lower threshold for grass
        }
    
    def _setup_sam3_path(self, sam3_path: Optional[str]):
        """
        Setup SAM3 import path. Tries direct import first, falls back to manual path if needed.
        
        Args:
            sam3_path: Optional path to SAM3 project directory
        """
        # First, try to import SAM3 directly (if installed via pip)
        try:
            import sam3.model_builder
            import sam3.model.sam3_image_processor
            # If import succeeds, SAM3 is installed and we don't need to add to path
            return
        except ImportError:
            pass
        
        # If direct import fails, try to use provided path or find it
        if sam3_path is None:
            # Try to find SAM3 in common locations
            current_dir = Path(__file__).parent.parent
            possible_paths = [
                current_dir.parent,
                Path.home() / "sam3",
            ]
            
            for path in possible_paths:
                if path.exists():
                    sam3_path = str(path)
                    break
        
        # Add to path if we found one
        if sam3_path:
            sam3_path = Path(sam3_path)
            if sam3_path.exists():
                sys.path.insert(0, str(sam3_path))
                print(f"  ℹ️  Using SAM3 from: {sam3_path}")
            else:
                print(f"  ⚠️  SAM3 path not found: {sam3_path}")
                print("     Trying to import SAM3 directly (assuming pip install)...")
        else:
            print("  ℹ️  No SAM3 path provided, trying direct import (assuming pip install)...")
    
    def load_model(self):
        """Load SAM3 model (lazy loading)."""
        if self._model_loaded:
            return
        self._load_model()
        self._model_loaded = True
    
    def _load_model(self):
        """Load SAM3 model with text-prompted segmentation.
        
        Models will be automatically downloaded to checkpoints/sam3/ if not already present.
        """
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            print(f"  🔄 Loading SAM3 model on {self.device}")
            print(f"     Models will be downloaded to checkpoints/sam3/ if needed")
            
            # build_sam3_image_model() will automatically download models to checkpoints/sam3/
            self.sam3_model = build_sam3_image_model()
            self.sam3_processor = Sam3Processor(
                self.sam3_model,
                device=self.device,
                confidence_threshold=self.confidence_threshold
            )
            
            print(f"  ✅ SAM3 model loaded successfully")
            
        except ImportError as e:
            print(f"  ❌ SAM3 not available: {e}")
            print("     Install SAM3 via: pip install sam3")
            print("     Or provide sam3_path if using local installation")
            self.sam3_model = None
            self.sam3_processor = None
        except Exception as e:
            print(f"  ❌ Error loading SAM3 model: {e}")
            print("     Check that models are available or will be downloaded to checkpoints/sam3/")
            self.sam3_model = None
            self.sam3_processor = None
    
    def annotate_image(
        self,
        image_path: str,
        existing_mask_path: Optional[str] = None,
        target_classes: List[str] = ['tree', 'grass'],
        output_path: Optional[str] = None
    ) -> str:
        """
        Annotate trees and grass in an image using SAM3.
        
        Args:
            image_path: Path to input image
            existing_mask_path: Path to existing OSM mask (to exclude labeled areas)
            target_classes: List of classes to annotate ('tree', 'grass')
            output_path: Path to save SAM3 mask
        
        Returns:
            Path to saved SAM3 mask
        """
        # Lazy load model
        if not self._model_loaded:
            self.load_model()
        
        image = Image.open(image_path).convert('RGB')
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_sam3_mask.png")
        
        if self.sam3_processor is None:
            # Return empty mask if SAM3 not available
            mask = np.zeros(image.size[::-1], dtype=np.uint8)
        else:
            mask = self._run_sam3(image, image_path, existing_mask_path, target_classes)
        
        # Save mask as colored RGB image with distinct colors
        mask = mask.astype(np.uint8)
        height, width = mask.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colors for each class
        for class_id, color in SAM3_CLASS_COLORS.items():
            mask_pixels = (mask == class_id)
            colored_image[mask_pixels] = color
        
        mask_image = Image.fromarray(colored_image, mode='RGB')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_image.save(output_path)
        
        return output_path
    
    def _classify_vegetation(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        class_name: str
    ) -> int:
        """
        Classify a vegetation mask as tree (5) or grass (6).
        
        Uses heuristics based on:
        - SAM3's class prediction (if "tree" or "grass" was detected)
        - Color intensity (darker green = tree, lighter = grass)
        - Texture variance (higher variance = tree canopy)
        - Shape compactness (compact = tree, spread = grass)
        - Size (smaller = tree, larger = grass field)
        
        Args:
            image: Original RGB image
            mask: Binary mask of the region
            class_name: The class name from SAM3 detection
            
        Returns:
            Class ID: 5 for tree, 6 for grass
        """
        # If SAM3 gave us a clear class name, use it
        class_lower = class_name.lower()
        if 'tree' in class_lower:
            return 5
        if 'grass' in class_lower:
            return 6
        
        # Otherwise use heuristics
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
        
        brightness = np.mean(avg_color)
        
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
    
    def _run_sam3(
        self,
        image: Image.Image,
        image_path: str,
        existing_mask_path: Optional[str],
        target_classes: List[str]
    ) -> np.ndarray:
        """
        Run SAM3 model on image to detect trees and grass using text prompts.
        
        Args:
            image: Input PIL image (RGB)
            image_path: Path to image file
            existing_mask_path: Path to existing OSM mask (to exclude)
            target_classes: Target classes ('tree', 'grass')
        
        Returns:
            Mask array with class 5 (Tree) and class 6 (Grass)
        """
        image_np = np.array(image)
        width, height = image.size
        
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
            if exclude_mask.shape != (height, width):
                exclude_img_resized = Image.fromarray(exclude_mask.astype(np.uint8) * 255).resize(
                    (width, height),
                    Image.Resampling.NEAREST
                )
                exclude_mask = np.array(exclude_img_resized) > 0
        
        output_mask = np.zeros((height, width), dtype=np.uint8)
        
        tree_count = 0
        grass_count = 0
        
        # Process each target class with SAM3
        for target_class in target_classes:
            threshold = self.class_thresholds.get(target_class, self.confidence_threshold)
            
            print(f"  🔍 Detecting '{target_class}' (threshold: {threshold:.2f})...")
            
            # Set confidence threshold for this class
            self.sam3_processor.confidence_threshold = threshold
            
            # Segment image with text prompt
            inference_state = self.sam3_processor.set_image(image)
            result = self.sam3_processor.set_text_prompt(
                state=inference_state,
                prompt=target_class
            )
            
            masks = result.get("masks", [])
            scores = result.get("scores", [])
            
            if len(masks) == 0:
                print(f"     No {target_class} detected")
                continue
            
            print(f"     Found {len(masks)} {target_class} mask(s)")
            
            # Process each detected mask
            for i, mask_tensor in enumerate(masks):
                mask = mask_tensor.squeeze().cpu().numpy().astype(bool)
                score = scores[i].item() if len(scores) > i else 0
                
                # Filter by quality
                if score < threshold:
                    continue
                
                # Skip if overlaps too much with existing OSM mask
                if exclude_mask is not None:
                    overlap = np.sum(mask & exclude_mask)
                    total = np.sum(mask)
                    if total > 0 and overlap / total > 0.3:
                        continue
                
                # Check if mask looks like vegetation (greenish colors)
                mask_pixels = image_np[mask]
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
                    # Classify as tree or grass based on detection and heuristics
                    class_id = self._classify_vegetation(image_np, mask, target_class)
                    
                    # Only overwrite background (0) pixels
                    output_mask[mask & (output_mask == 0)] = class_id
                    
                    if class_id == 5:
                        tree_count += 1
                    else:
                        grass_count += 1
        
        print(f"  ✅ Detected: {tree_count} trees, {grass_count} grass regions")
        
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
        output_dir: Optional[str] = None,
        target_classes: List[str] = ['tree', 'grass']
    ) -> List[str]:
        """
        Annotate multiple images in batch.
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing existing OSM masks
            output_dir: Directory to save output masks
            target_classes: Classes to detect
            
        Returns:
            List of output file paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        print(f"\n📦 Processing {len(image_files)} images...")
        
        output_paths = []
        for idx, img_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")
            
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
            
            try:
                result_path = self.annotate_image(
                    image_path, 
                    mask_path, 
                    target_classes=target_classes,
                    output_path=output_path
                )
                output_paths.append(result_path)
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
                continue
        
        print(f"\n✅ Batch processing complete: {len(output_paths)}/{len(image_files)} successful")
        return output_paths


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM3 Annotator for trees and grass')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--mask', help='Existing OSM mask path', default=None)
    parser.add_argument('--output', help='Output mask path', default=None)
    parser.add_argument('--classes', help='Classes to detect (comma-separated)', 
                       default='tree,grass')
    parser.add_argument('--sam3-path', help='Path to SAM3 directory (only needed if not installed via pip)', default=None)
    parser.add_argument('--device', help='Device (cuda/cpu)', default='cuda')
    parser.add_argument('--threshold', type=float, help='Confidence threshold', default=0.5)
    
    args = parser.parse_args()
    
    target_classes = [c.strip() for c in args.classes.split(',')]
    
    annotator = SAM3Annotator(
        device=args.device,
        confidence_threshold=args.threshold,
        sam3_path=args.sam3_path
    )
    
    result = annotator.annotate_image(
        args.image,
        args.mask,
        target_classes=target_classes,
        output_path=args.output
    )
    
    print(f"\n✅ Saved to: {result}")

