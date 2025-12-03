"""
Merge OSM masks and SAM masks into final segmentation labels.
Combines different mask sources with proper class handling.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Optional, Dict

# Unified color mapping for all mask classes (RGB values)
# Background is black, each class has a distinct visible color
CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building - Red (OSM)
    2: (255, 255, 0),    # Road - Yellow (OSM)
    3: (255, 0, 255),    # Parking - Magenta (OSM)
    4: (0, 0, 255),      # Water - Blue (OSM)
    5: (0, 128, 0),      # Tree - Dark Green (SAM)
    6: (144, 238, 144),  # Grass - Light Green (SAM)
}

# Reverse mapping: color to class ID
COLOR_TO_CLASS = {v: k for k, v in CLASS_COLORS.items()}


class MaskMerger:
    """Merges multiple mask sources (OSM, SAM) into final segmentation masks."""
    
    def __init__(self, output_dir: str = "./data/processed/masks"):
        """
        Initialize mask merger.
        
        Args:
            output_dir: Directory to save merged masks
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Class priority: higher number = higher priority
        # Buildings/Roads override vegetation, vegetation overrides background
        self.class_priority = {
            0: 0,  # Background (lowest)
            6: 1,  # Grass
            5: 2,  # Tree
            4: 3,  # Water
            3: 3,  # Parking
            2: 4,  # Road
            1: 5,  # Building (highest)
        }
    
    def _load_mask_as_class_ids(self, mask_path: str) -> np.ndarray:
        """
        Load a mask and convert to class IDs.
        Handles both RGB colored masks and grayscale/palette masks.
        
        Args:
            mask_path: Path to mask image
            
        Returns:
            2D numpy array with class IDs
        """
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img)
        
        # If RGB, convert colors back to class IDs
        if len(mask.shape) == 3 and mask.shape[2] >= 3:
            mask_grayscale = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            
            # Map RGB colors back to class IDs with tolerance
            for class_id, color in CLASS_COLORS.items():
                color_array = np.array(color)
                # Use tolerance for color matching
                r_match = np.abs(mask[:, :, 0].astype(int) - color_array[0]) < 15
                g_match = np.abs(mask[:, :, 1].astype(int) - color_array[1]) < 15
                b_match = np.abs(mask[:, :, 2].astype(int) - color_array[2]) < 15
                color_match = r_match & g_match & b_match
                mask_grayscale[color_match] = class_id
            
            return mask_grayscale
        elif len(mask.shape) == 3:
            # Multi-channel but not RGB, take first channel
            return mask[:, :, 0]
        else:
            # Already 2D (grayscale or palette mode)
            return mask
    
    def _save_colored_mask(self, mask: np.ndarray, output_path: str):
        """
        Save a class ID mask as a colored RGB image.
        
        Args:
            mask: 2D array with class IDs
            output_path: Path to save the colored mask
        """
        height, width = mask.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Map each class to its color
        for class_id, color in CLASS_COLORS.items():
            mask_pixels = (mask == class_id)
            colored_image[mask_pixels] = color
        
        colored_img = Image.fromarray(colored_image, mode='RGB')
        colored_img.save(output_path)
    
    def merge_masks(
        self,
        mask_paths: List[str],
        output_path: str,
        merge_strategy: str = "priority"
    ) -> str:
        """
        Merge multiple masks into a single mask.
        
        Args:
            mask_paths: List of paths to mask files
            output_path: Path to save merged mask
            merge_strategy: Strategy for merging ('priority', 'max', 'union')
        
        Returns:
            Path to merged mask
        """
        if not mask_paths:
            raise ValueError("No mask paths provided")
        
        # Load first mask
        first_mask = self._load_mask_as_class_ids(mask_paths[0])
        merged_mask = first_mask.copy()
        
        # Merge remaining masks
        for mask_path in mask_paths[1:]:
            mask = self._load_mask_as_class_ids(mask_path)
            
            if mask.shape != merged_mask.shape:
                # Resize mask to match
                mask = np.array(Image.fromarray(mask).resize(
                    (merged_mask.shape[1], merged_mask.shape[0]),
                    Image.Resampling.NEAREST
                ))
            
            if merge_strategy == "priority":
                merged_mask = self._merge_priority(merged_mask, mask)
            elif merge_strategy == "max":
                merged_mask = np.maximum(merged_mask, mask)
            elif merge_strategy == "union":
                merged_mask = np.where(mask > 0, mask, merged_mask)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Save as colored RGB image
        merged_mask = merged_mask.astype(np.uint8)
        self._save_colored_mask(merged_mask, output_path)
        
        return output_path
    
    def _merge_priority(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Merge masks using priority-based strategy.
        Higher priority classes override lower priority classes.
        
        Args:
            mask1: First mask (class IDs)
            mask2: Second mask (class IDs)
        
        Returns:
            Merged mask
        """
        merged = mask1.copy()
        
        # Create priority maps
        priority_map1 = np.zeros_like(mask1, dtype=int)
        priority_map2 = np.zeros_like(mask2, dtype=int)
        
        for class_id, priority in self.class_priority.items():
            priority_map1[mask1 == class_id] = priority
            priority_map2[mask2 == class_id] = priority
        
        # Where mask2 has higher priority and is not background, use mask2
        non_background = (mask2 > 0)
        higher_priority = (priority_map2 > priority_map1) & non_background
        merged[higher_priority] = mask2[higher_priority]
        
        # Also fill in background areas with mask2's content
        background_in_mask1 = (mask1 == 0)
        merged[background_in_mask1 & non_background] = mask2[background_in_mask1 & non_background]
        
        return merged
    
    def merge_osm_and_sam(
        self,
        osm_mask_path: str,
        sam_mask_path: str,
        output_path: str
    ) -> str:
        """
        Merge OSM mask and SAM mask.
        OSM classes (buildings, roads, parking) take priority over SAM (trees, grass).
        
        Args:
            osm_mask_path: Path to OSM-generated mask
            sam_mask_path: Path to SAM-generated mask
            output_path: Path to save merged mask
        
        Returns:
            Path to merged mask
        """
        return self.merge_masks([osm_mask_path, sam_mask_path], output_path, merge_strategy="priority")
    
    def clean_mask(
        self,
        mask_path: str,
        output_path: str,
        min_area: int = 10,
        remove_small_objects: bool = True
    ) -> str:
        """
        Clean mask by removing small objects and noise.
        
        Args:
            mask_path: Path to input mask
            output_path: Path to save cleaned mask
            min_area: Minimum area for objects to keep
            remove_small_objects: Whether to remove small objects
        
        Returns:
            Path to cleaned mask
        """
        from scipy import ndimage
        
        mask = self._load_mask_as_class_ids(mask_path)
        
        if remove_small_objects:
            for class_id in np.unique(mask):
                if class_id == 0:
                    continue
                
                class_mask = (mask == class_id).astype(int)
                labeled, num_features = ndimage.label(class_mask)
                
                # Use smaller threshold for vegetation
                class_min_area = min_area // 2 if class_id in [5, 6] else min_area
                
                for label_id in range(1, num_features + 1):
                    component = (labeled == label_id)
                    area = np.sum(component)
                    if area < class_min_area:
                        mask[component] = 0
        
        # Save as colored image
        mask = mask.astype(np.uint8)
        self._save_colored_mask(mask, output_path)
        
        return output_path
