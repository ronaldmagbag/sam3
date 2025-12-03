"""
Split dataset into train/val/test sets.
"""

import os
import shutil
import random
from typing import Tuple


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Split dataset into train/val/test splits.
    
    Args:
        input_dir: Input directory containing images/ and masks/ subdirectories
        output_dir: Output directory for splits
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"Masks directory not found: {masks_dir}")
    
    # Get all image files
    image_files = [
        f for f in os.listdir(images_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(image_files)
    
    # Calculate split indices
    n = len(image_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }
    
    # Create output directories and copy files
    for split_name, files in splits.items():
        split_images_dir = os.path.join(output_dir, split_name, "images")
        split_masks_dir = os.path.join(output_dir, split_name, "masks")
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_masks_dir, exist_ok=True)
        
        for f in files:
            # Copy image
            src_image = os.path.join(images_dir, f)
            dst_image = os.path.join(split_images_dir, f)
            shutil.copy(src_image, dst_image)
            
            # Copy corresponding mask (try different extensions)
            mask_name = os.path.splitext(f)[0]
            mask_extensions = ['.png', '.jpg', '.jpeg']
            mask_copied = False
            
            for ext in mask_extensions:
                src_mask = os.path.join(masks_dir, f"{mask_name}{ext}")
                if os.path.exists(src_mask):
                    dst_mask = os.path.join(split_masks_dir, f"{mask_name}{ext}")
                    shutil.copy(src_mask, dst_mask)
                    mask_copied = True
                    break
            
            if not mask_copied:
                print(f"Warning: No mask found for {f}")
        
        print(f"{split_name}: {len(files)} images")
    
    print(f"Dataset split completed. Output: {output_dir}")

