#!/usr/bin/env python
"""
SAM3 Group Mask Generator - Generate merged masks from images using text prompts.

This script takes an image folder path and text prompt(s), then generates
merged mask images grouped by prompt. For each prompt, all individual masks
are merged into a single mask. Additionally, a background mask is created
that represents the entire image minus all object masks.

Multiple prompts can be provided separated by commas.

Usage:
    python sam3_group_masks_generator.py <image_folder> <text_prompt>
    
Example:
    python sam3_group_masks_generator.py /path/to/folder "a building"
    python sam3_group_masks_generator.py ./images "a human"
    python sam3_group_masks_generator.py ./images "a building, tree, car"
    
For "building, tree", this will create:
    - building_merged.png (all building masks merged)
    - tree_merged.png (all tree masks merged)
    - background.png (entire image minus building and tree masks)
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def normalize_mask(mask, target_shape=None):
    """
    Normalize a mask to 2D uint8 format (0 or 255).
    
    Args:
        mask: Mask as torch.Tensor or numpy array
        target_shape: Optional target shape (H, W) to resize to
        
    Returns:
        Normalized 2D numpy array with values 0 or 255
    """
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)
    
    # Squeeze out singleton dimensions
    while mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    
    # Handle 3D case
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    elif mask_np.ndim != 2:
        # Fallback: reshape to 2D
        total_size = mask_np.size
        h = int(np.sqrt(total_size))
        w = total_size // h
        mask_np = mask_np.reshape(h, w)
    
    # Resize if target shape provided
    if target_shape is not None and mask_np.shape != target_shape:
        from PIL import Image
        mask_img = Image.fromarray(mask_np)
        mask_img = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        mask_np = np.array(mask_img)
    
    # Convert to binary (0 or 255)
    if mask_np.dtype == bool:
        mask_np = mask_np.astype(np.uint8) * 255
    elif mask_np.dtype in [np.float32, np.float64]:
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_np = (mask_np > 127).astype(np.uint8) * 255
    
    # Ensure values are exactly 0 or 255
    mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    
    return mask_np


def merge_masks(masks, target_shape):
    """
    Merge multiple masks into a single mask using logical OR.
    
    Args:
        masks: List of masks (torch.Tensor or numpy arrays)
        target_shape: Target shape (H, W) for all masks
        
    Returns:
        Merged mask as 2D numpy array with values 0 or 255
    """
    if len(masks) == 0:
        # Return empty mask
        return np.zeros(target_shape, dtype=np.uint8)
    
    # Normalize all masks to same shape
    normalized_masks = [normalize_mask(mask, target_shape) for mask in masks]
    
    # Merge using logical OR (any pixel that is 255 in any mask becomes 255)
    merged = np.zeros(target_shape, dtype=np.uint8)
    for mask in normalized_masks:
        merged = np.maximum(merged, mask)
    
    return merged


def save_mask_image(mask_np, mask_path):
    """
    Save a normalized mask array to a PNG file.
    
    Args:
        mask_np: 2D numpy array with values 0 or 255
        mask_path: Path where mask will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure array is contiguous
        if not mask_np.flags['C_CONTIGUOUS']:
            mask_np = np.ascontiguousarray(mask_np)
        
        # Create PIL Image
        mask_image = Image.fromarray(mask_np, mode='L')
        
        # Verify mode
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')
        
        # Save
        mask_image.save(mask_path, format='PNG', optimize=False)
        mask_image.close()
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save mask to {mask_path}: {e}")
        return False


def find_image_in_folder(folder_path):
    """
    Find image.png or image.jpg in the given folder.
    
    Args:
        folder_path: Path to folder containing the image
        
    Returns:
        Path to the image file, or None if not found
    """
    folder = Path(folder_path)
    
    # Try image.png first
    image_path = folder / "image.png"
    if image_path.exists():
        return image_path
    
    # Try image.jpg
    image_path = folder / "image.jpg"
    if image_path.exists():
        return image_path
    
    # Try image.JPG (uppercase)
    image_path = folder / "image.JPG"
    if image_path.exists():
        return image_path
    
    # Try image.PNG (uppercase)
    image_path = folder / "image.PNG"
    if image_path.exists():
        return image_path
    
    return None


def generate_masks_for_prompt(processor, inference_state, text_prompt, score_threshold=0.0):
    """
    Generate masks from a single text prompt.
    
    Args:
        processor: Sam3Processor instance
        inference_state: Inference state from processor.set_image()
        text_prompt: Text description of what to segment (e.g., "a building", "a tree")
        score_threshold: Minimum score threshold for masks (default: 0.0, keeps all)
        
    Returns:
        List of mask tensors/arrays for the prompt
    """
    # Prompt the model with text
    print(f"[INFO] Processing text prompt: '{text_prompt}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    print("[OK] Text prompt processed")
    
    # Get masks, boxes, and scores
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    print()
    print(f"[INFO] Found {len(masks)} mask(s) for prompt '{text_prompt}'")
    
    if len(masks) == 0:
        print(f"[WARNING] No masks generated for prompt '{text_prompt}'. Try a different prompt or check the image.")
        return []
    
    # Filter masks by score threshold
    if score_threshold > 0.0:
        valid_indices = (scores >= score_threshold).nonzero(as_tuple=True)[0]
        masks = [masks[i] for i in valid_indices]
        scores = scores[valid_indices]
        boxes = boxes[valid_indices]
        print(f"[INFO] Filtered to {len(masks)} mask(s) with score >= {score_threshold}")
    
    # Print score info
    if len(scores) > 0:
        avg_score = scores.mean().item() if isinstance(scores, torch.Tensor) else float(np.mean(scores))
        print(f"[INFO] Average score: {avg_score:.4f}")
    
    return masks


def generate_masks(image_folder, text_prompts, checkpoint_path=None, score_threshold=0.0):
    """
    Generate masks from an image using SAM3 with text prompt(s).
    
    Args:
        image_folder: Path to folder containing image.png or image.jpg
        text_prompts: Text description(s) of what to segment. Can be a single string or list of strings.
                     Multiple prompts separated by commas will be parsed automatically.
        checkpoint_path: Optional path to SAM3 checkpoint (uses default if None)
        score_threshold: Minimum score threshold for masks (default: 0.0, keeps all)
        
    Returns:
        Number of masks generated
    """
    print("=" * 60)
    print("SAM3 Mask Generator")
    print("=" * 60)
    
    # Find image in folder
    image_path = find_image_in_folder(image_folder)
    if image_path is None:
        print(f"[ERROR] No image found in folder: {image_folder}")
        print("Expected: image.png or image.jpg")
        return 0
    
    # Parse text prompts - split by comma and strip whitespace
    if isinstance(text_prompts, str):
        # Split by comma and strip whitespace from each prompt
        prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    else:
        prompt_list = text_prompts
    
    if not prompt_list:
        print(f"[ERROR] No valid prompts provided")
        return 0
    
    print(f"[INFO] Found image: {image_path}")
    print(f"[INFO] Text prompts: {prompt_list}")
    print(f"[INFO] Output folder: {image_folder}")
    print()
    
    try:
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        # Load the model
        print("[INFO] Loading SAM3 image model...")
        if checkpoint_path:
            print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
            # Build model and load checkpoint if provided
            model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            model = build_sam3_image_model()
        print("[OK] Model loaded successfully")
        
        # Create processor
        print("[INFO] Creating processor...")
        processor = Sam3Processor(model, device=device)
        print("[OK] Processor created")
        
        # Load image
        print(f"[INFO] Loading image: {image_path}...")
        image = Image.open(image_path)
        image = image.convert("RGB")  # Ensure RGB format
        print(f"[OK] Image loaded: {image.size[0]}x{image.size[1]}")
        
        # Set image in processor (only once for all prompts)
        print("[INFO] Setting image in processor...")
        inference_state = processor.set_image(image)
        print("[OK] Image set in processor")
        
        # Get image dimensions for mask normalization
        image_width, image_height = image.size
        target_shape = (image_height, image_width)
        
        # Create masks subfolder inside image folder
        folder_path = Path(image_folder)
        masks_folder = folder_path / "masks"
        masks_folder.mkdir(parents=True, exist_ok=True)
        
        # Collect masks for each prompt
        prompt_masks = {}
        print()
        print("[INFO] Collecting masks for each prompt...")
        print("-" * 60)
        
        for prompt_idx, prompt in enumerate(prompt_list):
            print()
            print(f"[INFO] Processing prompt {prompt_idx + 1}/{len(prompt_list)}: '{prompt}'")
            
            masks = generate_masks_for_prompt(
                processor=processor,
                inference_state=inference_state,
                text_prompt=prompt,
                score_threshold=score_threshold
            )
            
            if len(masks) > 0:
                prompt_masks[prompt] = masks
                print(f"[OK] Collected {len(masks)} mask(s) for prompt '{prompt}'")
            else:
                print(f"[WARNING] No masks collected for prompt '{prompt}'")
        
        if len(prompt_masks) == 0:
            print()
            print("[ERROR] No masks generated for any prompt.")
            return 0
        
        # Merge masks for each prompt and save
        print()
        print("[INFO] Merging masks for each prompt...")
        print("-" * 60)
        
        all_merged_masks = {}  # Store all merged masks for background calculation
        current_index = 0  # Sequential naming for merged outputs
        
        for prompt_idx, (prompt, masks) in enumerate(prompt_masks.items()):
            print(f"[INFO] Merging {len(masks)} mask(s) for prompt '{prompt}'...")
            
            # Merge masks using logical OR
            merged_mask = merge_masks(masks, target_shape)
            all_merged_masks[prompt] = merged_mask
            
            mask_path = masks_folder / f"{current_index}.png"
            
            if save_mask_image(merged_mask, mask_path):
                print(f"  ✓ Saved merged mask for '{prompt}': {mask_path.name}")
            else:
                print(f"  ✗ Failed to save merged mask: {mask_path.name}")
            
            current_index += 1
        
        # Create background mask (entire image minus all object masks)
        print()
        print("[INFO] Creating background mask...")
        print("-" * 60)
        
        # Combine all object masks
        combined_objects = np.zeros(target_shape, dtype=np.uint8)
        for prompt, merged_mask in all_merged_masks.items():
            combined_objects = np.maximum(combined_objects, merged_mask)
        
        # Background is inverse of combined objects
        background_mask = (255 - combined_objects).astype(np.uint8)
        
        background_path = masks_folder / f"{current_index}.png"
        if save_mask_image(background_mask, background_path):
            print(f"  ✓ Saved background mask: {background_path.name}")
        else:
            print(f"  ✗ Failed to save background mask: {background_path.name}")
        
        print()
        print("=" * 60)
        print(f"[SUCCESS] Generated and saved {len(all_merged_masks)} merged mask(s) + 1 background mask")
        print(f"  - Merged masks: 0.png ... {current_index - 1}.png")
        print(f"  - Background mask: {current_index}.png")
        print("=" * 60)
        
        return len(all_merged_masks) + 1
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate masks: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate masks from images using SAM3 text prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sam3_group_masks_generator.py ./images "a building"
  python sam3_group_masks_generator.py /path/to/folder "a human"
  python sam3_group_masks_generator.py ./images "a building, tree, car"
  python sam3_group_masks_generator.py ./images "a building" --score-threshold 0.5
  python sam3_group_masks_generator.py ./images "a building" --checkpoint ./models/sam3.pt
  
For "building, tree", this will create:
  - building_merged.png (all building masks merged)
  - tree_merged.png (all tree masks merged)
  - background.png (entire image minus building and tree masks)
        """
    )
    
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to folder containing image.png or image.jpg"
    )
    
    parser.add_argument(
        "text_prompt",
        type=str,
        help="Text prompt(s) for segmentation. Multiple prompts can be separated by commas (e.g., 'a building, tree, car')"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint file (uses default HuggingFace model if not specified)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold for masks (default: 0.0, keeps all masks)"
    )
    
    args = parser.parse_args()
    
    # Validate folder exists
    if not os.path.exists(args.image_folder):
        print(f"[ERROR] Folder does not exist: {args.image_folder}")
        sys.exit(1)
    
    # Generate masks
    num_masks = generate_masks(
        args.image_folder,
        args.text_prompt,
        checkpoint_path=args.checkpoint,
        score_threshold=args.score_threshold
    )
    
    if num_masks == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

