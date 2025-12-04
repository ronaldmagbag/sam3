#!/usr/bin/env python
"""
SAM3 Mask Generator - Generate masks from images using text prompts.

This script takes an image folder path and a text prompt, then generates
individual mask images (0.png, 1.png, etc.) in the same folder.

Usage:
    python sam3_mask_generator.py <image_folder> <text_prompt>
    
Example:
    python sam3_mask_generator.py /path/to/folder "a building"
    python sam3_mask_generator.py ./images "a human"
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


def generate_masks(image_folder, text_prompt, checkpoint_path=None, score_threshold=0.0):
    """
    Generate masks from an image using SAM3 with text prompt.
    
    Args:
        image_folder: Path to folder containing image.png or image.jpg
        text_prompt: Text description of what to segment (e.g., "a building", "a human")
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
    
    print(f"[INFO] Found image: {image_path}")
    print(f"[INFO] Text prompt: '{text_prompt}'")
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
        
        # Set image in processor
        print("[INFO] Setting image in processor...")
        inference_state = processor.set_image(image)
        print("[OK] Image set in processor")
        
        # Prompt the model with text
        print(f"[INFO] Processing text prompt: '{text_prompt}'...")
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        print("[OK] Text prompt processed")
        
        # Get masks, boxes, and scores
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print()
        print(f"[INFO] Found {len(masks)} mask(s)")
        
        if len(masks) == 0:
            print("[WARNING] No masks generated. Try a different prompt or check the image.")
            return 0
        
        # Filter masks by score threshold
        if score_threshold > 0.0:
            valid_indices = (scores >= score_threshold).nonzero(as_tuple=True)[0]
            masks = [masks[i] for i in valid_indices]
            scores = scores[valid_indices]
            boxes = boxes[valid_indices]
            print(f"[INFO] Filtered to {len(masks)} mask(s) with score >= {score_threshold}")
        
        # Create masks subfolder inside image folder
        folder_path = Path(image_folder)
        masks_folder = folder_path / "masks"
        masks_folder.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        print(f"[INFO] Saving masks to {masks_folder}...")
        for idx, mask in enumerate(masks):
            # Convert mask tensor to numpy array
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # Debug: print original shape
            original_shape = mask_np.shape
            print(f"[DEBUG] Mask {idx} original shape: {original_shape}, dtype: {mask_np.dtype}")
            
            # Ensure mask is 2D (squeeze out any singleton dimensions)
            # Handle shapes like [1, H, W], [H, W, 1], [H, W], etc.
            while mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np)
            
            # If still not 2D, take first channel or flatten
            if mask_np.ndim == 3:
                # If shape is [H, W, C], take first channel
                mask_np = mask_np[:, :, 0]
            elif mask_np.ndim != 2:
                # Fallback: reshape to 2D
                print(f"[WARNING] Unexpected mask shape: {mask_np.shape}, reshaping...")
                # Try to infer H and W from total size
                total_size = mask_np.size
                # Assume square-ish dimensions
                h = int(np.sqrt(total_size))
                w = total_size // h
                mask_np = mask_np.reshape(h, w)
            
            print(f"[DEBUG] Mask {idx} after processing: shape={mask_np.shape}, dtype={mask_np.dtype}, min={mask_np.min()}, max={mask_np.max()}")
            
            # Convert to boolean first, then to uint8 (0 or 255)
            # This ensures proper binary mask format
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.dtype in [np.float32, np.float64]:
                # Float masks: threshold at 0.5
                mask_np = (mask_np > 0.5).astype(np.uint8) * 255
            else:
                # Integer masks: threshold at 127
                mask_np = (mask_np > 127).astype(np.uint8) * 255
            
            # Ensure values are exactly 0 or 255
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
            
            # Verify it's 2D and uint8
            assert mask_np.ndim == 2, f"Mask {idx} must be 2D, got shape {mask_np.shape}"
            assert mask_np.dtype == np.uint8, f"Mask {idx} must be uint8, got {mask_np.dtype}"
            
            # Save mask as PNG (grayscale, mode='L') in masks/ subfolder
            mask_path = masks_folder / f"{idx}.png"
            
            # Create PIL Image explicitly as grayscale
            try:
                # Ensure array is contiguous and properly formatted
                if not mask_np.flags['C_CONTIGUOUS']:
                    mask_np = np.ascontiguousarray(mask_np)
                
                mask_image = Image.fromarray(mask_np, mode='L')
                
                # Verify the image mode
                if mask_image.mode != 'L':
                    print(f"[WARNING] Mask {idx} mode is {mask_image.mode}, converting to 'L'")
                    mask_image = mask_image.convert('L')
                    
            except Exception as e:
                print(f"[ERROR] Failed to create PIL Image from mask {idx}: {e}")
                print(f"  Shape: {mask_np.shape}, dtype: {mask_np.dtype}, min: {mask_np.min()}, max: {mask_np.max()}")
                continue
            
            # Save with PNG format explicitly - use optimize=False for maximum compatibility
            try:
                # Save with explicit format and ensure it's a valid PNG
                mask_image.save(mask_path, format='PNG', optimize=False)
                
                # Force close to ensure file is written
                mask_image.close()
                
            except Exception as e:
                print(f"[ERROR] Failed to save mask {idx} to {mask_path}: {e}")
                continue
            
            # Verify the saved file is valid and can be reopened
            try:
                verify_image = Image.open(mask_path)
                verify_array = np.array(verify_image)
                verify_image.close()
                print(f"[DEBUG] Verified mask {idx}: file size={mask_path.stat().st_size} bytes, shape={verify_array.shape}")
            except Exception as e:
                print(f"[WARNING] Saved mask {idx} verification failed: {e}")
            
            score = scores[idx].item() if isinstance(scores[idx], torch.Tensor) else float(scores[idx])
            print(f"  ✓ Saved mask {idx}: {mask_path.name} (score: {score:.4f})")
            saved_count += 1
        
        print()
        print("=" * 60)
        print(f"[SUCCESS] Generated and saved {saved_count} mask(s)")
        print("=" * 60)
        
        return saved_count
        
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
  python sam3_mask_generator.py ./images "a building"
  python sam3_mask_generator.py /path/to/folder "a human"
  python sam3_mask_generator.py ./images "a building" --score-threshold 0.5
  python sam3_mask_generator.py ./images "a building" --checkpoint ./models/sam3.pt
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
        help="Text prompt for segmentation (e.g., 'a building', 'a human')"
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

