#!/usr/bin/env python
"""
Test script for SAM3 image processing functionality.
Tests text-prompted segmentation on images.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def plot_results(image, output, image_path=None):
    """
    Draw bounding boxes and scores on the image and save it.
    
    Args:
        image: PIL.Image - The input image
        output: dict - SAM3 output containing 'boxes' and 'scores'
        image_path: str or Path - Original image path to determine save location.
                     If provided, saves as <original_name>_annotated.<ext>
    
    Returns:
        PIL.Image - The annotated image
    """
    # Convert to RGB if needed
    image = image.convert("RGB")
    
    # Create a copy to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Get boxes and scores
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    
    if len(boxes) == 0:
        print("[WARNING] No boxes found in output, nothing to draw")
        return img_with_boxes
    
    # Define colors for different boxes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    nb_objects = len(boxes)
    print(f"[INFO] Drawing {nb_objects} box(es) on image")
    
    # Draw each box and score
    for i in range(nb_objects):
        color = colors[i % len(colors)]
        
        # Get box coordinates (XYXY format)
        if isinstance(boxes[i], torch.Tensor):
            box = boxes[i].cpu().numpy()
        else:
            box = np.array(boxes[i])
        
        x1, y1, x2, y2 = box[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get score
        if isinstance(scores[i], torch.Tensor):
            score = scores[i].item()
        else:
            score = float(scores[i])
        
        # Draw bounding box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label text
        label_text = f"ID:{i} Score:{score:.3f}"
        
        # Get text bounding box to draw background
        try:
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        except AttributeError:
            # Fallback for older PIL versions
            text_size = draw.textsize(label_text, font=font)
            bbox = (x1, y1 - 20, x1 + text_size[0], y1 - 20 + text_size[1])
        
        text_bg = [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2]
        
        # Draw text background
        draw.rectangle(text_bg, fill=(0, 0, 0))
        
        # Draw text
        draw.text((x1, y1 - 20), label_text, fill=color, font=font)
    
    # Determine save path
    if image_path:
        base_path = Path(image_path)
        save_path = base_path.parent / f"{base_path.stem}_annotated{base_path.suffix}"
    else:
        # Default save location
        save_path = project_root / "tests" / "images" / "annotated_result.png"
    
    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    img_with_boxes.save(save_path)
    print(f"[INFO] Saved annotated image to: {save_path}")
    
    return img_with_boxes


def test_image_sam3(image_path=None, text_prompt="a car"):
    """
    Test SAM3 image processing with text prompt.
    
    Args:
        image_path: Path to image file. If None, uses sample image from sam3 assets.
        text_prompt: Text prompt for segmentation.
    """
    print("=" * 60)
    print("Testing SAM3 Image Processing")
    print("=" * 60)
    
    # Determine image path
    if image_path is None:
        # Try to use sample image from sam3 assets
        sample_image_path = project_root / "assets" / "images" / "test_image.jpg"
        if sample_image_path.exists():
            image_path = str(sample_image_path)
            print(f"[INFO] Using sample image: {image_path}")
        else:
            # Try other sample images
            for img_name in ["groceries.jpg", "truck.jpg"]:
                alt_path = project_root / "assets" / "images" / img_name
                if alt_path.exists():
                    image_path = str(alt_path)
                    print(f"[INFO] Using sample image: {image_path}")
                    break
            
            if image_path is None:
                print("[ERROR] No sample image found. Please provide an image path.")
                print("Usage: python test_image_sam3.py <image_path> [text_prompt]")
                return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    
    print(f"[INFO] Image path: {image_path}")
    print(f"[INFO] Text prompt: {text_prompt}")
    print()
    
    try:
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        # Load the model
        print("[INFO] Loading SAM3 image model...")
        print("[INFO] Models will be cached in sam3/models directory")
        model = build_sam3_image_model()
        print("[OK] Model loaded successfully")
        
        # Create processor
        print("[INFO] Creating processor...")
        processor = Sam3Processor(model, device=device)
        print("[OK] Processor created")
        
        # Load image
        print(f"[INFO] Loading image: {image_path}...")
        image = Image.open(image_path)
        print(f"[OK] Image loaded: {image.size[0]}x{image.size[1]}")
        
        # Set image in processor
        print("[INFO] Setting image in processor...")
        inference_state = processor.set_image(image)
        print("[OK] Image set in processor")
        
        # Prompt the model with text
        print(f"[INFO] Processing text prompt: '{text_prompt}'...")
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # Draw boxes and scores, then save the image
        annotated_image = plot_results(image, output, image_path)
        print("[OK] Text prompt processed")
        
        # Get the masks, bounding boxes, and scores
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Number of masks: {len(masks)}")
        print(f"Number of boxes: {len(boxes)}")
        print(f"Number of scores: {len(scores)}")
        
        if len(scores) > 0:
            print(f"\nScore range: {scores.min():.4f} - {scores.max():.4f}")
            print(f"Mean score: {scores.mean():.4f}")
            
            # Show top 5 results
            print("\nTop 5 results:")
            sorted_indices = torch.argsort(scores, descending=True)[:5]
            for i, idx in enumerate(sorted_indices):
                score = scores[idx].item()
                box = boxes[idx]
                print(f"  {i+1}. Score: {score:.4f}, Box: {box.tolist()}")
        
        if len(masks) > 0:
            mask_shapes = [m.shape for m in masks]
            print(f"\nMask shapes: {mask_shapes[:5]}")  # Show first 5
        
        print()
        print("[SUCCESS] Test completed successfully!")
        print("=" * 60)
        
        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "image": image,
            "output": output
        }
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Parse command line arguments
    image_path = "tests/images/house.png"
    text_prompt = "an object"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        text_prompt = sys.argv[2]
    
    test_image_sam3(image_path=image_path, text_prompt=text_prompt)

