#!/usr/bin/env python
"""
Test script for SAM3 video processing functionality.
Tests text-prompted segmentation on videos (MP4 files or JPEG folders).
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sam3.model_builder import build_sam3_video_predictor


def test_video_sam3(video_path=None, text_prompt="a person", frame_index=0):
    """
    Test SAM3 video processing with text prompt.
    
    Args:
        video_path: Path to video file (MP4) or folder of JPEG images. 
                   If None, uses sample video from sam3 assets.
        text_prompt: Text prompt for segmentation.
        frame_index: Frame index to add prompt to.
    """
    print("=" * 60)
    print("Testing SAM3 Video Processing")
    print("=" * 60)
    
    # Determine video path
    if video_path is None:
        # Try to use sample video from sam3 assets
        sample_video_path = project_root / "assets" / "videos" / "bedroom.mp4"
        if sample_video_path.exists():
            video_path = str(sample_video_path)
            print(f"[INFO] Using sample video: {video_path}")
        else:
            # Try JPEG folder
            jpeg_folder = project_root / "assets" / "videos" / "0001"
            if jpeg_folder.exists() and os.path.isdir(jpeg_folder):
                video_path = str(jpeg_folder)
                print(f"[INFO] Using JPEG folder: {video_path}")
            else:
                print("[ERROR] No sample video found. Please provide a video path.")
                print("Usage: python test_video_sam3.py <video_path> [text_prompt] [frame_index]")
                return
    
    # Check if video path exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video path not found: {video_path}")
        return
    
    # Check if it's a file or directory
    is_directory = os.path.isdir(video_path)
    if is_directory:
        # Check if directory has images
        image_files = [f for f in os.listdir(video_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) == 0:
            print(f"[ERROR] No image files found in directory: {video_path}")
            return
        print(f"[INFO] Found {len(image_files)} image files in directory")
    else:
        if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"[WARNING] File extension suggests this might not be a video file")
    
    print(f"[INFO] Video path: {video_path}")
    print(f"[INFO] Text prompt: {text_prompt}")
    print(f"[INFO] Frame index: {frame_index}")
    print()
    
    try:
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        # Load the video predictor
        print("[INFO] Loading SAM3 video predictor...")
        video_predictor = build_sam3_video_predictor()
        print("[OK] Video predictor loaded successfully")
        
        # Start a session
        print(f"[INFO] Starting session with resource: {video_path}...")
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        
        if "session_id" not in response:
            print(f"[ERROR] Failed to start session: {response}")
            return
        
        session_id = response["session_id"]
        print(f"[OK] Session started: {session_id}")
        
        # Add prompt
        print(f"[INFO] Adding text prompt at frame {frame_index}...")
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text_prompt,
            )
        )
        
        if "outputs" not in response:
            print(f"[ERROR] Failed to add prompt: {response}")
            return
        
        print("[OK] Prompt added successfully")
        
        # Get outputs
        output = response["outputs"]
        
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        
        # Analyze output structure
        if isinstance(output, dict):
            print("Output keys:", list(output.keys()))
            for key, value in output.items():
                if isinstance(value, (list, tuple)):
                    print(f"  {key}: {len(value)} items")
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"    First item shape: {value[0].shape}")
                elif isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor with shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        elif isinstance(output, (list, tuple)):
            print(f"Output: List/Tuple with {len(output)} items")
            if len(output) > 0:
                print(f"  First item type: {type(output[0])}")
        else:
            print(f"Output type: {type(output)}")
        
        # Try to extract masks, boxes, scores if available
        if isinstance(output, dict):
            masks = output.get("masks", None)
            boxes = output.get("boxes", None)
            scores = output.get("scores", None)
            
            if masks is not None:
                if isinstance(masks, (list, tuple)):
                    print(f"\nMasks: {len(masks)} items")
                elif isinstance(masks, torch.Tensor):
                    print(f"\nMasks: Tensor with shape {masks.shape}")
            
            if boxes is not None:
                if isinstance(boxes, (list, tuple)):
                    print(f"Boxes: {len(boxes)} items")
                elif isinstance(boxes, torch.Tensor):
                    print(f"Boxes: Tensor with shape {boxes.shape}")
            
            if scores is not None:
                if isinstance(scores, (list, tuple)):
                    print(f"Scores: {len(scores)} items")
                    if len(scores) > 0:
                        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
                elif isinstance(scores, torch.Tensor):
                    print(f"Scores: Tensor with shape {scores.shape}")
                    if scores.numel() > 0:
                        print(f"  Score range: {scores.min():.4f} - {scores.max():.4f}")
        
        print()
        print("[SUCCESS] Test completed successfully!")
        print("=" * 60)
        
        return {
            "session_id": session_id,
            "output": output,
            "response": response
        }
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Parse command line arguments
    video_path = None
    text_prompt = "a person"
    frame_index = 0
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        text_prompt = sys.argv[2]
    if len(sys.argv) > 3:
        frame_index = int(sys.argv[3])
    
    test_video_sam3(video_path=video_path, text_prompt=text_prompt, frame_index=frame_index)

