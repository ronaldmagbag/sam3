#!/usr/bin/env python
"""
Convert training checkpoint format to HuggingFace format (adds "detector." prefix).

This script is useful when using pip-installed sam3 where you can't modify
the source code directly.

Usage:
    python convert_checkpoint.py input_checkpoint.pt output_checkpoint.pt
"""

import sys
import torch
from pathlib import Path


def convert_training_to_hf_format(input_path: str, output_path: str):
    """
    Convert training checkpoint to HuggingFace format by adding "detector." prefix.
    
    Args:
        input_path: Path to training checkpoint (format: {"model": {...}})
        output_path: Path to save converted checkpoint
    """
    print(f"[INFO] Loading checkpoint from: {input_path}")
    
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    
    # Extract model state_dict
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint
    
    # Check if already in HuggingFace format
    has_detector_prefix = any("detector" in k for k in model_state_dict.keys())
    
    if has_detector_prefix:
        print("[INFO] Checkpoint already has 'detector.' prefix (HuggingFace format)")
        print("[INFO] No conversion needed. Copying as-is...")
        converted_state_dict = model_state_dict
    else:
        print("[INFO] Converting training checkpoint to HuggingFace format...")
        print("[INFO] Adding 'detector.' prefix to all model keys...")
        
        # Add "detector." prefix to all keys
        converted_state_dict = {
            f"detector.{k}": v for k, v in model_state_dict.items()
        }
        
        print(f"[INFO] Converted {len(converted_state_dict)} keys")
    
    # Create output checkpoint in HuggingFace format
    # HuggingFace checkpoints are just the state_dict directly
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Saving converted checkpoint to: {output_path}")
    torch.save(converted_state_dict, output_path)
    
    print("[SUCCESS] Checkpoint conversion complete!")
    return output_path


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_checkpoint.py <input_checkpoint.pt> <output_checkpoint.pt>")
        print("\nExample:")
        print("  python convert_checkpoint.py ./checkpoints/checkpoint.pt ./checkpoints/checkpoint_hf.pt")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(input_path).exists():
        print(f"[ERROR] Input checkpoint not found: {input_path}")
        sys.exit(1)
    
    try:
        convert_training_to_hf_format(input_path, output_path)
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

