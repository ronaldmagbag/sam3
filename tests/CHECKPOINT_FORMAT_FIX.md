# Checkpoint Format Compatibility Fix

## Problem

When using pip-installed `sam3`, the checkpoint loading code expects HuggingFace format (with "detector." prefix), but training checkpoints don't have this prefix. This causes all model weights to fail loading.

## Solutions

### Solution 1: Automatic Monkey-Patch (Recommended)

The `sam3_annotator.py` script now automatically patches the checkpoint loading function to handle both formats. **No action needed** - just use the script as normal:

```bash
python tests/sam3_annotator.py tests/trees --checkpoint ./checkpoints/checkpoint.pt
```

The patch is applied automatically when the script imports sam3, so it works with pip-installed versions.

### Solution 2: Convert Checkpoint Format

If you prefer to convert the checkpoint once and use it everywhere, use the conversion script:

```bash
# Convert training checkpoint to HuggingFace format
python tests/convert_checkpoint.py \
    ./checkpoints/checkpoint.pt \
    ./checkpoints/checkpoint_hf.pt

# Then use the converted checkpoint
python tests/sam3_annotator.py tests/trees --checkpoint ./checkpoints/checkpoint_hf.pt
```

### Solution 3: Manual Monkey-Patch in Your Code

If you're writing your own code and using pip-installed sam3, add this at the top of your script:

```python
import torch
from sam3.model_builder import build_sam3_image_model
from iopath.common.file_io import g_pathmgr

# Monkey-patch to handle both checkpoint formats
def patched_load_checkpoint(model, checkpoint_path):
    """Handles both HuggingFace and training checkpoint formats."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    
    # Check format
    has_detector_prefix = any("detector" in k for k in ckpt.keys())
    
    if has_detector_prefix:
        # HuggingFace format: remove "detector." prefix
        sam3_image_ckpt = {
            k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
        }
    else:
        # Training format: use keys as-is
        sam3_image_ckpt = dict(ckpt)
    
    model.load_state_dict(sam3_image_ckpt, strict=False)

# Apply patch
import sam3.model_builder
sam3.model_builder._load_checkpoint = patched_load_checkpoint

# Now use sam3 normally
model = build_sam3_image_model(checkpoint_path="./checkpoints/checkpoint.pt")
```

## Checkpoint Formats

### Training Checkpoint Format
```python
{
    "model": {
        "backbone.vision_backbone.trunk.*": ...,
        "backbone.language_backbone.*": ...,
        "transformer.*": ...,
    },
    "optimizer": ...,
    "epoch": ...
}
```

### HuggingFace Checkpoint Format
```python
{
    "detector.backbone.vision_backbone.trunk.*": ...,
    "detector.backbone.language_backbone.*": ...,
    "detector.transformer.*": ...,
}
```

## Which Solution to Use?

- **Solution 1 (Monkey-Patch)**: Use if you're using `sam3_annotator.py` - it's automatic
- **Solution 2 (Conversion)**: Use if you want to convert once and share the checkpoint
- **Solution 3 (Manual Patch)**: Use if you're writing custom code with pip-installed sam3

## Testing

After applying any solution, you should see:
- ✅ No "missing_keys" errors (or very few, only for optional components)
- ✅ Model loads successfully
- ✅ Segmentation works correctly

If you still see many missing keys, the checkpoint might be corrupted or incomplete.

