# SAM3 Model Cache Configuration

This document explains where SAM3 downloads and caches model files from HuggingFace.

## Overview

SAM3 downloads model checkpoints from HuggingFace (facebook/sam3). Models are automatically cached in the `sam3/models` directory (relative to the sam3 package root, i.e., `3rdparty/sam3/models`).

## Cache Directory Location

Models are automatically downloaded and cached in:
- **Relative path**: `sam3/models/` (from sam3 package root)
- **Absolute path**: `3rdparty/sam3/models/` (from project root)

The directory is created automatically if it doesn't exist.

## Docker Configuration

In Docker, you can mount the sam3/models directory as a volume to persist downloaded models:

```bash
# Mount the sam3 directory to persist models
docker run --gpus all -it --rm \
  -v $(pwd)/3rdparty/sam3:/app/3rdparty/sam3 \
  geoseg-cuda126:latest
```

This ensures that models downloaded inside the container are persisted on the host.

## Example Usage

```python
from sam3.model_builder import build_sam3_image_model

# Models will be automatically cached in sam3/models/
model = build_sam3_image_model()
```

```python
from sam3.model_builder import build_sam3_video_model

# Models will be automatically cached in sam3/models/
model = build_sam3_video_model()
```

## Troubleshooting

### Access Denied Error

If you see "Access to model facebook/sam3 is restricted", you need to:

1. **Authenticate with HuggingFace**:
   ```bash
   huggingface-cli login
   ```

2. **Set your token as environment variable**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **Or use the token in code**:
   ```python
   import os
   os.environ["HF_TOKEN"] = "your_token_here"
   ```

### Cache Directory Not Working

- Ensure the directory exists and is writable
- Check that you have sufficient disk space
- Verify the path is absolute (not relative)
- On Windows, use forward slashes or raw strings: `r"E:\models\cache"`

## Notes

- The cache directory will contain subdirectories for each model repository
- Downloaded files are typically named with their SHA hashes
- Once cached, models won't be re-downloaded unless the cache is cleared
- The cache directory can be shared across multiple projects using HuggingFace models

