# Triton Dependency - Performance & Installation Guide

## Why Triton is Used

Triton is used for **GPU-accelerated Euclidean Distance Transform (EDT)** operations. This is a critical optimization for interactive segmentation tasks.

### Performance Comparison (from code comments, tested on H100 GPU):
- **OpenCV CPU**: 1780ms (including CPU round-trip)
- **Triton O(N³)**: 627ms (~2.8x faster)
- **Triton O(N²)**: 322ms (~5.5x faster)

**Conclusion**: Triton provides significant speedup (5.5x faster) when processing large batches of masks on GPU.

## Where It's Used

The `edt_triton` function is used in:
- `sample_one_point_from_error_center()` - For interactive segmentation training/inference
- Finding optimal click points for correcting segmentation errors

**Important**: This is primarily used for **interactive segmentation workflows**, not for basic annotation tasks. If you're just running `sam3_annotator.py` for single-image segmentation, the performance impact may be minimal.

## Performance Impact of Making Triton Optional

### ✅ Good News:
1. **For basic annotation**: If you're just segmenting images without interactive correction, the performance loss is **minimal or zero** because the EDT function isn't called.

2. **Automatic fallback**: The code now automatically falls back to OpenCV when triton is unavailable. While slower, it's still functional.

### ⚠️ Performance Loss When EDT is Actually Used:
- **~5.5x slower** for EDT operations when using OpenCV fallback instead of triton
- This only matters if:
  - You're doing interactive segmentation with many iterations
  - You're processing very large batches (256 x 1024 x 1024 masks)
  - You're training models that use the interactive predictor

### For Your Use Case:
Since `sam3_annotator.py` doesn't appear to use the interactive EDT functions, **you likely won't notice any performance difference**.

## Why We Can't Install Triton on Windows

### Technical Reasons:
1. **No Official Windows Support**: Triton is primarily designed for Linux systems
2. **Compiler Dependencies**: Requires LLVM and other Unix-based build tools
3. **CUDA Toolkit Integration**: Better integrated with Linux CUDA installations

### Attempted Installation Options:
1. **Direct pip install**: `pip install triton` may fail or install a non-functional version on Windows
2. **WSL (Windows Subsystem for Linux)**: Could work, but requires running Python in Linux environment
3. **Docker/WSL2**: Possible workaround but adds complexity

### Our Solution:
Instead of trying to force triton installation, we:
1. Made triton import optional (graceful degradation)
2. Added OpenCV fallback (already installed in dev dependencies)
3. Maintained full functionality with acceptable performance for most use cases

## Recommendations

### If You Need Maximum Performance:
1. **Use Linux environment** (WSL2, Docker, or remote Linux server)
2. Install triton: `pip install triton`
3. Ensure CUDA is properly configured

### If You're on Windows (Current Setup):
1. **Keep the current setup** - it works fine for basic annotation
2. Install OpenCV for fallback: `pip install opencv-python`
3. Performance will be slower only if you use interactive segmentation features

### To Verify If Triton Would Help You:
1. Check if your workflow uses `sample_one_point_from_error_center()` function
2. If not, you won't benefit from triton
3. If yes, consider using a Linux environment for training/interactive workflows

## Installation Attempt (Optional)

If you want to try installing triton anyway:

```powershell
# This may or may not work on Windows
pip install triton
```

**Note**: Even if installation succeeds, it may not function correctly on Windows due to missing system dependencies.

## Summary

- **Triton provides ~5.5x speedup** for EDT operations on GPU
- **Making it optional causes performance loss** only when EDT is actually used
- **For basic annotation**, performance impact is **negligible**
- **Windows doesn't officially support triton**, but fallback works fine
- **Our solution maintains functionality** while allowing Windows compatibility

