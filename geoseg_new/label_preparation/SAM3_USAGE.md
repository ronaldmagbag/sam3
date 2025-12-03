# SAM3 Annotator Usage Guide

## Overview

The SAM3 Annotator uses Meta's Segment Anything Model 3 (SAM3) with text-prompted segmentation to automatically detect and annotate trees and grass in satellite imagery. It's designed to work with the geoseg_new project and can exclude areas already labeled by OSM masks.

## Features

- **Text-Prompted Segmentation**: Uses natural language prompts like "tree" and "grass"
- **OSM Mask Exclusion**: Skips areas already labeled in OSM masks
- **Intelligent Classification**: Distinguishes between trees and grass using heuristics
- **Batch Processing**: Process multiple images at once
- **GPU Acceleration**: Supports CUDA for faster processing

## Installation

Make sure SAM3 is installed and available in your project structure:

```
geoseg/
  3rdparty/
    sam3/
      sam3/
        model_builder.py
        model/
          sam3_image_processor.py
```

## Basic Usage

### Single Image

```python
from label_preparation.sam3_annotator import SAM3Annotator

# Initialize annotator
annotator = SAM3Annotator(
    output_dir="./output/sam3_masks",
    device="cuda",
    confidence_threshold=0.5
)

# Annotate a single image
output_path = annotator.annotate_image(
    image_path="path/to/image.png",
    existing_mask_path="path/to/osm_mask.png",  # Optional
    target_classes=['tree', 'grass'],
    output_path="path/to/output_mask.png"
)
```

### Batch Processing

```python
# Process multiple images
output_paths = annotator.annotate_batch(
    image_dir="./data/images",
    mask_dir="./data/osm_masks",  # Optional
    output_dir="./output/sam3_masks",
    target_classes=['tree', 'grass']
)

print(f"Processed {len(output_paths)} images")
```

## Command Line Usage

```bash
# Basic usage
python label_preparation/sam3_annotator.py image.png

# With OSM mask exclusion
python label_preparation/sam3_annotator.py image.png --mask osm_mask.png

# Custom output path
python label_preparation/sam3_annotator.py image.png --output result.png

# Detect only trees
python label_preparation/sam3_annotator.py image.png --classes tree

# Custom confidence threshold
python label_preparation/sam3_annotator.py image.png --threshold 0.3

# Specify SAM3 path (if not auto-detected)
python label_preparation/sam3_annotator.py image.png --sam3-path /path/to/sam3

# Use CPU instead of GPU
python label_preparation/sam3_annotator.py image.png --device cpu
```

## Configuration

### Class-Specific Thresholds

The annotator uses different confidence thresholds for different classes:

```python
annotator.class_thresholds = {
    "tree": 0.3,   # Lower threshold = more detections
    "grass": 0.3,  # Lower threshold for sparse grass
}
```

### Custom SAM3 Path

If SAM3 is not in the default location:

```python
annotator = SAM3Annotator(
    sam3_path="/path/to/geoseg/3rdparty/sam3"
)
```

## Output Format

The output mask is a colored RGB image with the following color mapping:

| Class      | Class ID | Color (RGB)       | Description        |
|------------|----------|-------------------|--------------------|
| Background | 0        | (0, 0, 0)         | Black              |
| Tree       | 5        | (0, 128, 0)       | Dark Green         |
| Grass      | 6        | (144, 238, 144)   | Light Green        |

## Integration with Existing Pipeline

### Merging with OSM Masks

```python
from PIL import Image
import numpy as np

# Load OSM mask (classes 1-4)
osm_mask = np.array(Image.open("osm_mask.png"))

# Load SAM3 mask (classes 5-6)
sam3_mask = np.array(Image.open("sam3_mask.png"))

# Merge: SAM3 only fills background (0) areas
merged_mask = osm_mask.copy()
sam3_classes = (sam3_mask[:, :, 0] == 0) & (sam3_mask[:, :, 1] > 0)  # Green pixels
merged_mask[sam3_classes] = sam3_mask[sam3_classes]

# Save merged mask
Image.fromarray(merged_mask).save("merged_mask.png")
```

## Classification Logic

The annotator uses multiple heuristics to distinguish trees from grass:

1. **SAM3 Detection**: If SAM3 specifically detects "tree" or "grass", uses that
2. **Color Analysis**: Darker green → tree, lighter green → grass
3. **Texture Variance**: Higher variance → tree canopy, lower → grass field
4. **Shape Compactness**: Compact shapes → trees, spread out → grass
5. **Size**: Small/medium areas → trees, very large → grass fields

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for 10-20x speedup
2. **Adjust Thresholds**: Lower thresholds detect more but may include false positives
3. **Batch Processing**: More efficient than processing images one by one
4. **Image Size**: SAM3 resizes images to 1008x1008 internally

## Troubleshooting

### "SAM3 path not found"
Ensure the SAM3 project is in the correct location or specify the path:
```python
annotator = SAM3Annotator(sam3_path="/path/to/sam3")
```

### Low detection rate
Try lowering the confidence threshold:
```python
annotator.class_thresholds = {
    "tree": 0.2,
    "grass": 0.2
}
```

### Too many false positives
Increase the confidence threshold or adjust the green color detection threshold in `_run_sam3()`.

### Out of memory (GPU)
- Use CPU: `device="cpu"`
- Process smaller batches
- Reduce image resolution before processing

## Comparison with SAM1/SAM2

| Feature              | SAM1 Annotator | SAM2 Annotator | SAM3 Annotator |
|---------------------|----------------|----------------|----------------|
| Model               | SAM            | SAM2           | SAM3           |
| Prompt Type         | Auto-generate  | Auto-generate  | Text prompts   |
| Accuracy            | Good           | Better         | Best           |
| Speed               | Fast           | Medium         | Medium         |
| Language Control    | No             | No             | Yes            |
| GPU Memory          | ~2GB           | ~3GB           | ~4GB           |

## Example Workflow

```python
from label_preparation.sam3_annotator import SAM3Annotator

# 1. Initialize
annotator = SAM3Annotator(
    output_dir="./sam3_output",
    device="cuda",
    confidence_threshold=0.5
)

# 2. Process images with OSM masks
annotator.annotate_batch(
    image_dir="./data/tiles/images",
    mask_dir="./data/tiles/osm_masks",
    output_dir="./data/tiles/sam3_masks",
    target_classes=['tree', 'grass']
)

# 3. Results are saved as colored PNG masks
# 4. Merge with OSM masks in your pipeline
```

## References

- SAM3 Project: `geoseg/3rdparty/sam3/`
- Original SAM Annotator: `label_preparation/sam_annotator.py`
- SAM2 Annotator: `label_preparation/sam2_annotator.py`

