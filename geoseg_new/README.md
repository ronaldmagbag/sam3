# GeoSeg - Geospatial Segmentation Dataset Pipeline

A self-contained pipeline for creating segmentation datasets from OpenStreetMap data and satellite imagery.

## Features

- **OSM Feature Extraction**: Extract buildings, roads, and parking areas from OSM data
- **Satellite Image Download**: Download tiles from Mapbox or OpenStreetMap
- **Mask Rasterization**: Convert GeoJSON features to segmentation masks
- **SAM Annotation** (Optional): Detect natural features (trees, grass) using Segment Anything Model
- **Mask Merging**: Combine OSM masks and SAM masks
- **COCO Export**: Convert to COCO format with train/val/test splits

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python full_pipeline.py \
    --osm-file data/osm/region.osm.pbf \
    --output ./data/coco_dataset/region \
    --zoom 18 \
    --feature-types all \
    --mapbox-token YOUR_MAPBOX_TOKEN

```

## Usage

### Basic Usage

```bash
# Extract buildings only
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types building

# Extract multiple feature types
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types building road parking

# Extract all features
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types all
```

### Test Mode (Quick Testing)

```bash
# Limit to 20 tiles for quick testing
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --test-tiles 20 \
    --feature-types parking
```

### With SAM Annotation (Trees/Grass Detection)

```bash
# Using SAM 1 (original)
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types all \
    --enable-sam \
    --sam-model vit_b \
    --sam-device cuda

# Using SAM 2 (recommended - better quality)
python full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --feature-types all \
    --enable-sam \
    --use-sam2 \
    --sam-model large \
    --sam-device cuda
```

**SAM 1 model options:**
- `vit_b`: Fastest, good quality
- `vit_l`: Balanced speed/quality
- `vit_h`: Best quality, slowest

**SAM 2 model options (recommended):**
- `tiny`: Fastest, lightest
- `small`: Good balance
- `base_plus`: Better quality
- `large`: Best quality (recommended)

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--osm-file` | Path to .osm.pbf file | Required |
| `--output` | Output directory | Required |
| `--zoom` | Zoom level for tiles | 19 |
| `--tile-size` | Tile size in pixels | 512 |
| `--feature-types` | Features to extract (building, road, parking, all) | building |
| `--mapbox-token` | Mapbox API token (or set MAPBOX_API_KEY env var) | None |
| `--test-tiles` | Limit tiles for quick testing | None |
| `--skip-download` | Skip image download step | False |
| `--skip-rasterize` | Skip mask rasterization step | False |
| `--enable-sam` | Enable SAM annotation for trees/grass | False |
| `--use-sam2` | Use SAM 2 instead of SAM 1 (better quality) | False |
| `--sam-model` | Model type. SAM1: vit_h/vit_l/vit_b. SAM2: tiny/small/base_plus/large | vit_b |
| `--sam-device` | Device for SAM (cuda, cpu) | cuda |

## Pipeline Steps

```
OSM File (.osm.pbf)
    ↓
[Step 1] Extract → GeoJSON features
    ↓
[Step 2] Cover → Tiles CSV
    ↓
[Step 3] Download → Satellite images
    ↓
[Step 4] Rasterize → Segmentation masks
    ↓
[Step 5] SAM Annotation → Natural feature masks (optional)
    ↓
[Step 6] Merge Masks → Combined masks (optional)
    ↓
[Step 7] Convert → COCO format dataset
```

## Output Structure

```
output/
├── coco_dataset/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── _annotations.coco.json
│   ├── val/
│   │   ├── images/
│   │   ├── masks/
│   │   └── _annotations.coco.json
│   └── test/
│       ├── images/
│       ├── masks/
│       └── _annotations.coco.json
└── geoseg_work/
    ├── geojson/
    ├── tiles/
    ├── images/
    ├── masks/
    ├── sam_masks/      (if --enable-sam)
    └── merged_masks/   (if --enable-sam)
```

## SAM Installation (Optional)

To enable SAM annotation for detecting trees and grass:

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Option 1: Install SAM 2 (recommended - better quality)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Option 2: Install SAM 1 (original)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**SAM 2 advantages:**
- Better mask quality
- Improved handling of complex scenes
- More accurate tree/grass classification

SAM checkpoints will be automatically downloaded on first use.

## Class Colors

| Class ID | Class Name | RGB Color | Source |
|----------|------------|-----------|--------|
| 0 | Background | Black (0,0,0) | - |
| 1 | Building | Red (255,0,0) | OSM |
| 2 | Road | Yellow (255,255,0) | OSM |
| 3 | Parking | Magenta (255,0,255) | OSM |
| 4 | Water | Blue (0,0,255) | OSM |
| 5 | Tree | Dark Green (0,128,0) | SAM |
| 6 | Grass | Light Green (144,238,144) | SAM |

## Data Sources

- **OSM Data**: Download from [Geofabrik](https://download.geofabrik.de/)
- **Satellite Imagery**: Mapbox (requires API token) or OpenStreetMap tiles

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## License

MIT License


- sam1
python full_pipeline.py --osm-file data/osm/andorra-251126.osm.pbf --output ./data/coco_dataset/andorra --zoom 18 --test-tiles 20 --tile-size 1024 --feature-types all --enable-sam --sam-model vit_h --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A

python full_pipeline.py --osm-file data/osm/andorra-251126.osm.pbf --output ./data/coco_dataset/andorra --zoom 18 --test-tiles 500 --feature-types all --enable-sam --sam-model vit_h --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A

- sam2
python full_pipeline.py --osm-file data/osm/england.osm.pbf --output ./data/coco_dataset/england --zoom 18 --test-tiles 20 --feature-types all --enable-sam --use-sam2 --sam-model large --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A

- sam3
python full_pipeline.py --osm-file data/osm/andorra-251126.osm.pbf --output ./data/coco_dataset/andorra --zoom 18 --test-tiles 100 --feature-types all --enable-sam --use-sam3 --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A

python full_pipeline.py --osm-file data/osm/andorra-251126.osm.pbf --output ./data/coco_dataset/andorra --zoom 18 --feature-types all --enable-sam --use-sam3 --mapbox-token pk.eyJ1IjoiZXZlbnR1bWFpIiwiYSI6ImNtaTZ6ZnBnZDAzeXEyaXB5a3FkamFveG0ifQ.VGWd4ptbCI_DyBTMWU6R1A



aws s3 cp andorra/coco_dataset s3://osm-data-export/sam3/ --recursive
