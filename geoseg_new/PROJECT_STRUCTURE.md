# GeoSeg Project Structure

A simplified, self-contained project for preparing COCO datasets from OSM data.

## Project Layout

```
geoseg/
├── geoseg/                    # Main package
│   ├── __init__.py
│   ├── tools/                 # Core tools
│   │   ├── __init__.py
│   │   ├── extract.py        # Extract features from OSM
│   │   ├── cover.py          # Generate tiles
│   │   ├── download.py       # Download images
│   │   └── rasterize.py      # Rasterize to masks
│   ├── osm/                   # OSM handlers
│   │   ├── __init__.py
│   │   ├── all_features.py   # Unified multi-type handler
│   │   └── core.py           # FeatureStorage, utilities
│   ├── tiles.py              # Tile utilities
│   ├── colors.py             # Color palette
│   └── config.py            # Configuration handling
├── scripts/
│   └── full_pipeline.py   # Main pipeline script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── PROJECT_STRUCTURE.md    # This file
```

## Key Features

1. **Extract** (`geoseg.tools.extract`): Extracts building, road, and parking features from OSM data
   - Supports single or multiple feature types
   - Uses `AllFeaturesHandler` for efficient single-pass extraction
   - Creates features with `feature_types` array property

2. **Cover** (`geoseg.tools.cover`): Generates tiles covering GeoJSON features
   - Uses supermercado to find tiles intersecting features
   - Outputs CSV file with tile coordinates

3. **Download** (`geoseg.tools.download`): Downloads satellite images
   - Supports Mapbox and OpenStreetMap tile servers
   - Rate-limited concurrent downloads

4. **Rasterize** (`geoseg.tools.rasterize`): Rasterizes features to segmentation masks
   - Multi-class support with different pixel values per feature type
   - Supports `feature_types` array property

## Dependencies

All dependencies are listed in `requirements.txt`:
- numpy, Pillow, mercantile, rasterio
- supermercado, geojson, osmium, shapely
- tqdm, toml, requests

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python scripts/full_pipeline.py \
    --osm-file data.osm.pbf \
    --output ./output \
    --zoom 18 \
    --feature-types all
```

