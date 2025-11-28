# SAM (Segment Anything Model) Explanation

## What is SAM?

SAM (Segment Anything Model) is a powerful AI model developed by Meta that can segment objects in images without requiring specific training data. It's used in this pipeline to automatically annotate natural features that might be missing from OpenStreetMap (OSM) data.

## --skip-sam Flag: When to Use It

### Without `--skip-sam` (Default Behavior)

**What happens:**
1. OSM data is converted to masks (buildings, roads, landuse)
2. SAM is applied to the satellite image to find additional features
3. SAM identifies natural features like:
   - Trees and vegetation
   - Grass and open spaces
   - Small objects not in OSM
4. SAM mask is merged with OSM mask
5. Final mask contains both OSM and SAM annotations

**Pros:**
- ✅ More complete annotations
- ✅ Includes natural features (trees, grass)
- ✅ Better for training models that need all classes

**Cons:**
- ❌ Slower processing (SAM inference takes time)
- ❌ Requires SAM model to be installed
- ❌ Uses more computational resources

**Use when:**
- You need complete segmentation masks
- Training a model that requires all classes
- You have time and computational resources
- Natural features (trees, grass) are important for your use case

### With `--skip-sam` Flag

**What happens:**
1. OSM data is converted to masks (buildings, roads, landuse)
2. SAM step is skipped
3. Only OSM mask is used as final mask

**Pros:**
- ✅ Much faster processing
- ✅ No SAM installation required
- ✅ Lower computational requirements
- ✅ Good for testing/development

**Cons:**
- ❌ Missing natural features (trees, grass may not be annotated)
- ❌ Less complete annotations
- ❌ May miss small objects not in OSM

**Use when:**
- Quick testing or development
- OSM data is sufficient for your needs
- You don't need natural feature annotations
- Limited computational resources
- SAM is not installed

## Example Usage

```bash
# With SAM (complete annotations, slower)
python scripts/prepare_dataset.py --bbox "..." --provider mapbox

# Without SAM (OSM only, faster)
python scripts/prepare_dataset.py --bbox "..." --provider mapbox --skip-sam
```

## Technical Details

### SAM Integration

The SAM annotator (`src/label_preparation/sam_annotator.py`) uses SAM to:
1. Analyze the satellite image
2. Generate segmentation masks for target classes (trees, grass, etc.)
3. Return masks that complement OSM data

### Mask Merging

The mask merger (`src/label_preparation/mask_merger.py`) combines:
- **OSM mask**: Buildings, roads, landuse from OpenStreetMap
- **SAM mask**: Natural features detected by SAM
- **Final mask**: Merged result with priority-based conflict resolution

## Recommendation

- **For production/training**: Don't use `--skip-sam` (get complete annotations)
- **For testing/development**: Use `--skip-sam` (faster iteration)
- **For OSM-only datasets**: Use `--skip-sam` (if natural features aren't needed)

