"""Color handling, color maps, color palettes."""

import colorsys
from enum import Enum, unique


def _rgb(v):
    r, g, b = v[1:3], v[3:5], v[5:7]
    return int(r, 16), int(g, 16), int(b, 16)


@unique
class Mapbox(Enum):
    """Mapbox-themed colors plus custom segmentation colors."""
    # Original Mapbox colors
    dark = _rgb("#404040")
    gray = _rgb("#eeeeee")
    light = _rgb("#f8f8f8")
    white = _rgb("#ffffff")
    cyan = _rgb("#3bb2d0")
    blue = _rgb("#3887be")
    bluedark = _rgb("#223b53")
    denim = _rgb("#50667f")
    navy = _rgb("#28353d")
    navydark = _rgb("#222b30")
    purple = _rgb("#8a8acb")
    teal = _rgb("#41afa5")
    green = _rgb("#56b881")
    yellow = _rgb("#f1f075")
    mustard = _rgb("#fbb03b")
    orange = _rgb("#f9886c")
    red = _rgb("#e55e5e")
    pink = _rgb("#ed6498")
    
    # Custom segmentation colors (matching SAM annotator)
    black = (0, 0, 0)              # Background
    building_red = (255, 0, 0)     # Building - Bright Red
    road_yellow = (255, 255, 0)    # Road - Bright Yellow
    parking_magenta = (255, 0, 255)  # Parking - Magenta
    water_blue = (0, 0, 255)       # Water - Bright Blue
    tree_green = (0, 128, 0)       # Tree - Dark Green
    grass_lightgreen = (144, 238, 144)  # Grass - Light Green


# Standard segmentation class colors (matching label_preparation)
CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building - Red
    2: (255, 255, 0),    # Road - Yellow
    3: (255, 0, 255),    # Parking - Magenta
    4: (0, 0, 255),      # Water - Blue
    5: (0, 128, 0),      # Tree - Dark Green
    6: (144, 238, 144),  # Grass - Light Green
}

# Feature type to class ID mapping
FEATURE_TYPE_TO_CLASS = {
    "background": 0,
    "building": 1,
    "road": 2,
    "parking": 3,
    "water": 4,
    "tree": 5,
    "grass": 6,
}

# Feature type to color name mapping
FEATURE_TYPE_TO_COLOR = {
    "background": "black",
    "building": "building_red",
    "road": "road_yellow",
    "parking": "parking_magenta",
    "water": "water_blue",
    "tree": "tree_green",
    "grass": "grass_lightgreen",
}


def make_palette(*colors):
    """Builds a PIL-compatible color palette from color names."""
    rgbs = [Mapbox[color].value for color in colors]
    flattened = sum(rgbs, ())
    return list(flattened)


def make_segmentation_palette(num_classes=7):
    """
    Create a palette using the standard segmentation colors.
    
    Args:
        num_classes: Number of classes (default 7: bg, building, road, parking, water, tree, grass)
    
    Returns:
        Flattened list of RGB values for PIL palette
    """
    palette = []
    for i in range(256):
        if i in CLASS_COLORS:
            palette.extend(CLASS_COLORS[i])
        else:
            palette.extend((0, 0, 0))  # Default to black
    return palette
