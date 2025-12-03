import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles

from config import load_config
from colors import make_palette, make_segmentation_palette
from tiles import tiles_from_csv


def add_parser(subparser):
    parser = subparser.add_parser(
        "rasterize", help="rasterize features to label masks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("features", type=str, help="path to GeoJSON features file")
    parser.add_argument("tiles", type=str, help="path to .csv tiles file")
    parser.add_argument("out", type=str, help="directory to write converted images")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("--zoom", type=int, required=True, help="zoom level of tiles")
    parser.add_argument("--size", type=int, default=512, help="size of rasterized image tiles in pixels")

    parser.set_defaults(func=main)


def feature_to_mercator(feature):
    """Normalize feature and converts coords to 3857."""
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)

    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        xys = (zip(*part) for part in geometry["coordinates"])
        xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
        yield {"coordinates": list(xys), "type": "Polygon"}
    elif geometry["type"] == "MultiPolygon":
        for component in geometry["coordinates"]:
            xys = (zip(*part) for part in component)
            xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
            yield {"coordinates": list(xys), "type": "Polygon"}


def burn(tile, features, size, feature_type_to_class=None):
    """Burn tile with features."""
    shapes = []
    for feature in features:
        if feature_type_to_class:
            props = feature.get("properties", {})
            feature_types = props.get("feature_types", None)
            if feature_types and isinstance(feature_types, list) and len(feature_types) > 0:
                feature_type = feature_types[0]
            else:
                feature_type = props.get("feature_type", None)
            burnval = feature_type_to_class.get(feature_type, 1)
        else:
            burnval = 1
        
        for geometry in feature_to_mercator(feature):
            shapes.append((geometry, burnval))

    bounds = mercantile.xy_bounds(tile)
    transform = from_bounds(*bounds, size, size)
    # Explicitly specify dtype to ensure uint8 output for PIL
    return rasterize(shapes, out_shape=(size, size), transform=transform, dtype=np.uint8)


def main(args):
    dataset = load_config(args.dataset)

    classes = dataset["common"]["classes"]
    colors = dataset["common"]["colors"]
    assert len(classes) == len(colors), "classes and colors coincide"

    feature_type_to_class = {}
    if len(classes) > 2:
        for idx, class_name in enumerate(classes):
            if class_name != "background":
                feature_type_to_class[class_name] = idx
    else:
        feature_type_to_class = None

    os.makedirs(args.out, exist_ok=True)
    assert all(tile.z == args.zoom for tile in tiles_from_csv(args.tiles))

    with open(args.features) as f:
        fc = json.load(f)

    feature_map = collections.defaultdict(list)
    with tqdm(fc["features"], ascii=True, unit="feature") as pbar:
        for i, feature in enumerate(pbar):
            if feature["geometry"]["type"] not in ["Polygon", "MultiPolygon"]:
                continue

            try:
                for tile in burntiles.burn([feature], zoom=args.zoom):
                    feature_map[mercantile.Tile(*tile)].append(feature)
            except ValueError as e:
                print("Warning: invalid feature {}, skipping".format(i), file=sys.stderr)
                continue

    with tqdm(list(tiles_from_csv(args.tiles)), ascii=True, unit="tile") as pbar:
        for tile in pbar:
            if tile in feature_map:
                out = burn(tile, feature_map[tile], args.size, feature_type_to_class)
            else:
                out = np.zeros(shape=(args.size, args.size), dtype=np.uint8)

            # Ensure output is uint8 for PIL Image.fromarray with mode="P"
            out = out.astype(np.uint8)

            out_dir = os.path.join(args.out, str(tile.z), str(tile.x))
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, "{}.png".format(tile.y))

            if os.path.exists(out_path):
                prev = np.array(Image.open(out_path))
                out = np.maximum(out, prev.astype(np.uint8))

            out = Image.fromarray(out, mode="P")
            palette = make_segmentation_palette()
            out.putpalette(palette)
            out.save(out_path, optimize=True)

