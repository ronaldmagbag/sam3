"""Slippy Map Tiles."""

import csv
import io
import os

from PIL import Image
import mercantile


def pixel_to_location(tile, dx, dy):
    """Converts a pixel in a tile to a coordinate."""
    assert 0 <= dx <= 1, "x offset is in [0, 1]"
    assert 0 <= dy <= 1, "y offset is in [0, 1]"

    west, south, east, north = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat


def fetch_image(session, url, timeout=10):
    """Fetches the image representation for a tile."""
    import requests
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception:
        return None


def tiles_from_slippy_map(root):
    """Loads files from an on-disk slippy map directory structure."""
    def isdigit(v):
        try:
            _ = int(v)
            return True
        except ValueError:
            return False

    for z in os.listdir(root):
        if not isdigit(z):
            continue

        for x in os.listdir(os.path.join(root, z)):
            if not isdigit(x):
                continue

            for name in os.listdir(os.path.join(root, z, x)):
                y = os.path.splitext(name)[0]

                if not isdigit(y):
                    continue

                tile = mercantile.Tile(x=int(x), y=int(y), z=int(z))
                path = os.path.join(root, z, x, name)
                yield tile, path


def tiles_from_csv(path):
    """Read tiles from a line-delimited csv file."""
    with open(path) as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            yield mercantile.Tile(*map(int, row))

