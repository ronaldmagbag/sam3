import argparse

from osm.all_features import AllFeaturesHandler


def add_parser(subparser):
    parser = subparser.add_parser(
        "extract",
        help="extracts GeoJSON features from OpenStreetMap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type", 
        type=str, 
        dest="types",
        action="append",
        choices=["building", "parking", "road", "all"], 
        help="type of feature to extract (can be specified multiple times for multiple types, or use 'all' for all features)"
    )
    parser.add_argument("--batch", type=int, default=100000, help="number of features to save per file")
    parser.add_argument("map", type=str, help="path to .osm.pbf base map")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):
    if args.types is None:
        raise ValueError("--type must be specified at least once")
    
    feature_types = args.types
    
    # Check if "all" is specified
    if "all" in feature_types:
        # Use AllFeaturesHandler to extract all features at once
        handler = AllFeaturesHandler(args.out, args.batch)
        handler.apply_file(filename=args.map, locations=True)
        handler.flush()
    else:
        # Multiple types: use AllFeaturesHandler with specific feature types
        types_to_extract = [t for t in feature_types if t != "all" and t in ["building", "parking", "road"]]
        handler = AllFeaturesHandler(args.out, args.batch, feature_types=types_to_extract)
        handler.apply_file(filename=args.map, locations=True)
        handler.flush()

