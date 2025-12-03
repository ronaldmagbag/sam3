"""Configuration handling."""

import toml


def load_config(path):
    """Loads a dictionary from configuration file."""
    return toml.load(path)


def save_config(attrs, path):
    """Saves a configuration dictionary to a file."""
    toml.dump(attrs, path)

