"""Load experiment defaults and optional TOML overrides."""

import argparse
from copy import deepcopy
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config' / 'experiments.toml'


def load_settings(path=DEFAULT_CONFIG):
    with DEFAULT_CONFIG.open('rb') as stream:
        settings = tomllib.load(stream)
    if Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with Path(path).open('rb') as stream:
            overrides = tomllib.load(stream)
        merge_settings(settings, overrides)
    return settings


def merge_settings(settings, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            merge_settings(settings[key], value)
        else:
            settings[key] = deepcopy(value)


def configure_parser(parser):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args, _ = preliminary.parse_known_args()
    try:
        settings = load_settings(args.config)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))
    parser.add_argument('--config', type=Path, default=args.config,
                        help='TOML overrides for experiment and solver settings')
    return settings
