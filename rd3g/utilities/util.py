"""Shared locations for solver outputs and visualization assets."""

import os

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASEDIR, 'outputs')
PICS_DIR = os.path.join(OUTPUT_DIR, 'pics')
GIFS_DIR = os.path.join(OUTPUT_DIR, 'gifs')


def resolve_logname(prefix='run', suffix='log'):
    """Return an unused output filename, creating its directory as needed."""
    if suffix == 'gif':
        folder = GIFS_DIR
    elif suffix in {'png', 'jpg', 'jpeg', 'pdf', 'svg'}:
        folder = PICS_DIR
    else:
        folder = os.path.join(BASEDIR, 'logs')
    os.makedirs(folder, exist_ok=True)
    number = 1
    filename = os.path.join(folder, f'{prefix}_{number}.{suffix}')
    while os.path.isfile(filename):
        number += 1
        filename = os.path.join(folder, f'{prefix}_{number}.{suffix}')
    return filename
