"""Load the frozen, pre-race model inputs used by the reproduction commands."""

import gzip
import json

from .paths import DATA


def load_frozen():
    path = DATA / 'model-inputs-2023-2026.json.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        snapshot = json.load(stream)
    if snapshot['schema_version'] != 1:
        raise ValueError(f'Unsupported model-input schema in {path.name}.')
    return snapshot['history'], snapshot['target'], snapshot['coverage'], snapshot['skipped']
