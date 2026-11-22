"""Project data locations shared by the command-line tools."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
RUNS = ROOT / 'runs'
CACHE = ROOT / '.openf1-cache'
TIMING_CACHE = ROOT / '.cache' / 'timing'
