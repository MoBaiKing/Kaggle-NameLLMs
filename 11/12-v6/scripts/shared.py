"""v6 outputs are isolated; the 11/12 native engine, the v3/v4 helpers and the 11/11 data layout are reused read-only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT.parent / '12-v5'
V4 = ROOT.parent / '12-v4'
V3 = ROOT.parent / '12-v3'
V1 = ROOT.parent / '12'
BASE = ROOT.parent / '11'
for _offset, _path in enumerate([V4 / 'scripts', V3 / 'scripts', ROOT.parent / '12-v2' / 'scripts',
                                 V1 / 'scripts', BASE / 'scripts'], 1):
    if str(_path) not in sys.path:
        sys.path.insert(_offset, str(_path))
