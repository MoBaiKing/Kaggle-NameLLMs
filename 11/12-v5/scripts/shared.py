"""v5 outputs are isolated; v4/v3 modules and the 11/11 data/native layout are reused read-only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT.parent / '12-v4'
V3 = ROOT.parent / '12-v3'
BASE = ROOT.parent / '11'
for _offset, _path in enumerate([V4 / 'scripts', V3 / 'scripts', ROOT.parent / '12-v2' / 'scripts',
                                 ROOT.parent / '12' / 'scripts', BASE / 'scripts'], 1):
    if str(_path) not in sys.path:
        sys.path.insert(_offset, str(_path))
