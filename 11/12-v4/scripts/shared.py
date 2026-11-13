"""v4 outputs are isolated; v3 modules, the 11/11 data/native layout and the v1/v2 engines are reused read-only."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / '12-v3'
V2 = ROOT.parent / '12-v2'
V1 = ROOT.parent / '12'
BASE = ROOT.parent / '11'
for _offset, _path in enumerate([V3 / 'scripts', V2 / 'scripts', V1 / 'scripts', BASE / 'scripts'], 1):
    if str(_path) not in sys.path:
        sys.path.insert(_offset, str(_path))


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
