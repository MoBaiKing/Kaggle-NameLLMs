"""Version 2 outputs are isolated; native architecture/data remain shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT.parent / '12'
BASE = ROOT.parent / '11'
sys.path.insert(1, str(V1 / 'scripts'))
sys.path.insert(2, str(BASE / 'scripts'))
