"""Reuse the frozen data, external assets and utility modules of 11/11."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / '11'
sys.path.insert(1, str(BASE / 'scripts'))
