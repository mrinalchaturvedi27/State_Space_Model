"""Make phase 2's `src` package and `slt_reporting` importable from phase 3 code."""
import os
import sys

PHASE3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = os.path.join(os.path.dirname(PHASE3_DIR), "phase 2")
for d in (PHASE2_DIR, PHASE3_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)
