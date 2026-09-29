from __future__ import annotations

import os
import runpy
from pathlib import Path


TRITON_ROOT = Path(__file__).parent / "triton_src"

os.chdir(TRITON_ROOT)
runpy.run_path(TRITON_ROOT / "setup.py", run_name="__main__")
