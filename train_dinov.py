from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.training import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
