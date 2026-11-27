import sys
from pathlib import Path

# the pinned emoji package is vendored into .pydeps (configs/difficulty/requirements.txt)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".pydeps"))
