"""Rebecca Platform Tests."""

import sys
from pathlib import Path

# Add the parent directory to Python path to make imports work
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
