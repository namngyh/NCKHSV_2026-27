import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import garch_vn30  # noqa: E402,F401  (dat float64 mac dinh)
