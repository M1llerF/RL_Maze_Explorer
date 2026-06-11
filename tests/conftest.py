from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
