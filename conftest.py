"""
Put the project root on sys.path so tests can import `graph`, `models`,
`training`, `detection` and `backend` regardless of how pytest is invoked.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
