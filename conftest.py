"""Make the ``src`` layout importable without requiring an editable install.

``pip install -e .`` is the documented setup, but adding ``src`` here keeps
``pytest`` working straight from a clone.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
