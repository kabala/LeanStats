import os
import sys

# Ensure project root and src are on sys.path when running pytest
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
for path in [ROOT, SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)

