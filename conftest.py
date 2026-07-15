import sys
from pathlib import Path

# Make `src/` importable without installation, so tests run in a clean checkout.
sys.path.insert(0, str(Path(__file__).parent / "src"))
