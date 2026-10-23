"""Train an isolated candidate; the final holdout is never opened here."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Public compatibility import used by scripts.check_runtime.
from app.metrics import metrics
from app.training import main

if __name__ == '__main__':
    main()
