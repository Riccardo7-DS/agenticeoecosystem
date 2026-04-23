from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# Folder paths
# ---------------------------------------------------------------------------
TASKS_FOLDER = DATA_DIR / "tasks"
GEOBENCH_RESULTS = ROOT_DIR / "output" / "geobenchX"
