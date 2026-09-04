import sys
from pathlib import Path

# Add submission_package to sys.path so pytest finds physics_study_buddy package automatically
ROOT = Path(__file__).resolve().parent
SUBMISSION_PACKAGE = ROOT / "submission_package"

if str(SUBMISSION_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_PACKAGE))
