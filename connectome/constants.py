import json
import re
from pathlib import Path

# Modes of transportation
MODES = [
    "CAR",
    "TRANSIT",
    "WALK",
    "BICYCLE",
    "RIDEHAIL",
]

# Exponential decay parameter for accessibility value calculation
DECAY_RATE = 0.05

# Path to traffic analysis parameters JSON
TRAFFIC_PARAMS_PATH = Path(__file__).parent / "traffic_utils" / "traffic_analysis_parameters.json"

# Regex for validating facility keys (toll exemptions, etc.)
_SAFE_FACILITY_RE = re.compile(r"^[A-Za-z0-9_]+$")


def load_traffic_params(path: str | Path = TRAFFIC_PARAMS_PATH) -> dict:
    """Load traffic analysis parameters (functional classes + clamps)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
