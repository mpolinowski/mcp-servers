"""Image preprocessing for the extraction retry loop.

- ``config``       — the three agent-tunable knobs + the 4-attempt budget.
- ``engine``       — OpenCV pipeline (downscale, shadow removal, CLAHE,
                     sharpen, small-angle deskew). Pure functions, no disk.
- ``quality``      — deterministic score + metrics for one attempt.
- ``attempt_store``— SQLite history (all attempts, the winner, paths).
"""

from .attempt_store import AttemptStore
from .config import DEFAULTS, FIELD_RANGES, MAX_ATTEMPTS, PreprocessConfig
from .engine import encode_image, load_image, process
from .quality import (
    OUTCOME_ERROR,
    OUTCOME_MANUAL_REVIEW,
    OUTCOME_RETRYABLE,
    OUTCOME_SUCCESS,
    SUCCESS_THRESHOLD,
    QualityReport,
    assess,
)

__all__ = [
    "AttemptStore",
    "DEFAULTS",
    "FIELD_RANGES",
    "MAX_ATTEMPTS",
    "PreprocessConfig",
    "encode_image",
    "load_image",
    "process",
    "OUTCOME_MANUAL_REVIEW",
    "OUTCOME_RETRYABLE",
    "OUTCOME_SUCCESS",
    "SUCCESS_THRESHOLD",
    "QualityReport",
    "assess",
]
