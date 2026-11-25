"""Preprocessing knobs and the attempt budget for the extraction retry loop.

The agent gets exactly three tunable knobs — deliberately small, so a retry
search stays bounded and predictable. Everything else in the pipeline
(shadow removal, downscale, deskew) is fixed inside the engine. ``RetryChanges``
is a partial config: on retry the fields the agent leaves out keep the
previous attempt's values, so each call only lists what actually changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

#: Hard cap for ONE cycle (extract + retries) per source image. Attempt 1 is
#: the extract call itself; the budget is consumed per resolved source path,
#: so extracting the same file again starts a fresh cycle.
MAX_ATTEMPTS = 4

#: name -> (low, high) — both bounds inclusive. A value outside this range is
#: rejected at the boundary, so the engine never sees garbage.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "clahe_clip": (1.0, 4.0),
    "clahe_tile": (4, 16),
    "sharpen_strength": (0.0, 0.8),
}

#: Baseline values — what attempt #1 runs with when the agent sends no changes.
DEFAULTS: dict[str, object] = {
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "sharpen_strength": 0.0,
}


@dataclass(frozen=True)
class PreprocessConfig:
    """One concrete knob set for one attempt.

    Frozen and value-based on purpose: an attempt's config is recorded as-is
    in the attempt history, and two different attempts must never share
    mutable state.

    Defaults aim at "safe for a typical photographed invoice": mild CLAHE
    contrast, fine tile grid, no edge sharpening (sharpening amplifies sensor
    noise on low-light photos).
    """

    clahe_clip: float = DEFAULTS["clahe_clip"]  # type: ignore[valid-type]
    clahe_tile: int = DEFAULTS["clahe_tile"]  # type: ignore[valid-type]
    sharpen_strength: float = DEFAULTS["sharpen_strength"]  # type: ignore[valid-type]

    def __post_init__(self) -> None:
        # Type gate before the range gate: an LLM may send "8" or 8.0 for the
        # tile grid. Integral floats are coerced; anything else is rejected
        # with a useful message instead of a comparison TypeError.
        tile = self.clahe_tile
        if isinstance(tile, float) and tile.is_integer():
            tile = int(tile)
            object.__setattr__(self, "clahe_tile", tile)
        if isinstance(tile, bool) or not isinstance(tile, int):
            raise ValueError(f"clahe_tile must be an integer grid size, got {self.clahe_tile!r}")
        for name, (lo, hi) in FIELD_RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (lo <= value <= hi):
                raise ValueError(f"{name}={value!r} outside allowed range [{lo}, {hi}]")

    @property
    def is_default(self) -> bool:
        """True when every knob is at its default (attempt #1 baseline)."""
        for name, value in DEFAULTS.items():
            if getattr(self, name) != value:
                return False
        return True

    @classmethod
    def from_dict(cls, changes: dict | None) -> "PreprocessConfig":
        """Defaults + ``changes``. ``changes`` may be empty (pure defaults).

        Every supplied key must be a known knob in range — unknown keys and
        out-of-range values raise ``ValueError`` so the model gets a useful
        error instead of a silently wrong pipeline.
        """
        changes = dict(changes or {})
        unknown = sorted(set(changes) - set(FIELD_RANGES))
        if unknown:
            raise ValueError(f"unknown preprocessing knob(s) {unknown}; allowed: {sorted(FIELD_RANGES)}")
        merged = {f.name: f.default for f in dataclasses.fields(cls)}
        merged.update(changes)
        return cls(**merged)

    def with_overrides(self, changes: dict) -> "PreprocessConfig":
        """This config + a retry diff (cumulative tuning, not an absolute reset).

        ``changes`` must be non-empty — a retry that changes nothing is a
        no-op the agent is trying to hide. Fields not listed keep this
        attempt's values.
        """
        if not changes:
            raise ValueError("retry_changes must set at least one knob")
        unknown = sorted(set(changes) - set(FIELD_RANGES))
        if unknown:
            raise ValueError(f"unknown preprocessing knob(s) {unknown}; allowed: {sorted(FIELD_RANGES)}")
        merged = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        merged.update(changes)
        return PreprocessConfig(**merged)
