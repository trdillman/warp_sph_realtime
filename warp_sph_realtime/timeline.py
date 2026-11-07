"""Timeline targets in seconds. No Blender or GPU dependencies."""

import math


def needs_step(frame, start, fps, elapsed, step_seconds):
    """Integrate to the requested frame, with less than one step of overshoot.

    A solver step includes all its substeps. Never multiply the target duration
    by the substep count. The initial state belongs to the start frame.
    """
    if not all(math.isfinite(v) for v in (fps, elapsed, step_seconds)) or fps <= 0 or step_seconds <= 0:
        raise ValueError("timeline requires positive finite FPS and solver step duration")
    if step_seconds > 1.0 / fps:
        raise ValueError("Reduce Time Step or Substeps: one solver step exceeds a timeline frame")
    return elapsed + 1e-10 < max(0, frame - start) / fps
