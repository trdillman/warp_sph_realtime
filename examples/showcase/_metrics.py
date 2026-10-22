"""Correctness-only host probes of cached particles, outside accelerated transport timing."""
import numpy as np


def paddle_influence(stationary, animated) -> dict:
    """Signed tangential velocity change for matched IDs near the paddle, in SI units.

    The same gravity, emitter and initial state must be used in both bakes. Sample
    before the pool covers the paddle. Threshold 0.02 m/s is below 4% of paddle
    tip speed (~0.63 m/s); at least 20 matched particles prevents sparse outliers.
    This measures influence, not full physical accuracy or transport performance.
    """
    if len(np.unique(stationary.id)) != len(stationary.id) or len(np.unique(animated.id)) != len(animated.id):
        return {"ok": False, "reason": "duplicate particle ids"}
    _, left, right = np.intersect1d(stationary.id, animated.id, return_indices=True)
    pos = stationary.position[left]
    radius = np.linalg.norm(pos[:, :2], axis=1)
    near = (radius >= 0.05) & (radius <= 0.30) & (pos[:, 2] >= 0.035) & (pos[:, 2] <= 0.285)
    delta = animated.velocity[right] - stationary.velocity[left]
    tangent = np.column_stack((-pos[:, 1], pos[:, 0])) / np.maximum(radius[:, None], 1e-12)
    signed = np.sum(delta[:, :2] * tangent, axis=1)[near]
    finite = bool(np.isfinite(pos).all() and np.isfinite(delta).all())
    mean = float(signed.mean()) if len(signed) else 0.0
    return {"ok": finite and len(signed) >= 20 and mean > 0.02,
            "matched_near_paddle": len(signed), "mean_tangential_delta_m_s": mean,
            "threshold_m_s": 0.02, "finite": finite}


def same_initial_state(a, b) -> bool:
    """Require the same pool/emitter particles at frame 1, independent of solver sorting."""
    ia, ib = np.argsort(a.id), np.argsort(b.id)
    return (len(np.unique(a.id)) == len(a.id) and len(np.unique(b.id)) == len(b.id)
            and np.array_equal(a.id[ia], b.id[ib])
            and np.allclose(a.position[ia], b.position[ib], atol=1e-6, rtol=0)
            and np.allclose(a.velocity[ia], b.velocity[ib], atol=1e-6, rtol=0))
