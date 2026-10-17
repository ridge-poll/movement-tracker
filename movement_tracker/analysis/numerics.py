"""Time-aware local polynomials and robust ballistic fits."""
import numpy as np

def local_polynomial(times, values, window_seconds=.15, max_points=5, order=2):
    """One contiguous finite segment. Fit around each actual timestamp.

    No resampling to a fixed FPS; no extrapolation outside the observed segment.
    Endpoints use the available one-sided neighborhood, still within half-window.
    """
    t, y = np.asarray(times, float), np.asarray(values, float)
    fitted, derivative = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i, time in enumerate(t):
        indices = np.flatnonzero(np.abs(t - time) <= window_seconds / 2 + 1e-12)
        indices = indices[np.argsort(np.abs(t[indices] - time), kind="stable")[:max_points]]
        if len(indices) < order + 1:
            continue
        dt = t[indices] - time
        design = np.polynomial.polynomial.polyvander(dt, order)
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[indices], rcond=None)
        if rank == order + 1:
            fitted[i], derivative[i] = coefficients[0], coefficients[1]
    return fitted, derivative

def robust_parabola(times_since_takeoff, values, quality):
    t, y, q = map(lambda x: np.asarray(x, float), (times_since_takeoff, values, quality))
    design = np.polynomial.polynomial.polyvander(t, 2)
    weights = np.clip(q, .001, 1).copy()
    coefficients = None
    for _ in range(4):
        used_weights = weights.copy()
        root = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        if rank != 3:
            raise ValueError("Ballistic fit is rank deficient")
        residuals = y - design @ coefficients
        scale = max(.5, 1.4826 * np.median(np.abs(residuals - np.median(residuals))))
        weights = q * np.minimum(1, 1.5 * scale / np.maximum(np.abs(residuals), 1e-12))
    residuals = y - design @ coefficients
    return coefficients, residuals, used_weights

def segments(mask, breaks):
    current = []
    for i, valid in enumerate(mask):
        if not valid or breaks[i]:
            if current:
                yield np.asarray(current, int)
            current = []
        if valid:
            current.append(i)
    if current:
        yield np.asarray(current, int)

def at_time(times, values, time, max_gap=.15):
    """Interpolate event values only between adjacent valid observations."""
    t, y = np.asarray(times), np.asarray(values)
    right = int(np.searchsorted(t, time))
    if right < len(t) and abs(t[right] - time) < 1e-9:
        return float(y[right]) if np.isfinite(y[right]) else None
    if right == 0 or right == len(t):
        return None
    left = right - 1
    if t[right] - t[left] > max_gap or not np.isfinite(y[[left, right]]).all():
        return None
    return float(np.interp(time, t[left:right + 1], y[left:right + 1]))
