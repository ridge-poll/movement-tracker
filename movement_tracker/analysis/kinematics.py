"""Inspectable side-view measurements from PoseFrame data, never pose repair."""
from dataclasses import asdict
import math
import numpy as np
from .models import ANALYSIS_VERSION, EVENT_NAMES, AnalysisParameters, AnalysisResult, Event, Measurement
from .numerics import at_time, local_polynomial, robust_parabola, segments

def source_times(frames):
    return np.asarray([p.source_timestamp if p.source_timestamp is not None else p.timestamp for p in frames], float)

def _validate(frames, p, t):
    if len(frames) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Analysis needs at least three frames with strictly increasing, finite source timestamps.")
    if p.facing not in ("left", "right"):
        raise ValueError("Set facing direction to left or right in the displayed image.")
    if not (t[0] <= p.takeoff < p.landing <= t[-1]):
        raise ValueError("Takeoff and landing must be ordered source times within the clip.")
    if len({(f.width, f.height) for f in frames}) != 1:
        raise ValueError("Image dimensions change within the clip; pixel measurements are not comparable.")
    if not (0 < p.min_confidence <= 1 and 0 < p.max_orientation_step_deg < 180):
        raise ValueError("Invalid quality or orientation-jump threshold.")
    if not (p.max_gap_seconds > 0 and p.min_torso_pixels > 0 and p.smoothing_window_seconds > 0
            and 1 <= p.smoothing_order < p.smoothing_max_points and p.gravity > 0):
        raise ValueError("Invalid smoothing, gap, torso, or gravity parameters.")
    if not (0 < p.tuck_start_fraction < p.opening_fraction < 1):
        raise ValueError("Tuck thresholds must satisfy 0 < start < opening < 1.")
    if (p.athlete_height_m is None) != (p.athlete_height_pixels is None):
        raise ValueError("Athlete height calibration needs both height in meters and a measured pixel span.")
    if p.athlete_height_m is not None:
        if not math.isfinite(p.athlete_height_m) or not math.isfinite(p.athlete_height_pixels) or p.athlete_height_m <= 0 or p.athlete_height_pixels <= 0 or not p.calibration_source:
            raise ValueError("Height calibration needs positive values and a documented source/reference.")
    for name, value in p.event_overrides.items():
        if name not in EVENT_NAMES or name in ("takeoff", "landing"):
            raise ValueError("Override only apex, tuck_start, maximum_tuck, or opening.")
        if value is not None and not p.takeoff <= value <= p.landing:
            raise ValueError(f"{name} must lie between takeoff and landing in source-video time.")

def _point(frame, name):
    p = frame.landmarks.get(name)
    if p is None or not (math.isfinite(p.x) and math.isfinite(p.y)):
        return None
    q = p.confidence
    q = float(np.clip(q, 0, 1)) if q is not None and math.isfinite(q) else 0.0
    if name in frame.manual_landmarks:
        q = 1.0  # User placement weight, not detector confidence.
    return np.array([p.x * frame.width, -p.y * frame.height]), q

def _midpoint(frame, joint, threshold):
    points = [p for side in ("left", "right") if (p := _point(frame, f"{side}_{joint}")) is not None]
    reliable = [p for p in points if p[1] >= threshold]
    use = reliable or points
    if not use:
        return None, 0.0, 0
    weights = np.asarray([max(.001, p[1]) for p in use])
    location = np.average(np.asarray([p[0] for p in use]), axis=0, weights=weights)
    quality = float(np.average([p[1] for p in use], weights=weights))
    return location, quality, len(reliable)

def _finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in values]

def analyze(frames, params: AnalysisParameters, source_metadata=None) -> AnalysisResult:
    frames = tuple(frames)
    t = source_times(frames)
    p = params
    _validate(frames, p, t)
    n = len(frames)
    flight = (t >= p.takeoff) & (t <= p.landing)
    flags = [{"manual_landmark:" + name for name in frame.manual_landmarks} for frame in frames]
    warnings = ["Hip midpoint is a body-center proxy, not center of mass.",
                "Assumed side view: camera tilt, perspective, and out-of-plane rotation affect these measurements."]
    raw = np.full(n, np.nan)
    vertical = np.full(n, np.nan)
    torso = np.full(n, np.nan)
    tuck = np.full(n, np.nan)
    hip_angle = np.full(n, np.nan)
    quality, hip_quality = np.zeros(n), np.zeros(n)
    sign = 1 if p.facing == "right" else -1
    for i, frame in enumerate(frames):
        if frame.source_timestamp is None or frame.timestamp_estimated:
            flags[i].add("source_timestamp_estimated")
        if not frame.landmarks:
            flags[i].add("no_pose")
        shoulder, sq, scount = _midpoint(frame, "shoulder", p.min_confidence)
        hip, hq, hcount = _midpoint(frame, "hip", p.min_confidence)
        hip_quality[i] = hq
        if hip is not None:
            vertical[i] = hip[1]
        if hq < p.min_confidence:
            flags[i].add("low_hip_quality")
        if shoulder is not None and hip is not None:
            axis = hip - shoulder  # shoulder → hip; image right/up coordinates
            torso[i] = np.linalg.norm(axis)
            raw[i] = sign * np.degrees(np.arctan2(axis[1], axis[0]))
            quality[i] = min(sq, hq)
            if scount == 1 or hcount == 1:
                flags[i].add("single_side_torso")
            if torso[i] < p.min_torso_pixels:
                quality[i] = 0
                flags[i].add("torso_too_short")
        if quality[i] < p.min_confidence:
            flags[i].add("low_torso_quality")
        metrics, angles, weights = [], [], []
        for side in ("left", "right"):
            triplet = [_point(frame, f"{side}_{joint}") for joint in ("shoulder", "hip", "knee")]
            if any(v is None for v in triplet):
                continue
            (s, qs), (h, qh), (k, qk) = triplet
            q = min(qs, qh, qk)
            length = np.linalg.norm(s - h)
            leg_length = np.linalg.norm(k - h)
            if q < p.min_confidence or length < p.min_torso_pixels or leg_length < 1:
                continue
            metrics.append(np.linalg.norm(s - k) / length)
            angles.append(np.degrees(np.arccos(np.clip(np.dot(s - h, k - h) / (length * leg_length), -1, 1))))
            weights.append(q)
        if metrics:
            tuck[i] = np.average(metrics, weights=weights)
            hip_angle[i] = np.average(angles, weights=weights)
        else:
            flags[i].add("tuck_unavailable")

    valid = np.isfinite(raw) & (quality >= p.min_confidence)
    breaks = np.zeros(n, bool)
    for i in range(1, n):
        if t[i] - t[i - 1] > p.max_gap_seconds:
            breaks[i] = True
            flags[i].add("timestamp_gap")
        if valid[i] and valid[i - 1]:
            delta = (raw[i] - raw[i - 1] + 180) % 360 - 180
            if abs(delta) > p.max_orientation_step_deg:
                breaks[i] = True
                flags[i].add("orientation_jump")
        if np.isfinite(torso[[i - 1, i]]).all() and min(torso[i - 1], torso[i]) > 0:
            if max(torso[i - 1], torso[i]) / min(torso[i - 1], torso[i]) > 2:
                flags[i].add("torso_length_change")
    unwrapped, cumulative, raw_velocity, velocity = [np.full(n, np.nan) for _ in range(4)]
    segment_id = np.full(n, np.nan)
    segment_groups = list(segments(valid, breaks))
    takeoff_segment = None
    for number, indices in enumerate(segment_groups):
        unwrapped[indices] = np.degrees(np.unwrap(np.radians(raw[indices])))
        segment_id[indices] = number
        if len(indices) > 1:
            raw_velocity[indices] = np.gradient(unwrapped[indices], t[indices])
            _, velocity[indices] = local_polynomial(t[indices], unwrapped[indices],
                                                    p.smoothing_window_seconds, p.smoothing_max_points,
                                                    p.smoothing_order)
        if t[indices[0]] <= p.takeoff <= t[indices[-1]]:
            baseline = at_time(t[indices], unwrapped[indices], p.takeoff, p.max_gap_seconds)
            if baseline is not None:
                takeoff_segment = number
                after = indices[t[indices] >= p.takeoff]
                cumulative[after] = unwrapped[after] - baseline
    for i in np.flatnonzero(flight):
        if not np.isfinite(cumulative[i]):
            flags[i].add("rotation_continuity_lost")
        if valid[i] and not np.isfinite(velocity[i]):
            flags[i].add("insufficient_derivative_support")
    if any("rotation_continuity_lost" in flags[i] for i in np.flatnonzero(flight)):
        warnings.append("Flight rotation continuity is broken; cumulative rotation is withheld beyond the break. No turns were repaired or inferred.")
    # Values exactly at event times can fall between frames; never bridge a bad segment.
    cumulative_for_events = cumulative.copy()
    if takeoff_segment is not None:
        group = segment_groups[takeoff_segment]
        baseline = at_time(t[group], unwrapped[group], p.takeoff, p.max_gap_seconds)
        cumulative_for_events[group] = unwrapped[group] - baseline

    T = p.landing - p.takeoff
    events = {name: Event(None, "unavailable") for name in EVENT_NAMES}
    events["takeoff"] = Event(p.takeoff, "manual", "User-specified source-video time")
    events["landing"] = Event(p.landing, "manual", "User-specified source-video time")
    fit = np.full(n, np.nan)
    residual = np.full(n, np.nan)
    fit_weights = np.full(n, np.nan)
    fit_rise, curvature_scale, apex = None, None, None
    calibration = {"athlete_height": None, "gravitational_curvature": None,
                   "airtime_rise_assumption": "Equal COM height at takeoff and landing; not a hip/COM height measurement",
                   "fit": None}
    metric_scale = None
    if p.athlete_height_m is not None:
        metric_scale = p.athlete_height_m / p.athlete_height_pixels
        calibration["athlete_height"] = {"method": "athlete_height_over_standing_pixel_span", "meters_per_pixel": metric_scale, "height_m": p.athlete_height_m,
                                         "pixel_span": p.athlete_height_pixels, "source": p.calibration_source,
                                         "reference_source_timestamp": p.calibration_timestamp,
                                         "assumptions": "Standing full-height span in the same depth plane; fixed camera and scale"}
    fit_valid = flight & np.isfinite(vertical) & (hip_quality >= p.min_confidence)
    fit_indices = np.flatnonzero(fit_valid)
    if len(fit_indices) >= 5 and np.ptp(t[fit_indices]) >= .1:
        coefficients, errors, weights = robust_parabola(t[fit_indices] - p.takeoff,
                                                        vertical[fit_indices], hip_quality[fit_indices])
        c, b, a = map(float, coefficients)
        fit[flight] = np.polynomial.polynomial.polyval(t[flight] - p.takeoff, coefficients)
        residual[fit_valid] = errors
        fit_weights[fit_valid] = weights
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        median_torso = float(np.nanmedian(torso[fit_indices])) if np.isfinite(torso[fit_indices]).any() else p.min_torso_pixels
        residual_limit = max(2.0, median_torso * p.fit_residual_torso_fraction)
        calibration["fit"] = {"method": "quality_weighted_Huber_quadratic", "robust_iterations": 4,
                              "coefficient_order": "constant, linear, quadratic", "coefficients": [c, b, a],
                              "time_origin_source_seconds": p.takeoff, "coordinate": "hip_midpoint_y_up_pixels",
                              "sample_count": len(fit_indices), "rmse_pixels": rmse,
                              "max_abs_residual_pixels": float(np.max(np.abs(errors))),
                              "residual_warning_threshold_pixels": residual_limit}
        for index, error in zip(fit_indices, errors):
            if abs(error) > residual_limit:
                flags[index].add("large_ballistic_residual")
        if rmse > residual_limit:
            warnings.append("Ballistic fit residual is large relative to torso length; inspect the hip proxy and flight boundaries.")
        if a < -1e-6:
            apex = p.takeoff - b / (2 * a)
            curvature_scale = p.gravity / (-2 * a)
            calibration["gravitational_curvature"] = {
                "method": "gravity_over_negative_fitted_pixel_acceleration", "meters_per_pixel": curvature_scale, "pixel_acceleration": 2 * a,
                "gravity_m_s2": p.gravity, "source": "hip_proxy_flight_quadratic",
                "assumptions": "Ballistic proxy, fixed camera, vertical side view, constant perspective scale; exploratory"}
            if p.takeoff <= apex <= p.landing:
                events["apex"] = Event(apex, "ballistic_fit", "Apex of fitted hip proxy, not measured COM")
                fit_rise = float(np.polynomial.polynomial.polyval(apex - p.takeoff, coefficients) - c)
            else:
                warnings.append("Fitted apex lies outside the selected flight interval.")
        else:
            warnings.append("Vertical curvature is not downward; gravitational calibration and fitted apex are unavailable.")
    else:
        warnings.append("Fewer than five reliable hip samples or insufficient time span: ballistic fit unavailable.")

    # Initial hysteresis event suggestions. Their low-confidence nature is explicit.
    tuck_indices = np.flatnonzero(flight & np.isfinite(tuck))
    if len(tuck_indices) >= 5:
        minimum_index = tuck_indices[np.argmin(tuck[tuck_indices])]
        entry = tuck_indices[t[tuck_indices] <= p.takeoff + .2 * T]
        if len(entry):
            baseline = float(np.median(tuck[entry]))
            minimum = float(tuck[minimum_index])
            depth = baseline - minimum
            if depth > max(.1, .1 * abs(baseline)):
                events["maximum_tuck"] = Event(float(t[minimum_index]), "tuck_metric_minimum", "Heuristic; review manually")
                low = minimum + p.tuck_start_fraction * depth
                high = minimum + p.opening_fraction * depth
                for i in tuck_indices:
                    if i + 1 >= n or not flight[i + 1] or t[i + 1] - t[i] > p.max_gap_seconds:
                        continue
                    if i <= minimum_index and events["tuck_start"].source_timestamp is None and tuck[i] <= low and tuck[i + 1] <= low:
                        events["tuck_start"] = Event(float(t[i]), "tuck_hysteresis", "Two adjacent samples below entry threshold; review manually")
                    if i > minimum_index and tuck[i] >= high and tuck[i + 1] >= high:
                        events["opening"] = Event(float(t[i]), "tuck_hysteresis", "Two adjacent samples above opening threshold; review manually")
                        break
    for name, value in p.event_overrides.items():
        events[name] = Event(value, "manual" if value is not None else "manually_unset")
    tuck_times = [events[name].source_timestamp for name in ("tuck_start", "maximum_tuck", "opening")]
    ordered = [value for value in tuck_times if value is not None]
    if ordered != sorted(ordered):
        warnings.append("Tuck events are out of chronological order; inspect manual event edits.")

    airtime_rise = p.gravity * T ** 2 / 8
    summary = {
        "airtime": Measurement(T, "s", "From manually set takeoff and landing"),
        "airtime_implied_rise": Measurement(airtime_rise, "m", "Equal COM heights at takeoff/landing assumed; not measured hip/COM rise"),
        "fitted_hip_proxy_rise_pixels": Measurement(fit_rise, "px", "Fitted rise above fitted takeoff position"),
        "height_calibrated_fit_rise": Measurement(None if fit_rise is None or metric_scale is None else fit_rise * metric_scale, "m", "Hip-proxy fit with athlete-height calibration"),
        "curvature_calibrated_fit_rise": Measurement(None if fit_rise is None or curvature_scale is None else fit_rise * curvature_scale, "m", "Exploratory gravitational self-calibration of hip proxy"),
        "total_rotation": Measurement(at_time(t, cumulative_for_events, p.landing, p.max_gap_seconds), "deg", "Positive backward, negative forward; unavailable after continuity loss"),
    }
    for name in ("apex", "opening"):
        time = events[name].source_timestamp
        summary[f"rotation_at_{name}"] = Measurement(None if time is None else at_time(t, cumulative_for_events, time, p.max_gap_seconds), "deg")
    good_velocity = np.flatnonzero(flight & np.isfinite(velocity))
    peak = float(velocity[good_velocity[np.argmax(np.abs(velocity[good_velocity]))]]) if len(good_velocity) else None
    summary["peak_angular_velocity"] = Measurement(peak, "deg/s", "Signed maximum magnitude over valid local fits; see smoothing parameters; not a peak across missing data")
    total = summary["total_rotation"].value
    summary["average_flight_angular_velocity"] = Measurement(None if total is None else total / T, "deg/s", "Total continuous rotation / airtime")
    for name in ("tuck_start", "maximum_tuck", "opening"):
        summary[f"{name}_source_time"] = Measurement(events[name].source_timestamp, "s", "Source-video time")
    if calibration["fit"] is not None:
        summary["ballistic_fit_rmse"] = Measurement(calibration["fit"]["rmse_pixels"], "px", "Includes all quality-eligible residuals, not only robust inliers")
    else:
        summary["ballistic_fit_rmse"] = Measurement(None, "px")
    if metric_scale is not None and curvature_scale is not None:
        disagreement = abs(metric_scale - curvature_scale) / metric_scale
        calibration["scale_disagreement_fraction"] = disagreement
        if disagreement > p.calibration_disagreement_fraction:
            warnings.append("Athlete-height and gravitational-curvature calibrations disagree beyond the configured tolerance.")
            for i in np.flatnonzero(flight):
                flags[i].add("calibration_disagreement")
    height_values = [summary[name].value for name in ("height_calibrated_fit_rise", "curvature_calibrated_fit_rise")]
    for label, value in zip(("athlete-height", "curvature"), height_values):
        if value is not None and abs(value - airtime_rise) / max(.001, airtime_rise) > p.calibration_disagreement_fraction:
            warnings.append(f"The {label} fitted hip rise disagrees with equal-height airtime rise; the proxy and/or equal-height assumptions may not hold.")
            for i in np.flatnonzero(flight):
                flags[i].add("height_estimates_disagree")
    if any("source_timestamp_estimated" in f for f in flags):
        warnings.append("Some timestamps were estimated; source-time accuracy is limited for those samples.")

    baseline_vertical = at_time(t, np.where(hip_quality >= p.min_confidence, vertical, np.nan), p.takeoff, p.max_gap_seconds)
    displacement = vertical - baseline_vertical if baseline_vertical is not None else np.full(n, np.nan)
    series = {
        "hip_vertical_px": vertical, "hip_displacement_from_takeoff_px": displacement,
        "fitted_hip_vertical_px": fit, "fit_residual_px": residual, "fit_weight": fit_weights,
        "hip_quality": hip_quality, "torso_length_px": torso, "orientation_quality": quality,
        "orientation_raw_deg": raw, "orientation_unwrapped_deg": unwrapped,
        "orientation_segment": segment_id, "cumulative_rotation_deg": cumulative,
        "angular_velocity_raw_deg_s": raw_velocity, "angular_velocity_deg_s": velocity,
        "tuck_ratio": tuck, "hip_angle_deg": hip_angle,
        "hip_displacement_athlete_m": displacement * metric_scale if metric_scale is not None else np.full(n, np.nan),
    }
    units = {key: ("deg/s" if key.endswith("deg_s") else "deg" if key.endswith("deg") else
                   "px" if key.endswith("px") else "m" if key.endswith("_m") else "1") for key in series}
    units["source_timestamp"] = "s"
    return AnalysisResult(ANALYSIS_VERSION, t.tolist(), [f.frame_index for f in frames],
                          {key: _finite_list(value) for key, value in series.items()}, events,
                          [sorted(f) for f in flags], units, calibration, asdict(p),
                          {"method": "local_polynomial_on_actual_timestamps", "window_seconds": p.smoothing_window_seconds,
                           "max_points": p.smoothing_max_points, "polynomial_order": p.smoothing_order,
                           "boundary": "available one-sided samples within half-window", "crosses_quality_breaks": False},
                          p.facing, p.viewpoint, summary, warnings,
                          {**(source_metadata or {}), "timestamp_basis": "source_video_PTS_seconds_with_flagged_fallback",
                           "vertical_axis": "upward pixels; negative source image y", "body_axis": "shoulder midpoint toward hip midpoint",
                           "rotation_sign": "backward positive; forward negative; facing is fixed for the clip",
                           "unwrapped_orientation": "segment-local after quality breaks; see orientation_segment",
                           "pose_correction_applied": any(f.manual_landmarks for f in frames),
                           "manual_landmarks_by_frame": {str(f.frame_index): list(f.manual_landmarks) for f in frames if f.manual_landmarks},
                           "manual_quality_policy": "User-placed XY has analysis weight 1; original detector confidence and Z are unchanged", "body_center_proxy": "visibility-weighted hip midpoint, not COM"})
