from dataclasses import replace
import numpy as np
import pytest
from movement_tracker.pose_data import PoseFrame, Landmark
from movement_tracker.analysis import analyze, AnalysisParameters

def synthetic(times=None, rate=720, facing="right", noise=0, seed=1):
    elapsed = np.linspace(0, 1, 41) if times is None else np.asarray(times, float)
    rng = np.random.default_rng(seed)
    scale = .005
    frames = []
    for index, dt in enumerate(elapsed):
        hip = np.array([500., -900 + (9.80665 / scale) / 2 * dt - (9.80665 / scale) / 2 * dt ** 2])
        rotation = rate(dt) if callable(rate) else rate * dt
        angle = np.pi / 2 + np.radians(rotation) * (1 if facing == "right" else -1)
        axis = np.array([np.cos(angle), np.sin(angle)])
        shoulder = hip + 100 * axis
        # Open at both ends, tucked in the middle.
        bend = np.radians(170 - 140 * np.sin(np.pi * dt) ** 2)
        knee = hip + 90 * np.array([np.cos(angle + bend), np.sin(angle + bend)])
        points = {}
        for side, offset in (("left", -2), ("right", 2)):
            for name, point in (("hip", hip), ("shoulder", shoulder), ("knee", knee)):
                point = point + np.array([offset, 0]) + rng.normal(0, noise, 2)
                points[f"{side}_{name}"] = Landmark(point[0] / 1000, -point[1] / 1200, visibility=.99, presence=.99)
        frames.append(PoseFrame(float(dt), index, 1000, 1200, points, source_timestamp=100 + float(dt)))
    return frames

def parameters(**kwargs):
    defaults = dict(facing="right", takeoff=100, landing=101, athlete_height_m=1.8,
                    athlete_height_pixels=360, calibration_source="Synthetic known scale")
    return AnalysisParameters(**(defaults | kwargs))

@pytest.mark.parametrize("facing,rate", [("right", 720), ("left", 720), ("right", -720), ("left", -720)])
def test_rotation_sign_unwrap_zero_and_derivatives(facing, rate):
    result = analyze(synthetic(rate=rate, facing=facing), parameters(facing=facing))
    assert result.source_timestamps[0] == 100
    assert result.series["cumulative_rotation_deg"][0] == pytest.approx(0)
    assert result.summary["total_rotation"].value == pytest.approx(rate)
    assert result.summary["average_flight_angular_velocity"].value == pytest.approx(rate)
    assert result.summary["peak_angular_velocity"].value == pytest.approx(rate)
    assert result.summary["rotation_at_apex"].value == pytest.approx(rate / 2)
    assert np.nanmax(np.abs(np.array(result.series["angular_velocity_raw_deg_s"], float) - rate)) < 1e-7

def test_parabola_scale_apex_and_three_rise_estimates():
    result = analyze(synthetic(), parameters())
    assert result.events["apex"].source_timestamp == pytest.approx(100.5)
    assert result.calibration["gravitational_curvature"]["meters_per_pixel"] == pytest.approx(.005)
    assert result.calibration["fit"]["rmse_pixels"] < 1e-8
    for key in ("airtime_implied_rise", "height_calibrated_fit_rise", "curvature_calibrated_fit_rise"):
        assert result.summary[key].value == pytest.approx(9.80665 / 8)

def test_accelerating_rotation_and_900_degrees_per_second_at_30fps():
    times = np.r_[0, np.cumsum(np.random.default_rng(4).uniform(.02, .03, 42))]
    times /= times[-1]
    result = analyze(synthetic(times, rate=lambda t: 600*t + 120*t*t), parameters())
    expected = 600 + 240 * times
    actual = np.asarray(result.series["angular_velocity_deg_s"], float)
    assert np.nanmax(np.abs(actual - expected)) < 1e-7
    assert result.summary["total_rotation"].value == pytest.approx(720)
    assert result.summary["peak_angular_velocity"].value == pytest.approx(840)
    fast = analyze(synthetic(np.linspace(0,1,31), rate=900), parameters())
    assert fast.summary["peak_angular_velocity"].value == pytest.approx(900)
    assert not any("orientation_jump" in f for f in fast.quality_flags)

def test_irregular_timestamps_used_for_fit_and_differentiation():
    rng = np.random.default_rng(2)
    steps = rng.uniform(.015, .035, 45)
    times = np.r_[0, np.cumsum(steps)]
    times /= times[-1]
    result = analyze(synthetic(times), parameters())
    assert result.summary["total_rotation"].value == pytest.approx(720)
    valid = [v for v in result.series["angular_velocity_deg_s"] if v is not None]
    assert np.max(np.abs(np.asarray(valid) - 720)) < 1e-7
    assert result.events["apex"].source_timestamp == pytest.approx(100.5)

def test_noise_smoothed_without_changing_input_or_raw_derivative():
    frames = synthetic(noise=.8)
    original = frames[12].landmarks["left_shoulder"]
    result = analyze(frames, parameters())
    raw = np.asarray(result.series["angular_velocity_raw_deg_s"], float)
    smooth = np.asarray(result.series["angular_velocity_deg_s"], float)
    assert np.std(smooth[3:-3] - 720) < np.std(raw[3:-3] - 720)
    assert result.events["apex"].source_timestamp == pytest.approx(100.5, abs=.005)
    assert result.summary["height_calibrated_fit_rise"].value == pytest.approx(9.80665 / 8, abs=.015)
    assert frames[12].landmarks["left_shoulder"] is original

def test_pose_error_flags_break_cumulative_without_repair():
    frames = synthetic()
    bad = frames[16]
    points = dict(bad.landmarks)
    for side in ("left", "right"):
        s, h = points[f"{side}_shoulder"], points[f"{side}_hip"]
        points[f"{side}_shoulder"] = replace(s, x=2*h.x-s.x, y=2*h.y-s.y)
    frames[16] = replace(bad, landmarks=points)
    result = analyze(frames, parameters())
    assert "orientation_jump" in result.quality_flags[16]
    assert result.summary["total_rotation"].value is None
    assert all(v is None for v in result.series["cumulative_rotation_deg"][16:])
    assert result.series["orientation_raw_deg"][16] is not None
    assert "rotation_continuity_lost" in result.quality_flags[-1]

def test_low_quality_and_timestamp_gap_breaks():
    frames = synthetic()
    frames[12] = replace(frames[12], landmarks={})
    result = analyze(frames, parameters())
    assert "low_torso_quality" in result.quality_flags[12]
    assert result.summary["total_rotation"].value is None
    frames = synthetic(np.r_[np.arange(0, .3, .025), np.arange(.6, 1.001, .025)])
    result = analyze(frames, parameters(landing=frames[-1].source_timestamp))
    assert any("timestamp_gap" in flags for flags in result.quality_flags)
    assert result.summary["total_rotation"].value is None

def test_manual_events_authoritative_and_interpolated_in_source_time():
    result = analyze(synthetic(), parameters(takeoff=100.0125, landing=100.9875,
                     event_overrides={"apex": 100.55, "opening": 100.8, "maximum_tuck": None}))
    assert result.events["apex"].source_timestamp == 100.55
    assert result.events["apex"].provenance == "manual"
    assert result.events["maximum_tuck"].source_timestamp is None
    assert result.summary["total_rotation"].value == pytest.approx(.975 * 720)
    assert result.summary["rotation_at_opening"].value == pytest.approx((.8 - .0125) * 720)

def test_tuck_hysteresis_and_disagreement_are_reported():
    result = analyze(synthetic(), parameters(athlete_height_pixels=720))
    times = [result.events[name].source_timestamp for name in ("tuck_start", "maximum_tuck", "opening")]
    assert all(v is not None for v in times)
    assert times == sorted(times)
    assert result.events["maximum_tuck"].source_timestamp == pytest.approx(100.5)
    assert any("calibration_disagreement" in flags for flags in result.quality_flags)
    assert result.calibration["scale_disagreement_fraction"] == pytest.approx(1)

def test_nonmonotonic_source_timestamps_rejected():
    frames = synthetic()
    frames[5] = replace(frames[5], source_timestamp=frames[4].source_timestamp)
    with pytest.raises(ValueError, match="timestamps"):
        analyze(frames, parameters())

def test_missing_source_time_is_flagged_not_hidden():
    frames = [replace(f, timestamp=f.source_timestamp, source_timestamp=None) for f in synthetic()]
    result = analyze(frames, parameters())
    assert all("source_timestamp_estimated" in f for f in result.quality_flags)

def test_shallow_or_upward_fit_cannot_calibrate_gravity():
    frames = synthetic()
    altered = []
    for frame in frames:
        dt = frame.timestamp
        # Replace the exact original quadratic by an upward parabola.
        shift_pixels = 1961.33 * dt * dt
        points = {name: replace(p, y=p.y - shift_pixels / frame.height) for name, p in frame.landmarks.items()}
        altered.append(replace(frame, landmarks=points))
    result = analyze(altered, parameters())
    assert result.calibration["gravitational_curvature"] is None
    assert result.summary["curvature_calibrated_fit_rise"].value is None
