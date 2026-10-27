# Phase 2 validation

The current test suite covers the tracking/media foundation and the side-view measurement layer. See `PHASE2.md` for definitions and assumptions. The pre-cleanup experiment is preserved at Git tag `rotation-experiment-v1` (commit `227b9bb`).

**Result: 50 tests passed on Python 3.11.15.** `git diff --check` is clean. No new runtime dependencies were added for analysis or plotting.

## Synthetic and automated checks

The analysis tests use an artificial skeleton with a known image scale and parabolic hip path, exact source timestamps with a nonzero origin, configurable rotation, configurable temporal spacing, and optional landmark noise.

They verify:

- Positive backward and negative forward rotation for both facing directions.
- Zero cumulative rotation at takeoff, multiple angle wraps, and a known 720° total.
- Exact constant velocity, time-varying angular velocity, and 900°/s rotation sampled at 30 FPS.
- Nonuniform timestamps used directly for derivatives and ballistic fitting.
- Expected parabola apex, gravitational-curvature scale, and calibrated/airtime rise.
- Light local-polynomial noise reduction with raw landmarks and derivatives retained.
- Low-quality torso, missing pose, timestamp gap, and discrete orientation error flags.
- Withholding cumulative rotation after a discontinuity rather than silently repairing it.
- Authoritative manual event times, including between-frame times and manually unset events.
- Tuck hysteresis ordering and explicit calibration/height disagreement.
- Rejection of nonmonotonic source times and explicit estimated-time flags.
- CSV/JSON units, event provenance, smoothing metadata, null values, and paired export rollback.
- Source-time synchronization between video scrubbing and plot seeks; settings changes invalidate export until recalculation.
- Retained image/landmark rotation geometry on non-square images.

The original tests also verify variable-rate decoding/export, portrait display rotation, 240 FPS frame retention, raw CSV missing-detection rows, source protection, cancellation, rendering immutability, and basic GUI navigation.

The native Qt plots, settings panel, and integrated main window were rendered offscreen and visually inspected using **synthetic data**. These screenshots are layout checks, not measurements of the user's videos.

## Real footage inspection

Decoded and visually inspected locally:

| Clip | Oriented dimensions | Decoded frames | Reported average FPS | Last relative frame time |
|---|---:|---:|---:|---:|
| `spidey_back.MOV` | 1080 × 1920 | 80 | 25.676 | 2.633 s |
| `dub_red_thing.MOV` | 1080 × 1920 | 156 | 30.000 | 5.167 s |

Container frame hints and average FPS can differ from decoded content (the first file advertises 95 frames). Analysis therefore uses decoded frames and their actual source PTS, never assumed `1/FPS` intervals.

Visual review windows: approximately 1.0–1.8 seconds for `spidey_back`, and 1.0–2.0 seconds for `dub_red_thing`. These are not exact event labels. The double-back footage has a different landing surface; equal-COM-height airtime assumptions should be inspected rather than presumed.

**Real model-derived kinematic accuracy remains unverified here.** A fresh MediaPipe 0.10.32 initialization attempt failed with `kGpuService / Could not create an NSOpenGLPixelFormat` in this restricted environment. No double-back 720° measurement, real peak velocity, or real calibrated height is claimed. The user has reported that Phase 1 tracking works in the normal desktop session.

## Desktop acceptance check

1. Relaunch the app, process `spidey_back.MOV`, and open Analyze flip.
2. Choose facing direction from the displayed image. Set takeoff and landing by frame stepping; do not accept the rough windows above as precise annotations.
3. Inspect the measured hip proxy against its fit, RMSE, and red quality bands.
4. Check cumulative rotation sign and trajectory. For the double-back clip, expect approximately two turns, allowing endpoint tilt; do not force the output to 720°.
5. Compare raw versus smoothed angular velocity. Adjust the light smoothing window and observe how the reported peak changes.
6. Review/correct tuck start, maximum tuck, opening, and apex events.
7. If using athlete-height calibration, record the standing reference span and source. Inspect disagreements between the two calibrations and airtime-derived rise.
8. Click plots and scrub video in both directions; verify all cursors stay synchronized.
9. Export CSV + JSON and confirm event source times, units, calibration assumptions, smoothing, quality flags, and analysis version are retained.

## Cork benchmark preservation

`benchmarks/cork/` contains eight original source frames with timestamps/source hash and an explicit pending-review manifest. No exported labels or pose-candidate results were found; none were synthesized. The full old experiment code remains recoverable via its Git tag.


Manual-correction validation: frame-local non-destructive overrides and undo/reset;
synthetic damaged shoulders corrected to recover 720° rotation; preserved raw low
confidence with explicit manual placement weight; CSV/JSON correction provenance;
portrait and landscape letterbox drag coordinates with one commit per gesture; stale
analysis invalidation and event-setting retention. Native MediaPipe inference was not
rerun for this change; actual footage should still receive manual desktop inspection.

Group editing validation: click/Shift-click selection, exact marquee selection,
anatomical hand/foot selection and individual-point mode, selection-only clicks,
frame-switch selection clearing, equal-displacement group movement with one undo,
shape-preserving boundary clamping, and atomic rejection of invalid batch coordinates.
