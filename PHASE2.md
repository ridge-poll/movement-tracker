# Phase 2: inspectable side-view flip measurements

The target is clean, approximately side-view front/back flips, especially doubles. This version measures already-tracked motion. It does not implement coaching, reference trajectories, automatic temporal pose correction, candidate selection, or four-orientation recovery.

## Start

Use the existing environment and launch command:

```bash
python -m movement_tracker
```

1. Open a clip and let every frame finish processing.
2. Click **Analyze flip…**. The measurement panel can be docked or floated using its title bar.
3. In **Events & settings**, choose the athlete's facing direction **before takeoff**, in the displayed unmirrored video.
4. Scrub/step to takeoff and click its **Set from video** button. Do the same for landing. Define your contact convention consistently; the first airborne / first contact frames are a reasonable starting point, but discrete video limits precision. Both boundaries must be explicitly selected or edited.
5. Click **Analyze / Recalculate**. Inspect all four synchronized plots and the **Summary & quality** tab.
6. Review the suggested apex/tuck events. Change an event to Manual and type its source time, or set it from the current video frame. Use Unset when the event is not identifiable.
7. Export **analysis CSV + JSON**. The regular main-window CSV button continues to export raw landmarks.

Click a plot to seek the nearest source frame. The video scrubber and frame buttons move the white position line on all plots. **Focus plots on flight** displays the selected interval with a little context; turn it off to show the entire clip. The label above the plots always gives the current source time, even when it is outside the focused interval. Red bands and hover text identify questionable frames. Settings changes mark displayed results stale and disable export until recalculated.

## Time and data contract

`analysis.analyze(pose_frames, AnalysisParameters(...))` depends on standardized `PoseFrame` data and NumPy, not MediaPipe or Qt. `AnalysisResult.analysis_version` is **2.1.0**. It contains:

- Source timestamps and frame indices.
- Derived time series and per-frame quality flags.
- Source-time events with provenance (`manual`, `ballistic_fit`, heuristic tuck, unavailable, or manually unset).
- Units, calibration methods/sources, full parameters, smoothing settings, facing direction, viewpoint, summary metrics, and warnings.

Source PTS seconds are authoritative. They can have a nonzero origin and need not have constant spacing. Neither derivatives nor fitting use `1/FPS`. Strictly non-increasing source times are rejected. Missing/previously estimated times use the existing fallback and receive an explicit flag. Event frame flags are conveniences; JSON event source times retain the authoritative values, including between-frame manual times.

The UI shows six decimals when editing timestamps to preserve frame identity, but summary precision is deliberately modest: seconds/meters to two decimals and angles/speeds to whole degrees. Exported numerical precision is for reproducibility, not a claim of measurement accuracy.

## Hip trajectory and height

The initial body-center proxy is the visibility-weighted hip midpoint. **It is not center of mass.** Hip location relative to true COM changes during tuck/opening.

Image coordinates are converted to pixels with +x right and +y up. `hip_vertical_px` is negative image y, so it is typically negative; increasing values mean upward motion. A displacement-from-takeoff series is also retained. The plot overlays the measured proxy and its fitted ballistic curve.

The flight model is `y = c + b·τ + a·τ²`, with `τ = source_time − takeoff`. The fit uses at least five reliable hip samples spanning at least 0.1 seconds. Four quality-weighted Huber least-squares passes reduce the influence of gross outliers. **No pose or measured trajectory is replaced.** Coefficients, weights, sample count, residual series, RMSE, and maximum residual are retained. RMSE includes all quality-eligible samples, not only robust inliers.

For downward curvature, fitted apex is `takeoff − b/(2a)`, provided it falls inside flight. Suggested apex is a hip-proxy fit apex, not ground-truth COM apex. Manual adjustment remains available. Fitted rise is relative to the fitted takeoff position.

Three height estimates remain distinct:

1. **Athlete-height calibrated fit rise:** manually supplied stature / measured full-height pixel span gives meters per pixel. Use an upright reference at the same depth/scale; document the reference in the text field. The application does not infer full stature from nose/ankle landmarks. Both values and a calibration source are required.
2. **Gravitational-curvature calibrated fit rise:** `meters_per_pixel = g / (−2a)`, using `g = 9.80665 m/s²`. This is exploratory self-calibration of the hip proxy under fixed-camera, approximately vertical side-view, constant-scale assumptions. It is not independent of the fitted trajectory.
3. **Airtime-implied rise:** `g·T²/8`. This assumes approximately equal **COM** heights at takeoff and landing. It is an assumption-based estimate, not a measured COM/hip height.

The summary shows each available result and fit RMSE separately. The default relative disagreement threshold is 25%; disagreements between scale estimates or between fitted/airtime rise produce explicit warnings and flight-frame flags. Raw residuals exceeding 15% of median torso length (minimum two pixels) are flagged. These are configurable initial heuristics, not validated accuracy guarantees. Poor fit or non-downward curvature makes unavailable quantities explicit rather than inventing a height.

## Rotation and quality

The body axis runs **shoulder midpoint → hip midpoint**, using reliable landmark sides preferentially and visibility weighting. Orientation is `atan2(axis_y_up, axis_x_right)`, multiplied by +1 for facing right and −1 for facing left.

- Backward rotation is positive; forward rotation is negative.
- Raw angle is wrapped; unwrapped angle uses `np.unwrap` inside each reliable continuous segment.
- Cumulative rotation is zeroed at the selected takeoff time. Event values between adjacent reliable observations use timestamp-based linear interpolation.
- A wrapped step greater than **90° per frame** (adjustable), a low-quality/missing torso, or a time gap greater than 0.15 seconds breaks continuity. The raw angle remains visible/exported. Local segments remain available, but cumulative rotation after a break is withheld because the number of missed turns is uncertain.
- `orientation_segment` identifies segment-local unwrapped angles; a later segment is not silently aligned to an earlier one. Angular-velocity fits never cross a segment boundary.

Per-frame flags include low hip/torso quality, no pose, a short torso, a single visible torso side, large torso-length changes, orientation jumps, timestamp gaps/estimates, lost rotation continuity, insufficient derivative support, fit residuals, and calibration disagreement. A confidently wrong pose can still evade these checks. Projected limb lengths are not forced to remain constant, and no landmark is repaired.

Total rotation is the observed continuous body-axis change between takeoff and landing. A clean double should be in the vicinity of 720°, but endpoint body tilt, event choices, projection, and measurement errors can shift that value. The software never rescales a trajectory to a desired number of turns.

## Angular velocity

Raw derivatives use `np.gradient(unwrapped_angle, actual_times)`. The smoothed derivative uses a small local polynomial fitted directly to actual times. This is analogous to Savitzky–Golay differentiation without falsely assuming uniform frame spacing.

Defaults: **0.150-second full window, at most five nearest samples, polynomial order 2**. Samples must lie within half the window of the evaluation time and within the same reliable orientation segment. Boundary fits use available one-sided samples; fewer than order+1 samples produce no smoothed derivative rather than a fabricated value.

Both raw and smoothed angular velocities are plotted and exported. Peak speed is the **signed value at maximum absolute smoothed speed over valid flight samples**, with its smoothing parameters shown in the summary and sidecar. It is not an estimate of a hidden peak inside a tracking gap. Average flight speed is total continuous rotation / airtime and is withheld when total rotation is unavailable.

## Tuck metric and events

For each reliable side, compute shoulder-to-knee distance / shoulder-to-hip length. Combine near/far sides using their minimum shoulder/hip/knee confidence as weight. Hip angle (shoulder–hip–knee, degrees) is a secondary exported series. No fixed limb proportions are enforced.

Initial event suggestions are deliberately simple: compare the early-flight median tuck ratio with the flight minimum. If the dip is substantial, maximum tuck is the minimum; tuck start needs two adjacent samples below a lower threshold, and opening needs two above a higher threshold after the minimum. The default hysteresis fractions are 0.5 and 0.7 of the dip depth. A tracking error can fool this heuristic; inspect and manually adjust or unset events. There is no automatic takeoff/landing detector in this version.

## Exports

The CSV includes source timestamp, frame, vertical position/displacement/fit/residual/fit weight, torso and hip quality, raw and segment-local unwrapped orientation, cumulative rotation, raw/smoothed angular velocity, tuck ratio, hip angle, optional athlete-calibrated displacement, quality flags, and event flags.

The matching JSON sidecar contains the complete versioned result, including units, event times/provenance, calibration/source, viewpoint/facing, parameters, smoothing, warnings, summary, and the CSV SHA-256. CSV event flags are placed on the nearest sampled frame; the JSON time is authoritative. Missing numerical values are empty CSV cells / JSON null, never zeros. Both files are staged before replacement, with rollback on a normal replacement failure. This is not a guarantee against process termination or power failure between two filesystem replacements.

## Recovery and benchmarks

The pre-cleanup commit is `227b9bb`, tagged **rotation-experiment-v1**. To inspect/recover it without altering the current checkout:

```bash
git show rotation-experiment-v1:ROTATION_EXPERIMENT.md
git archive --format=zip --output=/tmp/rotation-experiment-v1.zip rotation-experiment-v1
```

The former comparison UI, launcher, experiment runner, report code, and experiment tests are removed from the current tree. General rotation/coordinate geometry remains in `geometry.py`, with tests, and is reused for video display rotation.

`benchmarks/cork/` retains selected original frames, source timestamps/hash, the known reported failure, and the recovery tag. No exported human labels or candidate JSON results were found in the supplied workspace, so these entries explicitly remain review-pending; no labels or results were fabricated. Add those original results here if they become available. Automatic Cork/TDR recovery is not part of Phase 2.

## Real footage status

`spidey_back.MOV` and `dub_red_thing.MOV` were decoded and visually inspected. They are useful initial clean-motion targets. Approximate visual review windows are 1.0–1.8 seconds for `spidey_back`, and 1.0–2.0 seconds for `dub_red_thing`; these are **review windows, not accepted takeoff/landing annotations**. Select exact contact events with frame stepping.

The double-back clip uses a trampoline and a different landing surface; equal-COM-height airtime assumptions need particular scrutiny. Camera movement and oblique projection can also bias pixel-derived height. The application reports assumptions/disagreement instead of forcing estimates to agree.

Actual MediaPipe-derived kinematics cannot yet be verified inside the restricted build environment: native initialization has failed with `kGpuService / NSOpenGLPixelFormat`. Phase 1 has been confirmed working by the user in their normal desktop session. Run the existing app there, set events, and inspect the real curves before trusting quantitative values. Synthetic tests verify the mathematics and data/UI plumbing; they do not validate real-video accuracy.
