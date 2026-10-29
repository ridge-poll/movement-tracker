# Movement Tracker — Phase 2

A Python desktop application for tracking movement and measuring clean, approximately side-view front/back flips. Phase 2 adds inspectable hip-proxy trajectories, rotation, angular velocity, tuck metrics, event editing, and measurement-quality flags. It does not provide coaching or repair poses.

**New: side-view analysis.** Process a clip, click **Analyze flip…**, choose facing direction, and set takeoff/landing from the video. Synchronized plots and editable event suggestions appear after analysis. See [PHASE2.md](PHASE2.md) for definitions, calibration, exports, and limitations. The earlier four-orientation experiment is preserved in Git tag `rotation-experiment-v1`; its dedicated UI/scripts were removed from the current version.

## Install and run

Use native Apple Silicon **Python 3.11 or 3.12**, rather than macOS's system Python. Validation dependency versions are in `requirements-tested.txt`. MediaPipe is pinned to 0.10.32: 0.10.35 aborted during initialization in the restricted test environment; 0.10.32 reports a catchable graphics-context error there. Both use the supported Tasks API. Other requirements allow compatible updates.

```bash
cd /path/to/movement-tracker
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m movement_tracker.models
python -m movement_tracker
```

The download command retrieves Google's official Full Pose Landmarker model into `models/`. A downloaded Full model is included in this workspace deliverable. All video processing runs locally; no footage is uploaded. The app does not need network access after installation and model download.

To try another model:

```bash
python -m movement_tracker.models --variant lite
python -m movement_tracker --model models/pose_landmarker_lite.task
```

`heavy` is also available for offline experiments. Full is the initial default for both recorded and live input. Lite may improve camera throughput. No discrete GPU is required.

## Use

1. **Open Video**: choose an MP4, MOV, AVI, M4V, or MKV supported by FFmpeg/PyAV. Actual file metadata determines dimensions and timing. Wait for the full clip to finish processing; progress and cancellation are available.
2. **Play / Pause**, drag the scrubber, or use **< Frame / Frame >**. Keyboard shortcuts: Space, Left, Right. Frames are numbered from zero. Playback speeds include 0.25×, 0.5×, 1×, and 2×.
3. Toggle skeleton, points, and labels. The right panel shows per-landmark visibility and presence. Amber denotes confidence below 0.5; green denotes confidence at least 0.5. Low-confidence points remain visible for inspection. Off-image landmarks remain in the data but are not drawn.
4. **Export CSV** includes every analyzed frame and all 33 landmarks, including blank values for frames without a pose.
5. **Export Overlay MP4** writes a full-resolution H.264 video using the current overlay toggles. It retains every analyzed frame and relative presentation timing. **Exports are silent; audio is not copied.** Original videos are protected from overwrite. Odd dimensions receive at most one black padding pixel on the bottom/right for H.264 compatibility.
6. **Live Camera** opens the chosen numeric camera index (usually 0 for the built-in camera). If another device appears, stop and try another index. macOS may request Camera permission for the terminal/application launching Python. Stop returns to the previously processed video. Live capture has no recording or export.
7. **Analyze flip…** opens a measurement panel for the processed video. Set takeoff/landing explicitly, choose facing direction, then **Analyze / Recalculate**. Click plots to seek the video. Use the events/settings tab to adjust suggested events and smoothing. Analysis CSV + JSON export is separate from the unchanged raw landmark CSV export.

Inference FPS measures the model call for the displayed frame, not camera FPS or end-to-end application throughput. Source FPS is the stream's reported average. The preview is unmirrored, so landmark left/right refers to the person's anatomical side.

## Data contract

`PoseFrame` is a plain dataclass independent of MediaPipe and the GUI. It contains relative timestamp (seconds), zero-based frame index, oriented source width/height, named landmarks, original presentation timestamp, timestamp-estimation flag, and inference duration.

CSV columns begin with:

```text
timestamp,frame,width,height,source_timestamp,timestamp_estimated,pose_detected
```

Each of the 33 names then has `x`, `y`, `z`, `confidence`, `visibility`, and `presence` columns. Missing values are empty, never fabricated zeros.

- `x` and `y` are normalized to the oriented image, with origin at its top-left. Predictions can extend outside [0, 1].
- `z` is MediaPipe's relative landmark depth (hip midpoint origin, approximately x-coordinate scale), **not calibrated metric depth**.
- `visibility` and `presence` retain the backend values separately. `confidence = min(visibility, presence)` when both exist; this is a display convention, not a calibrated accuracy measure. The UI's mean confidence averages that value over detected landmarks.
- `source_timestamp` preserves decoded PTS in seconds. `timestamp` subtracts the first presentation time for playback. If PTS is missing or non-increasing, the reported frame rate supplies a fallback and `timestamp_estimated=1` identifies it. Files without usable timestamps and frame rate are rejected.
- MediaPipe requires increasing integer milliseconds. Its internal clock may be nudged forward; stored source timestamps are not rounded to that clock.
- All 33 normalized landmarks are retained. MediaPipe's separate world-coordinate prediction array is not exported in this milestone.

The tracker considers up to three detected people and selects the largest visible landmark bounding box each frame. This is a simple prominence heuristic, not persistent identity tracking. Background people can cause selection switches, especially when the athlete tucks or leaves view.

## Performance and storage

Inference runs on every decoded source frame at its original resolution; the model performs its own required input preprocessing. High-frame-rate files are not temporally downsampled. Sequential processing runs on a background thread.

Previews use temporary JPEG files with longest edge at most 960 pixels. This bounds working memory and makes random frame access independent of long-GOP video seeking. Only preview pixels are resized/compressed. The cache is removed when a clip is replaced or the app exits normally. A forced process termination may leave `movement-tracker-*` temporary directories in the OS temp directory; those can be removed when the app is not running. Disk requirements grow with frame count.

Playback follows timestamps and can skip *displaying* frames when the screen/UI cannot keep up (for example 240 FPS on a 60 Hz display). Every frame remains available through stepping, CSV, and video export. The application does not retain all full-resolution decoded frames in RAM.

Camera capture and inference use single-slot mailboxes. Capture continually replaces stale frames; the worker processes the newest available frame, and the UI displays the exact image associated with that pose. No growing frame queue accumulates latency. Camera mode requests 1280×720 at 30 FPS, subject to device support; approximately 30 inference FPS is a target, not a guarantee.

Exports re-decode the source and use saved poses without running inference again. They write to a temporary file and replace the destination only after completion. Keep the source file available and unchanged until export. Rotation in multiples of 90° is applied before tracking and export. Unusual arbitrary-angle display rotations are rejected. HDR tone mapping, mirrored display matrices, anamorphic pixel-aspect correction, and audio playback/export are not implemented.

## Architecture

```text
video/video_reader.py → tracking/PoseTracker → pose_data.PoseFrame
                                ↓                    ↓
video/processing.py → temporary preview cache   visualization / export
                                ↓                    ↓
                              ui/main_window.py
```

- `tracking/pose_tracker.py`: backend protocol; receives RGB pixels, timestamp, and index.
- `tracking/mediapipe_tracker.py`: current Tasks API in VIDEO mode, used sequentially in a worker for both file and camera input. No deprecated `mp.solutions` API.
- `pose_data.py`: portable named data and fixed skeleton topology; no GUI/backend imports.
- `video/video_reader.py`: PyAV decoding, PTS handling, and portrait rotation.
- `video/processing.py`: all-frame analysis and disk preview cache.
- `video/camera.py`: independent capture and inference with bounded buffers.
- `visualization/skeleton_renderer.py`: independent overlay rendering; does not modify raw landmarks or pixels.
- `export/`: CSV and timestamp-aware MP4 export.
- `jobs.py`: cooperative background jobs and coalesced progress/results.
- `ui/main_window.py`: playback, inspection, and job orchestration.
- `analysis/`: versioned result schema, quality flags, time-aware derivatives, ballistic fitting, and event suggestions; consumes only `PoseFrame` data and NumPy.
- `ui/analysis_panel.py`, `ui/analysis_plot.py`: synchronized native Qt plots and manual event controls.
- `export/analysis_export.py`: analysis time series and JSON metadata/units/calibration sidecar.
- `geometry.py`: reusable image rotation and inverse landmark-coordinate mapping; no multi-orientation recovery.

A future iPhone implementation can reuse the data contract and processing concepts, with a native pose adapter and GUI. This is not a promise that Python/PySide code will run unchanged on iOS.

## Verify

```bash
python -m pip install pytest
python -m pytest -q
```

Tests generate short videos and check variable frame timing, portrait rotation, 240 FPS frame retention, CSV missing detections, synchronized export, cancellation, source protection, and GUI navigation. They use a deterministic fake pose backend so media/GUI failures are separable from model quality. See `VALIDATION.md` for the actual checks performed on this build and the manual movement checklist.

Additional synthetic skeleton tests verify known rotation signs and magnitudes, irregular timestamp handling, 900°/s motion at 30 FPS, angular acceleration, noise filtering, parabolic flight/calibration, event provenance, and quality breaks. They do not establish real-footage measurement accuracy.

## Troubleshooting

- **Missing model:** run `python -m movement_tracker.models` or pass `--model PATH`.
- **NSOpenGLPixelFormat / kGpuService error:** the macOS MediaPipe wheel needs access to desktop graphics services even when CPU inference is selected. Run from a normal logged-in desktop terminal, not a restricted/headless agent environment. Real inference could not be verified inside the build sandbox; see `VALIDATION.md`.
- **Camera cannot open:** check System Settings → Privacy & Security → Camera; allow the launching app, close other camera clients, and check the camera index.
- **Slow live view:** try the Lite model; stop other heavy processes. File analysis intentionally favors completeness over real-time speed.
- **Video fails to decode:** try a standard H.264 MP4 copy; support depends on the codecs shipped with PyAV. Errors leave the previous processed clip available.
- **High confidence but wrong limbs:** confidence is not proof of correct tracking. Inspect inversion, blur, occlusion, and tucked frames manually before drawing conclusions.

## Upstream references

- [MediaPipe Tasks Pose Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [MediaPipe pose model and landmark documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [PyAV video frame and stream API](https://pyav.basswood.io/docs/stable/api/video.html)

Google's model remains an upstream artifact, rather than an original asset of this project. Review upstream terms before redistributing a packaged application.

## Manual pose correction

Pause on a problematic frame, click **Edit Pose**, and drag the incorrect landmark.
Purple points mark manual XY placements. Hover to identify a point; use the landmark
selector when left/right points overlap. Frame stepping and scrubbing retain edits.
**Undo** reverses the last change on this frame; **Reset Frame** removes its overrides
(and can itself be undone). Corrections never propagate to neighboring frames.

Click **Recalculate** to regenerate all kinematics, fits, automatic events, plots and
summary using the corrected sequence. Set takeoff, landing and facing direction first
in the analysis panel if you have not already. Manual event choices remain intact.
Previous analysis is marked stale after any edit, undo or reset and cannot be exported
until recalculated.

Raw `PoseFrame` objects remain unchanged. Only edited XY coordinates are overridden;
Z, visibility, presence, timestamps and untouched landmarks remain exactly as detected.
Manual positions use analysis weight 1, explicitly representing user placement rather
than model confidence. Confidence inspection continues to show original detector values.
Manual placement does not guarantee accurate 3D motion or eliminate side-view assumptions.

Analysis CSV quality flags identify each manually adjusted landmark on its frame.
The JSON sidecar records original and corrected normalized XY, source timestamps,
frame indices, correction revision and weighting policy. **Export Raw CSV** still
exports original detections. Overlay MP4 uses corrected poses, with purple edited
points when landmark points are enabled. Session corrections remain in memory until
another video replaces the clip or the app closes; session reopening is not implemented.

### Selecting and moving groups

In Edit Pose, click a point to select it; white rings show the current selection.
Shift-click adds or removes an individual point. Drag on empty image space to marquee
exactly the points inside the rectangle (Shift-marquee adds to the selection).
Click empty image space to clear selection. Drag any selected point to move the
whole selection by the same image-space displacement. At image edges, the entire
group stops together without changing its shape. Each group movement is one Undo step.

**Group hands / feet** is enabled by default: clicking a hand selects that side's
wrist, thumb, index and pinky; clicking a foot selects its ankle, heel and foot index.
Turn the checkbox off to start an individual selection, or use the exact landmark
selector to isolate an overlapping point. Shift-click and marquee always operate on
individual points, without expanding anatomical groups. Selection clears when changing
frames or leaving editing; corrections remain stored for the loaded video.

### Clearing and rebuilding a missing pose

**Clear Pose** suppresses the current frame's detection without deleting it.
**Place Rough Pose** replaces that frame with a guided sequence: head, near shoulder,
near hip, near knee, near ankle. Click the same visible side consistently. Back revisits
the previous point; Skip leaves an obscured point absent; Cancel Placement restores the
pre-placement state. Skipping all five points makes no change. Frame navigation is
paused during placement. Completing the sequence is one Undo step.

The placed purple points remain draggable, including when the original frame had no
pose. Reset Frame restores the original detection; Undo reverses clear, placement,
or dragging. Recalculate refreshes the analysis. Shoulder/hip support orientation,
shoulder/hip/knee support tuck, and hip supports the height proxy. Missing quantities
remain unavailable. Rough poses use a single near-side hip, not a two-hip midpoint;
transitions between these proxies can affect measured height and velocity. No hidden
side, depth, or model confidence is invented. CSV flags and JSON provenance record
cleared and rough-pose frames and all manually supplied coordinates.

### Guided second image estimate

After placing a rough pose, click **Estimate Pose** in the video editor. The anchors
guide a padded crop and shoulder-to-hip upright rotation of the current preview image;
MediaPipe runs again on that image and maps its landmarks back to video coordinates.
This is crop-guided inference, not a landmark-conditioned model: the returned points
are not constrained to pass through your clicks and may still be wrong. Shoulder and
hip are required; additional guides help define the crop. Review the result and use
Undo to recover the rough pose. Failed or cancelled inference leaves edits unchanged.
The second estimate retains model confidence, has no inferred source-image depth, and
is recorded with its guide anchors in export provenance. To refresh kinematics, use
**Analyze / Recalculate** in the plot panel.

Frame diagnostics sit below the plots in a fixed-height scrolling area. Hover for the
full diagnostic text. Missing mean confidence is displayed as N/A.
