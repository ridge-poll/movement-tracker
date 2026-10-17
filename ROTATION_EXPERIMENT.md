# Phase 2 experiment: alternate image orientations

This experiment asks one question: **can rotating the input yield a correct anatomical interpretation when the normal input yields a wrong one?** It does not pick a winning pose, correct a trajectory, interpolate gaps, smooth landmarks, or implement temporal scoring.

## Run on the cork clip

From this project folder, with the same environment used for Phase 1:

```bash
python -m movement_tracker.compare_rotations "/Users/ridgepoll/Documents/Codex/2026-10-17/videos/IMG_3877.MOV" \
  --start-frame 18 --end-frame 48
```

The prepared `compare-cork.command` launcher uses this workspace's existing Python environment (or a project `.venv` if available).

The clip is portrait 1080×1920. Its reported average rate is approximately 29.297 FPS; source PTS is retained rather than assuming 30 FPS. It decoded to 215 frames in the build environment, despite a 221-frame container hint. Visual inspection places the entry, airborne rotation, and landing around 0.6–1.6 seconds, so **frames 18–48 inclusive** are the initial benchmark window. These observations identify a test range; they do not establish which frames MediaPipe mislabels.

Alternatively, load a video in the existing app, step to an interesting frame, and click **Compare rotations…**. The new window defaults to that frame ±15, within the decoded clip bounds. Adjust the inclusive range and click **Run four orientations**. The existing processed clip and its raw poses are not modified.

## Compare and review

- The four panels show input rotations 0°, 90°, 180°, and 270°, all counterclockwise. Every overlay is mapped back onto the **same unrotated source frame**. Rotation refers to the experiment after any MOV display-orientation metadata has already been applied.
- Step with Left/Right, the frame buttons, or the slider. All four panels stay on the same decoded frame and timestamp.
- Green/amber still encode the existing confidence convention. A confidently wrong skeleton can remain green.
- Toggle skeleton, landmark points, and labels. The **Landmark measurements** tab shows source coordinates alongside untouched rotated-input coordinates, visibility, presence, and confidence for all 33 landmarks.
- Give each panel a manual label: unreviewed, correct anatomy, wrong anatomy, no pose, or ambiguous. Mark correctness for the **main athlete**; a pose on a background person is not a successful recovery. Use ambiguous when occlusion prevents a confident visual judgment.
- Optionally mark the frame category (correct tracking, inversion, occlusion, first failure, major anatomical misidentification, recovery) and enter notes. These are human annotations, not automated conclusions.
- **Export measurements + review JSON** saves every selected frame's four candidates and separate manual labels. **Save frame comparison PNG** saves the current four-panel comparison. Labels are held in memory until exported; closing with unsaved review prompts to save.

The review summary counts a manually identified recovery only when 0° is labeled wrong/no-pose and at least one alternate orientation is labeled correct. It also records frames where 0° is correct but an alternate is wrong/no-pose. Unreviewed candidates never count as recoveries. This is a small qualitative benchmark, not an accuracy estimate or automatic selection rule.

## Two inference modes

**VIDEO (default)** uses four distinct MediaPipe tracker instances. Each sees only its own orientation, in chronological order from **frame 0 through the selected end frame**. This retains the lead-in that can produce a tracking failure and avoids contaminating one orientation's history with another. Only the selected range is retained for inspection. For frames 18–48, this is 49 × 4 = 196 inference calls.

**IMAGE** is an optional control for isolating per-frame detector behavior. It runs independent IMAGE-mode inference on only the selected range, with no temporal history. For the same range, it makes 31 × 4 = 124 calls. Select it in the window or pass `--mode image`.

Do not compare IMAGE results directly against an old VIDEO result and attribute all changes to rotation. Compare the four orientations within the same mode first. The default 0° VIDEO pass is a fresh baseline rerun with the Phase 1 settings and full lead-in; it is not the exact previously cached Phase 1 object. Small nondeterministic differences may occur. The original Phase 1 results remain available in the original window/export.

## Coordinate mapping and raw evidence

Each candidate retains:

- `raw_rotated_input_pose`: the selected MediaPipe pose in the actual rotated model-input image, with all 33 x/y/z, visibility, presence, and derived confidence values untouched.
- `source_coordinates_pose`: a distinct `PoseFrame` mapped into the original displayed source image.
- Rotation angle, frame index, source and relative timestamps, original/rotated dimensions, inference duration, measurement provenance, and `interpolated=false`.

For a rotated-input point `(u, v)`, source-normalized `(x, y)` is:

| Input CCW rotation | Inverse mapping |
|---|---|
| 0° | `(u, v)` |
| 90° | `(1-v, u)` |
| 180° | `(1-u, 1-v)` |
| 270° | `(v, 1-u)` |

Coordinates outside [0, 1] are retained. Left/right anatomical names are never swapped by an image rotation. Visibility and presence are unchanged. Since MediaPipe's relative `z` is approximately normalized to image width, the mapped value is `raw_z × rotated_width / source_width`; raw `z` is also retained exactly. This is a change of scale, not calibrated 3D reconstruction, and does not make independently predicted depth values ground truth.

Confidence is still `min(visibility, presence)` when both are available. It is not a new plausibility score. Raw candidate `anatomy_correctness` remains `unreviewed`; manual judgments live separately under `human_reviews` so original measurements cannot be mistaken for corrected data.

As in Phase 1, up to three people are detected and the largest visible landmark bounding box is selected separately for each orientation/frame. The experiment retains all landmarks of that selected pose, not all background-person hypotheses or the separate world-landmark array. Subject selection can differ between orientations; inspect the pixels when judging a recovery.

JSON metadata includes the model path and SHA-256, MediaPipe version, inference mode, history start, source size/mtime, range, angle convention, and person-selection rule. Reports are portable JSON, not Python serialization. A no-detection candidate remains explicitly empty; there is no filled-in pose.

For scripted export:

```bash
python -m movement_tracker.compare_rotations video.mov \
  --start-frame 18 --end-frame 48 --headless --output rotations.json
```

Headless mode still requires whatever native graphics services the installed MediaPipe macOS build uses. It only omits the Qt window.

## Implementation and validation

- `experiments/rotations.py`: inverse geometry, candidate data, independent streams, cancellation, and temporary preview cache.
- `experiments/report.py`: portable measurements, manual review summary, and contact-sheet rendering.
- `ui/rotation_window.py`: four-panel frame comparison and confidence inspection.
- `compare_rotations.py`: standalone launch and headless JSON export.
- `tracking/mediapipe_tracker.py`: optional IMAGE mode; the default Phase 1 VIDEO behavior is unchanged.

Automated tests cover non-square inverse transforms, z scaling, anatomical identity, confidence preservation, missing/out-of-bounds landmarks, pixel rotation, independent temporal histories, exact frame ranges, raw JSON preservation, cancellation/cleanup, manual-only conclusions, and synchronized GUI review. Existing Phase 1 tests also run.

Validation result: **33 tests passed** (16 existing Phase 1 tests plus 17 experiment cases). The four-panel GUI was rendered and visually inspected using explicitly synthetic landmark data on decoded cork frames; that layout check is not an actual inference result. The standalone command's argument/help path was also checked.

**Real cork inference remains unverified in the build environment.** The provided video can be decoded, but MediaPipe 0.10.32 fails during initialization with `kGpuService / Could not create an NSOpenGLPixelFormat`, before generating any candidates. The user has confirmed that Phase 1 works in their normal desktop session. Run this experiment there to obtain the four actual candidate skeletons. No claim that input rotation fixes the cork failure is made without those results.

Once results are available, review a few frames covering correct tracking, first failure, inverted anatomy error, and recovery. If rotated candidates recover anatomy on failed frames while sometimes failing on otherwise good frames, that supports exploring candidate scoring/path selection next. If all orientations share the failure, it argues for trying another candidate generator before building a more complex selector.
