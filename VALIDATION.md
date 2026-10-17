# Build validation

Phase 2 adds a separate rotation experiment: **33 combined tests pass**. See `ROTATION_EXPERIMENT.md` for its validation and the unresolved real-model runtime limitation. The Phase 1 results below describe the original milestone.

## Automated checks

Result: **16 tests passed** on Python 3.11.15. `pip check` reported no broken requirements. Direct dependency versions are recorded in `requirements-tested.txt`.

The test suite covers:

- Full-clip processing with synthetic variable-frame-rate H.264 video; no missing frames.
- CSV columns for all 33 landmarks; blank values and an explicit flag for undetected frames.
- Overlay H.264 export decoded again to verify frame count, variable timestamps, final frame duration, and changed overlay pixels.
- Pixel orientation for +90°, −90°, and 180° display rotations.
- All frames retained from a 240 FPS source.
- Cancelled export preserves an existing destination and removes its partial file.
- Refusal to overwrite or export against a changed source.
- Rendering does not modify source pixels or raw pose data.
- Processing cancellation.
- GUI frame stepping, bounds, scrubbing, play/pause, and no-pose status in Qt's offscreen mode.
- Backend adapter selection of the largest pose, all landmarks, precise stored timestamps despite backend millisecond quantization, and missing detections.
- Coalesced worker results and surfaced errors.

The UI was rendered offscreen and visually inspected at 1100×850. This checks layout, not interactive native camera access.

## Real-model check: blocked in this environment

The official Full model downloaded successfully. A real MediaPipe inference attempt using Google's public pose sample could not initialize the macOS graphics context in this restricted execution environment:

- MediaPipe 0.10.35 aborted natively in `DrishtiMetalHelper` with `Service is unavailable`, even with the CPU delegate explicitly selected.
- MediaPipe 0.10.32 returned a catchable `kGpuService / Could not create an NSOpenGLPixelFormat` error. The deliverable pins this version so that this initialization failure can be shown by the GUI instead of aborting it.

No claim of successful real pose inference, 30 FPS throughput, live-camera operation, or backflip tracking quality is made by these tests. Run the app in a normal logged-in macOS desktop terminal to perform those checks. The Tasks API integration is implemented, but this environment cannot establish whether it initializes successfully in your desktop session.

## Manual movement evaluation

Start with one athlete fully visible, a fixed camera, and the source files preserved. Use actual source metadata; do not convert to 30 FPS or otherwise drop frames just for this evaluation.

| Test | Inspect |
|---|---|
| Standing, walking, arms raised, squats | Baseline detection, stability, left/right correctness |
| Jumps, spins, kicks | Blur, fast movement, limb swaps |
| Handstands and cartwheels | Inversion, overlap, reacquisition |
| Backflips (primary test) | Takeoff through landing: inversion, tuck, blur, missing pose, incorrect confident landmarks |
| Partial body leaving the frame | Missing joints, stale-looking estimates, recovery |
| Background person | Prominence selection switches |

For each clip, note the source filename, failing frame ranges, failure type, confidence behavior, and whether tracking recovers. These are manual notes outside the application; no annotation system is implemented. Compare the overlay with the actual pixels rather than treating confidence as ground truth.

Also verify on the target Mac:

1. Grant camera access and select the built-in camera index.
2. Confirm live latency does not accumulate during rapid movement; observe inference FPS.
3. Stop camera, open a real portrait MOV, wait for processing, and inspect orientation.
4. Scrub back and forth and step around a failed backflip frame; confirm exact overlay alignment.
5. Export CSV and silent overlay MP4. Open the MP4 in QuickTime and check orientation, duration, overlay, and smooth playback.
6. Close the window during processing/export and confirm it exits after the current operation stops.

Stop after this tracking-quality evaluation. Coaching, derived kinematics, scoring, filtering, and trick recognition remain out of scope.
