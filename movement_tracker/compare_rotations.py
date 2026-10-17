"""Run the rotation experiment independently of the Phase 1 GUI."""
import argparse
from pathlib import Path
from threading import Event
from movement_tracker.models import DEFAULT_MODEL

def main():
    parser = argparse.ArgumentParser(description="Compare 0/90/180/270-degree pose hypotheses")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, required=True, help="inclusive, zero-based")
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, help="JSON export for headless mode")
    args = parser.parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        parser.error("Expected 0 <= start-frame <= end-frame")
    if args.headless:
        if args.output is None:
            parser.error("--headless requires --output result.json")
        from movement_tracker.experiments.rotations import run_rotations
        from movement_tracker.experiments.report import export_report
        result = None
        try:
            result = run_rotations(args.video, args.model, args.start_frame, args.end_frame, Event(), mode=args.mode)
            export_report(result, {}, args.output)
            print(f"Saved {len(result.frames)} frames × 4 orientations to {args.output}")
        except Exception as error:
            parser.exit(1, f"Rotation experiment failed: {error}\n")
        finally:
            if result is not None:
                result.close()
    else:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from movement_tracker.ui.rotation_window import RotationWindow
        app = QApplication([])
        window = RotationWindow(args.video, args.model, args.start_frame, args.end_frame, args.mode)
        window.show()
        QTimer.singleShot(0, window.run)
        raise SystemExit(app.exec())

if __name__ == "__main__":
    main()
