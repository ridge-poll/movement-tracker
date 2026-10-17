"""Portable measurements and human review; no inferred correctness labels."""
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import os
import cv2
import numpy as np
from movement_tracker.visualization.skeleton_renderer import render, OverlayOptions

RATINGS = ("unreviewed", "correct_anatomy", "wrong_anatomy", "no_pose", "ambiguous")
TAGS = ("unlabeled", "correct_tracking", "inversion", "occlusion", "first_failure",
        "anatomical_misidentification", "recovery")

def review_summary(frames, reviews):
    recovered, regressed = [], []
    reviewed = 0
    for frame in frames:
        ratings = reviews.get(str(frame.frame_index), {}).get("ratings", {})
        baseline = ratings.get("0", "unreviewed")
        alternatives = [ratings.get(str(angle), "unreviewed") for angle in (90, 180, 270)]
        if baseline != "unreviewed" or any(r != "unreviewed" for r in alternatives):
            reviewed += 1
        if baseline in ("wrong_anatomy", "no_pose") and "correct_anatomy" in alternatives:
            recovered.append(frame.frame_index)
        if baseline == "correct_anatomy" and any(r in ("wrong_anatomy", "no_pose") for r in alternatives):
            regressed.append(frame.frame_index)
    return {"total_frames": len(frames), "frames_with_any_human_rating": reviewed,
            "human_reviewed_recovery_frames": recovered,
            "human_reviewed_alternate_regression_frames": regressed,
            "interpretation": "Manual frame labels only; not a temporal correction or accuracy score."}

def pose_dict(pose):
    data = asdict(pose)
    for name, point in pose.landmarks.items():
        data["landmarks"][name]["confidence"] = point.confidence
    data["pose_detected"] = bool(pose.landmarks)
    data["mean_confidence"] = pose.confidence
    return data

def report_dict(experiment, reviews):
    frames = []
    for frame in experiment.frames:
        candidates = []
        for candidate in frame.candidates:
            candidates.append({
                "angle_ccw": candidate.angle_ccw, "provenance": candidate.provenance,
                "interpolated": candidate.interpolated,
                "anatomy_correctness": candidate.anatomy_correctness,
                "raw_rotated_input_pose": pose_dict(candidate.input_pose),
                "source_coordinates_pose": pose_dict(candidate.source_pose),
            })
        frames.append({"frame_index": frame.frame_index, "timestamp": frame.timestamp,
                       "candidates": candidates})
    return {"metadata": experiment.metadata, "frames": frames, "human_reviews": reviews,
            "review_summary": review_summary(experiment.frames, reviews)}

def export_report(experiment, reviews, destination):
    destination = Path(destination)
    if destination.resolve() == experiment.source.resolve():
        raise ValueError("The source video cannot be overwritten.")
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                suffix=".json", delete=False) as file:
            temporary = Path(file.name)
            json.dump(report_dict(experiment, reviews), file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)

def comparison_image(rgb, frame, options=OverlayOptions(), reviews=None):
    """A contact sheet in source orientation, also usable without the GUI."""
    panels = []
    ratings = (reviews or {}).get(str(frame.frame_index), {}).get("ratings", {})
    for candidate in frame.candidates:
        pose = candidate.source_pose
        rendered = render(rgb, pose, options)
        h, w = rendered.shape[:2]
        header = np.full((88, w, 3), (24, 29, 36), np.uint8)
        title = f"Input {candidate.angle_ccw} deg CCW | frame {frame.frame_index}"
        score = f"{pose.confidence:.3f}" if pose.confidence is not None else "NO POSE"
        cv2.putText(header, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(header, f"t={frame.timestamp:.3f}s | confidence {score}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 210), 1, cv2.LINE_AA)
        rating = ratings.get(str(candidate.angle_ccw), "unreviewed")
        cv2.putText(header, f"Human: {rating}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (210, 210, 210), 1, cv2.LINE_AA)
        panels.append(np.vstack((header, rendered)))
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
