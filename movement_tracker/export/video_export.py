"""Re-decode source pixels and encode every frame with original relative timing."""
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import av
import cv2
from movement_tracker.video.video_reader import VideoReader
from movement_tracker.video.processing import Cancelled
from movement_tracker.visualization.skeleton_renderer import render

def export_video(clip, destination, options, cancel, progress=lambda *args: None):
    destination = Path(destination)
    if destination.resolve() == clip.source.resolve():
        raise ValueError("Choose an output file different from the source video.")
    stat = clip.source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (clip.source_size, clip.source_mtime_ns):
        raise ValueError("The source video changed. Reopen and analyze it before export.")
    with NamedTemporaryFile(dir=destination.parent, suffix=".mp4", delete=False) as file:
        temporary = Path(file.name)
    try:
        with VideoReader(clip.source) as reader, av.open(str(temporary), "w", format="mp4") as output:
            rate = Fraction(clip.fps or 30).limit_denominator(100000)
            stream = output.add_stream("libx264", rate=rate)
            first = clip.frames[0]
            stream.width = first.width + first.width % 2
            stream.height = first.height + first.height % 2
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1_000_000)
            stream.codec_context.time_base = stream.time_base
            stream.codec_context.max_b_frames = 0
            stream.codec_context.thread_count = 2
            stream.options = {"crf": "18", "preset": "fast"}
            durations = {}
            for i, pose in enumerate(clip.frames):
                next_time = clip.frames[i + 1].timestamp if i + 1 < len(clip.frames) else pose.timestamp + clip.last_duration
                durations[round(pose.timestamp * 1_000_000)] = max(1, round((next_time - pose.timestamp) * 1_000_000))
            def mux(packets):
                for packet in packets:
                    if packet.pts in durations:
                        packet.duration = durations[packet.pts]
                    output.mux(packet)
            count = 0
            for decoded in reader:
                if cancel.is_set():
                    raise Cancelled()
                if decoded.index >= len(clip.frames):
                    raise ValueError("Source frame count changed.")
                pose = clip.frames[decoded.index]
                if decoded.rgb.shape[:2] != (pose.height, pose.width):
                    raise ValueError("Video dimensions changed within the clip.")
                rgb = render(decoded.rgb, pose, options)
                if (pose.width, pose.height) != (stream.width, stream.height):
                    rgb = cv2.copyMakeBorder(rgb, 0, stream.height - pose.height, 0,
                                             stream.width - pose.width, cv2.BORDER_CONSTANT)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = round(pose.timestamp * 1_000_000)
                frame.time_base = Fraction(1, 1_000_000)
                mux(stream.encode(frame))
                count += 1
                progress(count, len(clip.frames))
            if count != len(clip.frames):
                raise ValueError("Source frame count changed.")
            mux(stream.encode())
        if cancel.is_set():
            raise Cancelled()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
