"""Capture and inference have one-slot mailboxes: stale frames are discarded."""
from threading import Condition, Event, Thread
from time import perf_counter
import sys
import cv2

class CameraCapture:
    def __init__(self, index):
        self.index = index
        self.condition = Condition()
        self.stop_event = Event()
        self.latest = None
        self.error = None
        self.done = False
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        capture = None
        try:
            api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.index, api)
            if not capture.isOpened():
                raise RuntimeError("Camera could not open. Check the camera index and macOS Camera permission for your terminal/app.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            origin = perf_counter()
            index = 0
            while not self.stop_event.is_set():
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped delivering frames.")
                timestamp = perf_counter() - origin
                with self.condition:
                    self.latest = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp, index)
                    self.condition.notify()
                index += 1
        except Exception as error:
            self.error = error
        finally:
            if capture is not None:
                capture.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def get(self):
        with self.condition:
            if self.latest is None and not self.done:
                self.condition.wait(timeout=0.1)
            item, self.latest = self.latest, None
            if self.error:
                raise self.error
            return item

    def close(self):
        self.stop_event.set()
        self.thread.join()

def run_camera(job, camera_index, tracker_factory):
    tracker = tracker_factory()
    camera = CameraCapture(camera_index)
    try:
        while not job.cancel.is_set():
            item = camera.get()
            if item is None:
                continue
            rgb, timestamp, index = item
            pose = tracker.process_frame(rgb, timestamp, index)
            job.publish((rgb, pose))
    finally:
        camera.close()
        tracker.close()
