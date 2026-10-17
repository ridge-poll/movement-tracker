"""Small thread job with coalesced progress, not a queue of UI events."""
from threading import Event, Lock, Thread
from movement_tracker.video.processing import Cancelled

class Job:
    def __init__(self, function):
        self.cancel = Event()
        self.done = Event()
        self.lock = Lock()
        self.progress = (0, 0)
        self.result = None
        self.error = None
        self.cancelled = False
        self.latest = None
        self.thread = Thread(target=self._run, args=(function,), daemon=True)
        self.thread.start()

    def _run(self, function):
        try:
            self.result = function(self)
        except Cancelled:
            self.cancelled = True
        except Exception as error:
            self.error = str(error)
        finally:
            self.done.set()

    def update(self, count, total):
        with self.lock:
            self.progress = (count, total)

    def publish(self, item):
        with self.lock:
            self.latest = item

    def take_latest(self):
        with self.lock:
            item, self.latest = self.latest, None
            return item
