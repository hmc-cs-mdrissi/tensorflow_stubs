from tensorflow.python.training.tracking.base import Trackable

class AutoTrackable(Trackable):
    def _delete_tracking(self, name: str) -> None: ...
