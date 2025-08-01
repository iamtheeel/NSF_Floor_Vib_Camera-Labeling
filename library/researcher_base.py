import importlib
import os
import cv2 as cv

class Researcher:
    def __init__(self, researcher: str):
        self.settings = importlib.import_module(f"settings.{researcher}_settings")
        self.video_capture = None
        self.video_metadata = {}

    def load_video(self):
        if self.video_capture is not None:
            return self.video_capture  # already loaded

        video_path = os.path.join(self.settings.FOLDER_DIR, self.settings.VIDEO_FILENAME)
        vid = cv.VideoCapture(video_path)

        if not vid.isOpened():
            print("Error: Could not open video.")
            return None

        self.video_capture = vid
        self.video_metadata = {
            "fps": vid.get(cv.CAP_PROP_FPS),
            "width": int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(vid.get(cv.CAP_PROP_FRAME_COUNT)),
            "path": video_path,
        }

        print(f"Video loaded: {video_path}")
        return vid
