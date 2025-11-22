# video_io.py
# Handles video input, output, playback navigation, and safe reads.

import cv2


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def set_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def close(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, output_path, fps, resolution=(640,480)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        self.output_path = output_path

    def write(self, frame):
        
        self.writer.write(frame)

    def close(self):
        self.writer.release()


# Optional: Simple keyboard handler

def handle_keyboard(idx, start_idx, fps):
    key = cv2.waitKey(1)

    if key == ord('q'):
        return None  # Signal quit

    if key == 32:  # Space
        return idx  # Pause

    if key == ord('d'):
        return idx + 1  # Forward frame

    if key == ord('a'):
        return max(start_idx, idx - 1)  # Back frame

    if key == ord('w'):
        return idx + fps  # Forward 1 sec

    if key == ord('s'):
        return max(start_idx, idx - fps)  # Back 1 sec

    return idx + 1  # Default: next frame
