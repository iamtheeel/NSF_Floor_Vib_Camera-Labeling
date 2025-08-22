from ultralytics import YOLO
import cv2
import os
from library.camera_calibrator import CameraCalibrator  # Import the calibrator

class RoboflowRunner:
    def __init__(self, researcher):
        self.researcher = researcher
        self.settings = researcher.settings

        # Init calibrator
        self.calibrator = CameraCalibrator(researcher)
        self.calibrator.load_calibration()

    def run_YOLO(self, start_time=0, end_time=None, output_filename="YOLO_output.avi"):
        model = YOLO('yolov8n.pt')
        vid = self.researcher.load_video()

        if not vid.isOpened():
            print("Error: Cannot open video.")
            return

        # Get original video metadata
        meta = self.researcher.video_metadata
        fps = meta.get("fps", 30)
        total_frames = meta.get("total_frames", int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames

        # Seek to starting frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        paused = False

        while vid.isOpened() and frame_id < end_frame:
            if not paused:
                ret, frame = vid.read()
                if not ret:
                    break

                # NOTE: Do NOT crop or resize the frame
                results = model.track(frame, persist=True, classes=[0])
                annotated_frame = results[0].plot()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Bottom center of the bounding box
                    x_center = (x1 + x2) / 2
                    y_bottom = y2

                    # These are now full-frame coordinates
                    x_m, y_m = self.calibrator.pixel_to_meters(x_center, y_bottom)
                    track_id = int(box.id.item()) if box.id is not None else -1

                    text = f"ID:{track_id} ({x_m:.2f}m, {y_m:.2f}m)"
                    cv2.putText(annotated_frame, text, (int(x_center), int(y_bottom) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    print(text)

                cv2.imshow("YOLOv8 Person Tracking", annotated_frame)
                frame_id += 1

            # Controls
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord('p'):
                paused = not paused
            elif key == ord('a') and frame_id > 1:
                frame_id = max(start_frame, frame_id - 2)
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            elif key == ord('d') and frame_id < end_frame:
                continue
            elif key == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()
