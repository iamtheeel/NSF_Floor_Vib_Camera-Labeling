
import mediapipe as mp 


from Scripts.cv2Utils import isPersonInFrame


class PoseDetector:
    def __init__(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path,
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=True
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame, frame_idx, frame_time_ms):
        return isPersonInFrame(frame, frame_idx, frame_time_ms, self.landmarker)
