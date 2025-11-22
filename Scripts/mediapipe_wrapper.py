
import mediapipe as mp 


from Scripts.cv2Utils import isPersonInFrame


class PoseDetector:
    def __init__(self, model_path, use_gpu=False, output_masks=True):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Choose delegate
        delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU

        # Build options
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=delegate
            ),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=output_masks
        )

        # Create persistent landmarker
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame, frame_idx, frame_time_ms):
        return isPersonInFrame(frame, frame_idx, frame_time_ms, self.landmarker)
