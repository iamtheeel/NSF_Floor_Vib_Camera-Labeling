import importlib
from library.researcher_base import Researcher
from library.robo_flow import RoboflowRunner

# load researcher settings found in settings directory
RESEARCHER = "duy"
researcher = Researcher(RESEARCHER)

roboflow_runner = RoboflowRunner(researcher)
roboflow_runner.run_YOLO(start_time=7, end_time=30, output_filename="roboflow_output.avi")






