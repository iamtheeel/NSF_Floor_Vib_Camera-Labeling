# Hallway Pose Tracking & Vibration Synchronization
A multimodal processing pipeline for synchronizing hallway video, human pose tracking, camera calibration, and vibration sensor data (STARS Study – MIC Lab).

---

## Setup

### 1. Clone the repository

```
git clone https://github.com/iamtheeel/NSF_Floor_Vib_Camera-Labeling.git
cd NSF_Floor_Vib_Camera-Labeling
```

---

### 2. Required Downloads

#### **A. Download MediaPipe Pose Model**
Google MediaPipe Pose Landmarker models:  
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Recommended file:
- `pose_landmarker_heavy.task`

Place it in:

```
Models/pose_landmarker_heavy.task
```

#### **B. Install Tesseract OCR**
Tesseract is required for reading timestamps directly from video frames.

Download (Windows – UB Mannheim build):  
https://github.com/UB-Mannheim/tesseract/wiki

After installation, ensure Tesseract is on your PATH:
```
tesseract --version
```

---

### 3. Create a Python Virtual Environment

Open a terminal **inside the repository directory**:

```
python -m venv venv
```

Activate it:

**Mac/Linux**
```
source venv/bin/activate
```

**Windows**
```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Directory Setup

Your folder structure should look like this:

```
Models/
    pose_landmarker_heavy.task

StudentData/
    video_hallwayTests/
    calibration_images/
    csv_files/
        all_checkerboard_points.csv
    vibration_test_data/
```

---
