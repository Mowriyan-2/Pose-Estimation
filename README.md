# MediaPipe Pose Estimation

A real-time human pose estimation project using MediaPipe and OpenCV that detects body landmarks, calculates joint angles, and counts bicep curl repetitions through a webcam feed.

## Features

- Real-time pose detection via webcam
- Body landmark detection and visualization
- Joint angle calculation (shoulder → elbow → wrist)
- Bicep curl rep counter with stage tracking (up/down)
- Live angle overlay on video feed

## Tech Stack

- Python 3.7+
- [MediaPipe](https://mediapipe.dev/) — Pose landmark detection
- [OpenCV](https://opencv.org/) — Video capture and rendering
- [NumPy](https://numpy.org/) — Angle calculations

## Installation

```bash
pip install mediapipe opencv-python numpy
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mowriyan/mediapipe-pose-estimation.git
   cd mediapipe-pose-estimation
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Media_pipe_pose_estimation.ipynb
   ```

3. Run the cells in order:
   - **Section 0** — Install and import dependencies
   - **Section 1** — Start webcam and make pose detections
   - **Section 2** — Explore and extract joint landmarks
   - **Section 3** — Calculate joint angles in real time
   - **Section 4** — Run the full bicep curl counter

4. Press **`q`** to exit the webcam feed at any time.

## How It Works

### Pose Detection
MediaPipe's Pose solution detects 33 body landmarks. Each frame is converted from BGR to RGB before processing, then converted back for display.

### Angle Calculation
The elbow angle is calculated using the arctangent of vectors formed by the shoulder, elbow, and wrist coordinates:

```python
def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
```

### Curl Counter Logic
- **Down** stage: elbow angle > 160°
- **Up** stage: elbow angle < 30° (after being in down stage)
- A rep is counted each time a full down → up transition completes.

## Project Structure

```
mediapipe-pose-estimation/
│
├── Media_pipe_pose_estimation.ipynb   # Main notebook
└── README.md
```

## Author

**Mowriyan**  
📧 [mowriyan52@gmail.com](mailto:mowriyan52@gmail.com)

## License

This project is open source and available under the [MIT License](LICENSE).
