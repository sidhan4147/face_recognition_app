# Face Recognition App

A real-time face recognition application built with Python, OpenCV, and face_recognition library. This application recognizes faces from a webcam feed and displays the name and confidence percentage for recognized individuals.

## Features

- Real-time face detection and recognition
- Display of recognition confidence percentage
- Ability to save frames from the video feed
- Trained on characters from Friends TV show by default

## Requirements

- Python 3.8+ (3.11 recommended)
- OpenCV
- face_recognition library
- numpy

## Installation

1. Clone the repository:
```
git clone https://github.com/sidhan4147/face_recognition_app.git
cd face_recognition_app
```

2. Create a virtual environment (recommended):
```
# On Windows
python -m venv face_recognition_env
face_recognition_env\Scripts\activate

# On macOS/Linux
python -m venv face_recognition_env
source face_recognition_env/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Before running, make sure you have some training images in the `train/` directory.
   - Each image should contain exactly one face
   - The filename (without extension) will be used as the person's name
   - Example: `train/john.jpg` will be recognized as "John"

2. Run the application:
```
python main.py
```

3. Controls:
   - Press 'q' to quit
   - Press 's' to save the current frame

## Adding New Faces

To add new faces for recognition:

1. Place clear images of faces in the `train/` directory
2. Name the files appropriately (e.g., `firstname.jpg`)
3. Restart the application to include the new faces

## Troubleshooting

- If the camera doesn't open, check if it's being used by another application
- If face recognition isn't working well, ensure good lighting and clear images
- Make sure the training images have exactly one face per image
