# Face Detection & Recognition System

## Overview

This project implements a complete Face Detection and Recognition system using OpenCV and the face_recognition library. The system can detect faces in images, videos, and live webcam streams, identify known individuals, and allow registration of new people.

## Features

* Face detection using HOG/CNN models
* Face recognition using 128-d embeddings
* Real-time recognition via webcam
* Supports image and video input
* Multiple face detection and labeling
* Register new people using webcam or image
* Confidence score display
* Simple and modular code structure

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install opencv-python face-recognition numpy
```

## How to Run

### 1. Register a New Person

Using webcam:

```
python register_face.py webcam "YourName" -n 5
```

Using image:

```
python register_face.py image "YourName" path/to/image.jpg
```

### 2. List Registered People

```
python register_face.py list
```

### 3. Run Face Recognition

Using webcam:

```
python recognize_faces.py
```

Using image:

```
python recognize_faces.py image.jpg -o output.jpg
```

Using video:

```
python recognize_faces.py video.mp4 -o output.mp4
```

### 4. Optional Settings

Stricter matching:

```
python recognize_faces.py -t 0.5
```

Use CNN model (more accurate, slower):

```
python recognize_faces.py --model cnn
```

## How It Works

1. Face Detection: Detects faces using HOG or CNN model
2. Face Encoding: Converts faces into 128-d feature vectors
3. Matching: Compares with stored encodings using distance metric
4. Output: Displays name and confidence with bounding boxes

## Output

* Bounding boxes around detected faces
* Name label for recognized faces
* "Unknown" for unregistered faces

## Notes

* Ensure good lighting while capturing faces
* Capture multiple samples for better accuracy
* CNN model requires more processing power
