"""Face detection and recognition on images, video files, or webcam."""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np

from utils import load_encodings


class FaceRecognizer:
    """Handles face detection and recognition."""
    
    def __init__(self, tolerance: float = 0.6, model: str = "hog"):
        """
        Initialize the recognizer.
        
        Args:
            tolerance: Distance threshold for face matching (lower = stricter)
            model: Detection model - 'hog' (faster) or 'cnn' (more accurate)
        """
        self.tolerance = tolerance
        self.model = model
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self._load_known_faces()
    
    def _load_known_faces(self) -> None:
        """Load known face encodings from database."""
        encodings_dict = load_encodings()
        
        for name, encodings in encodings_dict.items():
            for encoding in encodings:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
        
        print(f"Loaded {len(self.known_names)} face encodings "
              f"for {len(encodings_dict)} people")
    
    def process_frame(
        self,
        frame: np.ndarray,
        scale: float = 0.25
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            scale: Downscale factor for faster processing
            
        Returns:
            Annotated frame and list of detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
        
        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(
            small_frame, model=self.model
        )
        face_encodings = face_recognition.face_encodings(
            small_frame, face_locations
        )
        
        results = []
        
        for (top, right, bottom, left), encoding in zip(
            face_locations, face_encodings
        ):
            # Scale coordinates back to original size
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            
            # Find best match
            name = "Unknown"
            confidence = 0.0
            
            if self.known_encodings:
                distances = face_recognition.face_distance(
                    self.known_encodings, encoding
                )
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]
                
                if best_distance < self.tolerance:
                    name = self.known_names[best_idx]
                    confidence = 1.0 - best_distance
            
            results.append({
                "name": name,
                "confidence": confidence,
                "box": (left, top, right, bottom)
            })
            
            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label = f"{name}"
            if confidence > 0:
                label += f" ({confidence:.0%})"
            
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (left, bottom),
                (left + label_w + 10, bottom + label_h + 15),
                color, -1
            )
            cv2.putText(
                frame, label,
                (left + 5, bottom + label_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return frame, results


def process_image(
    recognizer: FaceRecognizer,
    image_path: str,
    output_path: Optional[str] = None
) -> None:
    """Process a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    annotated, results = recognizer.process_frame(frame, scale=1.0)
    
    # Print results
    print(f"\nDetected {len(results)} face(s):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} (confidence: {r['confidence']:.2%})")
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Saved to {output_path}")
    else:
        cv2.imshow("Face Recognition", annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    recognizer: FaceRecognizer,
    source: str | int,
    output_path: Optional[str] = None
) -> None:
    """
    Process video file or webcam stream.
    
    Args:
        recognizer: FaceRecognizer instance
        source: Video file path or camera index (0 for default webcam)
        output_path: Optional path to save output video
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing video... Press Q to quit")
    
    frame_count = 0
    process_every_n = 2  # Process every Nth frame for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame (skip some for performance)
        if frame_count % process_every_n == 0:
            annotated, results = recognizer.process_frame(frame)
        else:
            annotated = frame
        
        # Display FPS
        cv2.putText(
            annotated, f"Frame: {frame_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        if writer:
            writer.write(annotated)
        
        cv2.imshow("Face Recognition", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    if writer:
        writer.release()
        print(f"Saved output to {output_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Face detection and recognition system"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Image/video path or camera index (default: 0 for webcam)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for annotated image/video"
    )
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.6,
        help="Face matching tolerance (default: 0.6, lower=stricter)"
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Detection model: hog (faster) or cnn (accurate, needs GPU)"
    )
    
    args = parser.parse_args()
    
    recognizer = FaceRecognizer(
        tolerance=args.tolerance,
        model=args.model
    )
    
    source = args.source
    
    # Determine source type
    if source.isdigit():
        # Webcam
        process_video(recognizer, int(source), args.output)
    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Image file
        process_image(recognizer, source, args.output)
    else:
        # Video file
        process_video(recognizer, source, args.output)


if __name__ == "__main__":
    main()
