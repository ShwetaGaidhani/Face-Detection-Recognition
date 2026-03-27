"""Register new faces into the recognition system."""

import argparse
import shutil
from pathlib import Path

import cv2

from utils import (
    KNOWN_FACES_DIR,
    ensure_directories,
    get_face_encoding,
    load_encodings,
    save_encodings,
)


def register_from_image(name: str, image_path: str) -> bool:
    """
    Register a new person from an existing image file.
    
    Args:
        name: Person's name (used as identifier)
        image_path: Path to the image file
        
    Returns:
        True if registration successful, False otherwise
    """
    ensure_directories()
    
    # Create person's directory
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    
    # Copy image to known_faces directory
    src_path = Path(image_path)
    dest_path = person_dir / src_path.name
    
    # Handle duplicate filenames
    counter = 1
    while dest_path.exists():
        dest_path = person_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
        counter += 1
    
    shutil.copy2(image_path, dest_path)
    
    # Extract and save encoding
    encoding = get_face_encoding(str(dest_path))
    if encoding is None:
        print(f"Error: No face detected in {image_path}")
        dest_path.unlink()  # Remove copied file
        return False
    
    encodings = load_encodings()
    if name not in encodings:
        encodings[name] = []
    encodings[name].append(encoding)
    save_encodings(encodings)
    
    print(f"Successfully registered {name} from {image_path}")
    return True


def register_from_webcam(name: str, num_samples: int = 5) -> bool:
    """
    Capture and register face samples from webcam.
    
    Args:
        name: Person's name
        num_samples: Number of face samples to capture
        
    Returns:
        True if at least one sample was captured
    """
    ensure_directories()
    
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return False
    
    print(f"Capturing {num_samples} samples for {name}")
    print("Press SPACE to capture, Q to quit early")
    
    samples_captured = 0
    encodings = load_encodings()
    if name not in encodings:
        encodings[name] = []
    
    # Load face detector for preview
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    while samples_captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces for preview (using Haar for speed)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        # Draw rectangles on detected faces
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show status
        status = f"Samples: {samples_captured}/{num_samples} | SPACE=capture, Q=quit"
        cv2.putText(
            display_frame, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        cv2.imshow("Register Face", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # Space to capture
            # Save frame temporarily
            temp_path = person_dir / f"sample_{samples_captured}.jpg"
            cv2.imwrite(str(temp_path), frame)
            
            # Try to extract encoding
            encoding = get_face_encoding(str(temp_path))
            if encoding is not None:
                encodings[name].append(encoding)
                samples_captured += 1
                print(f"  Captured sample {samples_captured}")
            else:
                print("  No face detected, try again")
                temp_path.unlink()
    
    cap.release()
    cv2.destroyAllWindows()
    
    if samples_captured > 0:
        save_encodings(encodings)
        print(f"Registered {name} with {samples_captured} samples")
        return True
    
    return False


def list_registered() -> None:
    """List all registered people and their sample counts."""
    encodings = load_encodings()
    
    if not encodings:
        print("No registered faces found.")
        return
    
    print("\nRegistered People:")
    print("-" * 30)
    for name, encs in sorted(encodings.items()):
        print(f"  {name}: {len(encs)} sample(s)")
    print()


def remove_person(name: str) -> bool:
    """Remove a person from the database."""
    encodings = load_encodings()
    
    if name not in encodings:
        print(f"Person '{name}' not found in database")
        return False
    
    # Remove from encodings
    del encodings[name]
    save_encodings(encodings)
    
    # Remove images directory
    person_dir = KNOWN_FACES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)
    
    print(f"Removed {name} from database")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Register faces for recognition system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add from image
    img_parser = subparsers.add_parser("image", help="Register from image file")
    img_parser.add_argument("name", help="Person's name")
    img_parser.add_argument("path", help="Path to image file")
    
    # Add from webcam
    cam_parser = subparsers.add_parser("webcam", help="Capture from webcam")
    cam_parser.add_argument("name", help="Person's name")
    cam_parser.add_argument(
        "-n", "--samples", type=int, default=5,
        help="Number of samples to capture (default: 5)"
    )
    
    # List registered
    subparsers.add_parser("list", help="List registered people")
    
    # Remove person
    rm_parser = subparsers.add_parser("remove", help="Remove a person")
    rm_parser.add_argument("name", help="Person's name to remove")
    
    args = parser.parse_args()
    
    if args.command == "image":
        register_from_image(args.name, args.path)
    elif args.command == "webcam":
        register_from_webcam(args.name, args.samples)
    elif args.command == "list":
        list_registered()
    elif args.command == "remove":
        remove_person(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
