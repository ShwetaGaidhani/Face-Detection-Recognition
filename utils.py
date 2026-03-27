"""Utilities for face detection and recognition."""

import os
import pickle
from pathlib import Path
from typing import Optional

import face_recognition
import numpy as np

KNOWN_FACES_DIR = Path("known_faces")
ENCODINGS_FILE = Path("encodings/face_encodings.pkl")


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    ENCODINGS_FILE.parent.mkdir(exist_ok=True)


def load_encodings() -> dict[str, list[np.ndarray]]:
    """Load saved face encodings from disk."""
    if ENCODINGS_FILE.exists():
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_encodings(encodings: dict[str, list[np.ndarray]]) -> None:
    """Save face encodings to disk."""
    ensure_directories()
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)


def get_face_encoding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract face encoding from an image file.
    
    Returns None if no face is detected.
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if encodings:
        return encodings[0]  # Return first face found
    return None


def rebuild_encodings_from_images() -> dict[str, list[np.ndarray]]:
    """
    Rebuild all encodings from images in known_faces directory.
    
    Directory structure: known_faces/<person_name>/<image_files>
    """
    encodings = {}
    
    for person_dir in KNOWN_FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        encodings[person_name] = []
        
        for image_file in person_dir.iterdir():
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                encoding = get_face_encoding(str(image_file))
                if encoding is not None:
                    encodings[person_name].append(encoding)
                    print(f"  Encoded: {image_file.name}")
    
    return encodings
