"""Webcam capture and face recognition helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - handled by the GUI at runtime
    cv2 = None

try:
    import face_recognition
except ImportError:  # pragma: no cover - handled by the GUI at runtime
    face_recognition = None


class RecognitionError(RuntimeError):
    """Raised when camera or recognition support is not available."""


def dependency_status() -> str:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if face_recognition is None:
        missing.append("face-recognition")
    return "" if not missing else "Missing package(s): " + ", ".join(missing)


def capture_encoding(window_title: str = "Face capture") -> bytes:
    if cv2 is None or face_recognition is None:
        raise RecognitionError(dependency_status())

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RecognitionError("Could not open the webcam.")

    encoding = None
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RecognitionError("Could not read a frame from the webcam.")
            display = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            for top, right, bottom, left in locations:
                cv2.rectangle(display, (left, top), (right, bottom), (0, 190, 120), 2)
            cv2.putText(
                display,
                "Look at camera | SPACE: capture | ESC: cancel",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and len(locations) == 1:
                candidate = face_recognition.face_encodings(rgb_frame, locations)[0]
                encoding = pickle.dumps(candidate)
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if encoding is None:
        raise RecognitionError("No single face was captured. Please try again.")
    return encoding


def recognize_from_webcam(students: list[dict], window_title: str = "Mark attendance") -> int | None:
    if cv2 is None or face_recognition is None:
        raise RecognitionError(dependency_status())

    known = [(student["id"], pickle.loads(student["face_encoding"])) for student in students if student.get("face_encoding")]
    if not known:
        raise RecognitionError("No student face data is available yet.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RecognitionError("Could not open the webcam.")
    matched_id = None
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            for location, candidate in zip(locations, encodings):
                matches = face_recognition.compare_faces([item[1] for item in known], candidate, tolerance=0.48)
                if True in matches:
                    matched_id = known[matches.index(True)][0]
                    break
            cv2.putText(frame, "ESC: cancel", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_title, frame)
            if matched_id is not None or (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return matched_id
