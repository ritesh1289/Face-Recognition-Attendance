"""
Face recognition functions for Streamlit.

The browser camera is handled by Streamlit's
st.camera_input().

This module only receives the captured image
and performs face recognition.
"""

from __future__ import annotations

import pickle
from io import BytesIO

try:
    import face_recognition

except ImportError:

    face_recognition = None


class RecognitionError(RuntimeError):
    """Raised when face recognition is unavailable."""


# ---------------------------------------------------------
# DEPENDENCY CHECK
# ---------------------------------------------------------

def dependency_status() -> str:

    if face_recognition is None:

        return (
            "The face-recognition package is not installed. "
            "Please check requirements.txt."
        )

    return ""


# ---------------------------------------------------------
# CAPTURE / ENCODE FACE FROM STREAMLIT IMAGE
# ---------------------------------------------------------

def capture_encoding_from_image(
    image_bytes: bytes,
) -> bytes:

    if face_recognition is None:

        raise RecognitionError(
            dependency_status()
        )

    try:

        image = face_recognition.load_image_file(
            BytesIO(image_bytes)
        )

    except Exception as error:

        raise RecognitionError(
            f"Could not read the captured image: {error}"
        )

    try:

        locations = face_recognition.face_locations(
            image
        )

    except Exception as error:

        raise RecognitionError(
            f"Could not detect face: {error}"
        )

    if len(locations) == 0:

        raise RecognitionError(
            "No face was detected. "
            "Please capture a clear image with the face visible."
        )

    if len(locations) > 1:

        raise RecognitionError(
            "Multiple faces detected. "
            "Please capture an image containing only one student."
        )

    try:

        encodings = face_recognition.face_encodings(
            image,
            locations,
        )

    except Exception as error:

        raise RecognitionError(
            f"Could not create face encoding: {error}"
        )

    if not encodings:

        raise RecognitionError(
            "Face encoding could not be generated."
        )

    return pickle.dumps(
        encodings[0]
    )


# ---------------------------------------------------------
# RECOGNIZE FACE
# ---------------------------------------------------------

def recognize_from_image(
    image_bytes: bytes,
    students: list,
) -> int | None:

    if face_recognition is None:

        raise RecognitionError(
            dependency_status()
        )

    if not students:

        raise RecognitionError(
            "No registered students are available."
        )

    # ---------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------

    try:

        image = face_recognition.load_image_file(
            BytesIO(image_bytes)
        )

    except Exception as error:

        raise RecognitionError(
            f"Could not read image: {error}"
        )

    # ---------------------------------------------
    # DETECT FACE
    # ---------------------------------------------

    try:

        locations = face_recognition.face_locations(
            image
        )

    except Exception as error:

        raise RecognitionError(
            f"Face detection failed: {error}"
        )

    if not locations:

        raise RecognitionError(
            "No face was detected."
        )

    # ---------------------------------------------
    # CREATE ENCODINGS
    # ---------------------------------------------

    try:

        encodings = face_recognition.face_encodings(
            image,
            locations,
        )

    except Exception as error:

        raise RecognitionError(
            f"Could not create face encoding: {error}"
        )

    if not encodings:

        raise RecognitionError(
            "Could not encode the detected face."
        )

    # ---------------------------------------------
    # LOAD REGISTERED FACES
    # ---------------------------------------------

    known_encodings = []

    known_ids = []

    for student in students:

        stored_encoding = student["face_encoding"]

        if not stored_encoding:

            continue

        try:

            encoding = pickle.loads(
                stored_encoding
            )

            known_encodings.append(
                encoding
            )

            known_ids.append(
                student["id"]
            )

        except Exception:
            # Ignore corrupted face data
            continue

    if not known_encodings:

        raise RecognitionError(
            "No valid registered face data is available."
        )

    # ---------------------------------------------
    # COMPARE
    # ---------------------------------------------

    tolerance = 0.48

    for candidate in encodings:

        try:

            distances = face_recognition.face_distance(
                known_encodings,
                candidate,
            )

            if len(distances) == 0:
                continue

            best_index = int(
                distances.argmin()
            )

            best_distance = float(
                distances[best_index]
            )

            if best_distance <= tolerance:

                return known_ids[
                    best_index
                ]

        except Exception as error:

            raise RecognitionError(
                f"Face comparison failed: {error}"
            )

    return None
