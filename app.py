"""
Face Recognition Attendance System
Streamlit Web Application

Technology:
- Python
- Streamlit
- OpenCV
- LBPH Face Recognition
- SQLite
- Pandas

Features:
- Student registration
- Browser camera capture
- Face detection
- Face recognition
- Attendance once per day
- Attendance reports
- Search and date filtering
- CSV export
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #12304a;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #5a6872;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        color: #12304a;
        margin-bottom: 18px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        background: #f5f7f9;
        border: 1px solid #e1e5e8;
        margin-bottom: 15px;
    }

    .success-card {
        padding: 18px;
        border-radius: 12px;
        background: #eaf8f2;
        border: 1px solid #b7e4d1;
    }

    .warning-card {
        padding: 18px;
        border-radius: 12px;
        background: #fff8e1;
        border: 1px solid #ffe082;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATABASE
# =========================================================

DATABASE_FILE = Path("attendance.db")


@st.cache_resource
def get_connection():
    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    initialize_database(connection)

    return connection


def initialize_database(connection):

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT NOT NULL UNIQUE,

            name TEXT NOT NULL,

            course TEXT NOT NULL,

            email TEXT DEFAULT '',

            face_image BLOB,

            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER NOT NULL,

            attended_on TEXT NOT NULL,

            attended_at TEXT NOT NULL,

            UNIQUE(student_id, attended_on),

            FOREIGN KEY(student_id)
                REFERENCES students(id)
                ON DELETE CASCADE
        );
        """
    )

    connection.commit()


db = get_connection()


# =========================================================
# FACE DETECTOR
# =========================================================

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


# =========================================================
# CHECK OPENCV
# =========================================================

if face_detector.empty():

    st.error(
        "OpenCV face detection model could not be loaded."
    )

    st.stop()


# =========================================================
# FACE DETECTION
# =========================================================

def detect_face(image_bytes: bytes):

    """
    Detect exactly one face from uploaded/camera image.

    Returns:
        grayscale cropped face
        OR None
    """

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    face = gray[
        y:y + h,
        x:x + w
    ]

    if face.size == 0:
        return None

    face = cv2.resize(
        face,
        (200, 200),
    )

    return face


# =========================================================
# SAVE FACE IMAGE
# =========================================================

def face_to_bytes(face):

    success, encoded = cv2.imencode(
        ".png",
        face,
    )

    if not success:
        raise ValueError(
            "Could not save face image."
        )

    return encoded.tobytes()


# =========================================================
# GET STUDENTS
# =========================================================

def get_students():

    return list(
        db.execute(
            """
            SELECT *
            FROM students
            ORDER BY name COLLATE NOCASE
            """
        )
    )


# =========================================================
# GET STUDENT
# =========================================================

def get_student_by_id(student_id):

    return db.execute(
        """
        SELECT *
        FROM students
        WHERE id = ?
        """,
        (student_id,),
    ).fetchone()


# =========================================================
# GET STUDENT BY STUDENT ID
# =========================================================

def get_student_by_student_id(student_id):

    return db.execute(
        """
        SELECT *
        FROM students
        WHERE student_id = ?
        """,
        (student_id,),
    ).fetchone()


# =========================================================
# TRAIN LBPH MODEL
# =========================================================

def create_lbph_model():

    # Check OpenCV contrib
    if not hasattr(cv2, "face"):

        raise RuntimeError(
            "OpenCV Face module is unavailable. "
            "Make sure opencv-contrib-python-headless "
            "is installed."
        )

    rows = db.execute(
        """
        SELECT id, face_image
        FROM students
        WHERE face_image IS NOT NULL
        """
    ).fetchall()

    if not rows:

        return None, []

    images = []

    labels = []

    valid_student_ids = []

    for row in rows:

        try:

            image_array = np.frombuffer(
                row["face_image"],
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (200, 200),
            )

            images.append(image)

            labels.append(
                int(row["id"])
            )

            valid_student_ids.append(
                int(row["id"])
            )

        except Exception:
            continue

    if not images:

        return None, []

    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    model.train(
        images,
        np.array(
            labels,
            dtype=np.int32,
        ),
    )

    return model, valid_student_ids


# =========================================================
# RECOGNIZE STUDENT
# =========================================================

def recognize_student(
    image_bytes: bytes,
):

    face = detect_face(
        image_bytes
    )

    if face is None:

        return None, (
            "Please capture an image containing "
            "exactly one clear face."
        )

    model, known_ids = create_lbph_model()

    if model is None:

        return None, (
            "No registered face data is available."
        )

    try:

        label, confidence = model.predict(
            face
        )

    except Exception as error:

        return None, (
            f"Face recognition failed: {error}"
        )

    # Lower LBPH confidence is better.
    #
    # This threshold can be adjusted.
    # 65 is used as a reasonably strict starting point.
    threshold = 65.0

    if confidence > threshold:

        return None, (
            "Face was detected, but the student "
            "could not be confidently recognized."
        )

    if label not in known_ids:

        return None, (
            "Recognized face does not match "
            "a registered student."
        )

    student = get_student_by_id(
        int(label)
    )

    if student is None:

        return None, (
            "Student record was not found."
        )

    return student, confidence


# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(student_id):

    now = datetime.now()

    cursor = db.execute(
        """
        INSERT OR IGNORE INTO attendance
        (
            student_id,
            attended_on,
            attended_at
        )
        VALUES (?, ?, ?)
        """,
        (
            student_id,
            now.date().isoformat(),
            now.isoformat(
                timespec="seconds"
            ),
        ),
    )

    db.commit()

    return cursor.rowcount == 1


# =========================================================
# GET ATTENDANCE
# =========================================================

def get_attendance(
    search="",
    selected_date="",
    limit=None,
):

    query = """
        SELECT
            a.attended_on,
            a.attended_at,
            s.student_id,
            s.name,
            s.course
        FROM attendance a
        JOIN students s
            ON s.id = a.student_id
        WHERE 1 = 1
    """

    values = []

    if search:

        query += """
            AND (
                s.name LIKE ?
                OR s.student_id LIKE ?
                OR s.course LIKE ?
            )
        """

        wildcard = f"%{search}%"

        values.extend(
            [
                wildcard,
                wildcard,
                wildcard,
            ]
        )

    if selected_date:

        query += """
            AND a.attended_on = ?
        """

        values.append(
            selected_date
        )

    query += """
        ORDER BY a.attended_at DESC
    """

    if limit is not None:

        query += " LIMIT ?"

        values.append(
            limit
        )

    return list(
        db.execute(
            query,
            values,
        )
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    "🎓 Face Recognition Attendance System"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Register students, capture face data, "
    "recognize students, and maintain attendance records."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

students = get_students()

attendance_records = get_attendance()

students_with_faces = [
    student
    for student in students
    if student["face_image"] is not None
]


st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "👨‍🎓 Student Registration",
        "📷 Mark Attendance",
        "📊 Attendance Report",
    ],
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Registered Students",
    len(students),
)

st.sidebar.metric(
    "Face Profiles",
    len(students_with_faces),
)

st.sidebar.metric(
    "Attendance Records",
    len(attendance_records),
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">'
        "🏠 Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👨‍🎓 Students",
            len(students),
        )

    with col2:

        st.metric(
            "🙂 Face Profiles",
            len(students_with_faces),
        )

    with col3:

        st.metric(
            "📝 Attendance",
            len(attendance_records),
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader(
            "👨‍🎓 Student Registration"
        )

        st.write(
            """
            Register students with:

            • Student ID  
            • Full name  
            • Course / class  
            • Email  
            • Face photograph
            """
        )

    with right:

        st.subheader(
            "📷 Face Attendance"
        )

        st.write(
            """
            Attendance is recorded by:

            • Opening the browser camera  
            • Capturing a face  
            • Detecting the face  
            • Comparing it with registered faces  
            • Recording attendance once per day
            """
        )

    st.markdown("---")

    st.subheader(
        "📋 Recent Attendance"
    )

    recent = get_attendance(
        limit=10
    )

    if recent:

        data = []

        for row in recent:

            data.append(
                {
                    "Date": row["attended_on"],
                    "Time": row["attended_at"]
                    .split("T")[-1],
                    "Student ID": row["student_id"],
                    "Name": row["name"],
                    "Course": row["course"],
                }
            )

        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No attendance records yet."
        )


# =========================================================
# STUDENT REGISTRATION
# =========================================================

elif page == "👨‍🎓 Student Registration":

    st.markdown(
        '<div class="section-title">'
        "👨‍🎓 Student Registration"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "Enter the student details and capture "
        "one clear face image."
    )

    col1, col2 = st.columns(2)

    with col1:

        student_id = st.text_input(
            "Student ID *",
            placeholder="Example: STU001",
        )

        name = st.text_input(
            "Full Name *",
            placeholder="Example: Ritesh Patwa",
        )

        course = st.text_input(
            "Course / Class *",
            placeholder="Example: B.Sc IT",
        )

        email = st.text_input(
            "Email",
            placeholder="student@example.com",
        )

    with col2:

        st.subheader(
            "📷 Face Capture"
        )

        captured_image = st.camera_input(
            "Take a clear face photo"
        )

    st.markdown("---")

    if st.button(
        "💾 Register Student",
        type="primary",
        use_container_width=True,
    ):

        # -----------------------------------------
        # VALIDATION
        # -----------------------------------------

        if not student_id.strip():

            st.warning(
                "Student ID is required."
            )

            st.stop()

        if not name.strip():

            st.warning(
                "Full name is required."
            )

            st.stop()

        if not course.strip():

            st.warning(
                "Course / class is required."
            )

            st.stop()

        if captured_image is None:

            st.warning(
                "Please capture the student's face."
            )

            st.stop()

        # -----------------------------------------
        # DUPLICATE CHECK
        # -----------------------------------------

        existing = get_student_by_student_id(
            student_id.strip()
        )

        if existing:

            st.error(
                f"Student ID '{student_id}' "
                "is already registered."
            )

            st.stop()

        # -----------------------------------------
        # DETECT FACE
        # -----------------------------------------

        try:

            with st.spinner(
                "Detecting face..."
            ):

                face = detect_face(
                    captured_image.getvalue()
                )

            if face is None:

                st.error(
                    "Exactly one face must be visible "
                    "in the photograph."
                )

                st.info(
                    "Please look directly at the camera "
                    "and capture the image again."
                )

                st.stop()

            face_bytes = face_to_bytes(
                face
            )

        except Exception as error:

            st.error(
                f"Face processing failed: {error}"
            )

            st.stop()

        # -----------------------------------------
        # SAVE
        # -----------------------------------------

        try:

            db.execute(
                """
                INSERT INTO students
                (
                    student_id,
                    name,
                    course,
                    email,
                    face_image,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id.strip(),
                    name.strip(),
                    course.strip(),
                    email.strip(),
                    face_bytes,
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                ),
            )

            db.commit()

            st.success(
                f"✅ {name} registered successfully!"
            )

            st.info(
                "Face data has been saved successfully."
            )

            st.balloons()

            st.rerun()

        except sqlite3.IntegrityError:

            st.error(
                "That Student ID is already registered."
            )

        except Exception as error:

            st.error(
                f"Could not save student: {error}"
            )


# =========================================================
# MARK ATTENDANCE
# =========================================================

elif page == "📷 Mark Attendance":

    st.markdown(
        '<div class="section-title">'
        "📷 Mark Attendance"
        "</div>",
        unsafe_allow_html=True,
    )

    if not students_with_faces:

        st.warning(
            "No students with face profiles are registered."
        )

        st.info(
            "First go to Student Registration "
            "and register at least one student."
        )

    else:

        st.success(
            f"{len(students_with_faces)} "
            "student face profile(s) available."
        )

        st.write(
            "Allow camera access and capture the "
            "student's face."
        )

        attendance_image = st.camera_input(
            "📷 Capture Face for Attendance"
        )

        if attendance_image is not None:

            try:

                with st.spinner(
                    "Recognizing student..."
                ):

                    student, confidence = recognize_student(
                        attendance_image.getvalue()
                    )

                if student is None:

                    st.error(
                        "❌ " + str(confidence)
                    )

                else:

                    marked = mark_attendance(
                        student["id"]
                    )

                    if marked:

                        st.success(
                            f"✅ Attendance marked for "
                            f"**{student['name']}**"
                        )

                        st.write(
                            f"**Student ID:** "
                            f"{student['student_id']}"
                        )

                        st.write(
                            f"**Course:** "
                            f"{student['course']}"
                        )

                        st.write(
                            f"Recognition confidence: "
                            f"{confidence:.2f}"
                        )

                    else:

                        st.warning(
                            f"⚠️ {student['name']} "
                            "has already been marked "
                            "present today."
                        )

            except Exception as error:

                st.error(
                    f"Recognition error: {error}"
                )


# =========================================================
# ATTENDANCE REPORT
# =========================================================

elif page == "📊 Attendance Report":

    st.markdown(
        '<div class="section-title">'
        "📊 Attendance Report"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        search = st.text_input(
            "🔍 Search",
            placeholder=(
                "Name, Student ID or Course"
            ),
        )

    with col2:

        selected_date = st.date_input(
            "📅 Date",
            value=None,
        )

    date_filter = ""

    if selected_date:

        date_filter = selected_date.isoformat()

    records = get_attendance(
        search=search.strip(),
        selected_date=date_filter,
    )

    if records:

        data = []

        for row in records:

            data.append(
                {
                    "Date": row["attended_on"],
                    "Time": row["attended_at"]
                    .split("T")[-1],
                    "Student ID": row["student_id"],
                    "Name": row["name"],
                    "Course": row["course"],
                }
            )

        dataframe = pd.DataFrame(
            data
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name=(
                "attendance_report.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    else:

        st.info(
            "No attendance records found."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Face Recognition Attendance System | "
    "Streamlit + Python + OpenCV + SQLite"
)
