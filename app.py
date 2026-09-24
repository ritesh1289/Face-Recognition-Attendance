"""
Face Recognition Attendance System
Streamlit Web Application
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #12304a;
    }

    .subtitle {
        color: #5a6872;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #12304a;
        margin-bottom: 20px;
    }

    .student-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background: #f8f9fa;
        margin-bottom: 10px;
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
def get_database():

    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

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

    return connection


db = get_database()


# =========================================================
# OPENCV FACE DETECTOR
# =========================================================

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


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
# FACE DETECTION
# =========================================================

def detect_face(image_bytes):

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

    gray = cv2.equalizeHist(
        gray
    )

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
# FACE TO BYTES
# =========================================================

def face_to_bytes(face):

    success, encoded = cv2.imencode(
        ".png",
        face,
    )

    if not success:

        raise ValueError(
            "Could not save face."
        )

    return encoded.tobytes()


# =========================================================
# TRAIN LBPH
# =========================================================

def create_lbph_model():

    if not hasattr(cv2, "face"):

        raise RuntimeError(
            "OpenCV Face module is unavailable. "
            "Check opencv-contrib-python-headless."
        )

    rows = db.execute(
        """
        SELECT id, face_image
        FROM students
        WHERE face_image IS NOT NULL
        """
    ).fetchall()

    if not rows:

        return None

    images = []

    labels = []

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

        except Exception:
            continue

    if not images:

        return None

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

    return model


# =========================================================
# RECOGNIZE
# =========================================================

def recognize_student(image_bytes):

    face = detect_face(
        image_bytes
    )

    if face is None:

        return None, (
            "Please capture exactly one clear face."
        )

    model = create_lbph_model()

    if model is None:

        return None, (
            "No registered face data is available."
        )

    try:

        label, confidence = model.predict(
            face
        )

    except Exception as error:

        return None, str(error)

    # Lower is better for LBPH.
    threshold = 65.0

    if confidence > threshold:

        return None, (
            "Face detected, but student "
            "could not be recognized."
        )

    student = get_student_by_id(
        int(label)
    )

    if student is None:

        return None, (
            "Student record not found."
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
# ATTENDANCE
# =========================================================

def get_attendance(
    search="",
    selected_date="",
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
    "Student registration, face recognition and attendance management"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

students = get_students()

attendance = get_attendance()

students_with_faces = [
    student
    for student in students
    if student["face_image"] is not None
]


st.sidebar.title("📌 Menu")

page = st.sidebar.radio(
    "Select",
    [
        "🏠 Dashboard",
        "👨‍🎓 Student Registration",
        "📷 Mark Attendance",
        "📋 Registered Students",
        "📊 Attendance Report",
    ],
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Students",
    len(students),
)

st.sidebar.metric(
    "Face Profiles",
    len(students_with_faces),
)

st.sidebar.metric(
    "Attendance",
    len(attendance),
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

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "👨‍🎓 Students",
            len(students),
        )

    with c2:
        st.metric(
            "🙂 Face Profiles",
            len(students_with_faces),
        )

    with c3:
        st.metric(
            "📝 Attendance",
            len(attendance),
        )

    st.markdown("---")

    st.subheader(
        "📋 Recent Attendance"
    )

    recent = get_attendance()[:10]

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
        "Fill in ALL student details and then capture the student's face."
    )

    # -----------------------------------------------------
    # STUDENT ID - CLEARLY VISIBLE
    # -----------------------------------------------------

    st.subheader(
        "1️⃣ Student Information"
    )

    student_id = st.text_input(
        "Student ID *",
        placeholder="Enter Student ID — example: STU001",
        key="student_id_input",
    )

    name = st.text_input(
        "Full Name *",
        placeholder="Enter student's full name",
        key="name_input",
    )

    course = st.text_input(
        "Course / Class *",
        placeholder="Example: B.Sc Information Technology",
        key="course_input",
    )

    email = st.text_input(
        "Email",
        placeholder="student@example.com",
        key="email_input",
    )

    st.markdown("---")

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    st.subheader(
        "2️⃣ Capture Student Face"
    )

    st.write(
        "Make sure only ONE person's face is visible."
    )

    captured_image = st.camera_input(
        "📷 Open Camera and Capture Student Face",
        key="registration_camera",
    )

    st.markdown("---")

    # -----------------------------------------------------
    # REGISTER BUTTON
    # -----------------------------------------------------

    st.subheader(
        "3️⃣ Save Student"
    )

    if st.button(
        "💾 Register Student",
        type="primary",
        use_container_width=True,
    ):

        # Student ID
        if not student_id.strip():

            st.error(
                "❌ Student ID is required."
            )

            st.stop()

        # Name
        if not name.strip():

            st.error(
                "❌ Full Name is required."
            )

            st.stop()

        # Course
        if not course.strip():

            st.error(
                "❌ Course / Class is required."
            )

            st.stop()

        # Camera
        if captured_image is None:

            st.error(
                "❌ Please capture the student's face."
            )

            st.stop()

        # Duplicate
        existing = get_student_by_student_id(
            student_id.strip()
        )

        if existing:

            st.error(
                f"❌ Student ID '{student_id}' "
                "already exists."
            )

            st.stop()

        # Face
        with st.spinner(
            "Processing student face..."
        ):

            face = detect_face(
                captured_image.getvalue()
            )

        if face is None:

            st.error(
                "❌ Exactly one face must be visible."
            )

            st.info(
                "Please capture another image "
                "with one clear face."
            )

            st.stop()

        # Save face
        face_bytes = face_to_bytes(
            face
        )

        # Database
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
                "✅ Student registered successfully!"
            )

            st.write(
                f"**Student ID:** {student_id}"
            )

            st.write(
                f"**Name:** {name}"
            )

            st.write(
                f"**Course:** {course}"
            )

            st.write(
                f"**Email:** {email if email else 'Not provided'}"
            )

            st.balloons()

        except sqlite3.IntegrityError:

            st.error(
                "This Student ID already exists."
            )

        except Exception as error:

            st.error(
                f"Database error: {error}"
            )


# =========================================================
# REGISTERED STUDENTS
# =========================================================

elif page == "📋 Registered Students":

    st.markdown(
        '<div class="section-title">'
        "📋 Registered Students"
        "</div>",
        unsafe_allow_html=True,
    )

    students = get_students()

    if not students:

        st.info(
            "No students registered yet."
        )

        st.write(
            "Go to **Student Registration** "
            "to add the first student."
        )

    else:

        st.success(
            f"{len(students)} student(s) registered."
        )

        table_data = []

        for student in students:

            table_data.append(
                {
                    "Student ID": student["student_id"],
                    "Name": student["name"],
                    "Course": student["course"],
                    "Email": student["email"],
                    "Face Data": (
                        "✅ Captured"
                        if student["face_image"]
                        else "❌ Not Captured"
                    ),
                    "Registered": student["created_at"],
                }
            )

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True,
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
            "⚠️ No student face profiles are available."
        )

        st.info(
            "Register a student first."
        )

    else:

        st.success(
            f"{len(students_with_faces)} "
            "face profile(s) available."
        )

        attendance_image = st.camera_input(
            "📷 Capture Face for Attendance",
            key="attendance_camera",
        )

        if attendance_image is not None:

            with st.spinner(
                "Recognizing student..."
            ):

                student, result = recognize_student(
                    attendance_image.getvalue()
                )

            if student is None:

                st.error(
                    "❌ " + str(result)
                )

            else:

                confidence = result

                marked = mark_attendance(
                    student["id"]
                )

                if marked:

                    st.success(
                        f"✅ Attendance marked for "
                        f"**{student['name']}**"
                    )

                else:

                    st.warning(
                        f"⚠️ {student['name']} "
                        "is already marked present today."
                    )

                st.write(
                    f"**Student ID:** "
                    f"{student['student_id']}"
                )

                st.write(
                    f"**Name:** "
                    f"{student['name']}"
                )

                st.write(
                    f"**Course:** "
                    f"{student['course']}"
                )

                st.write(
                    f"Recognition distance: "
                    f"{confidence:.2f}"
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

    c1, c2 = st.columns(2)

    with c1:

        search = st.text_input(
            "🔍 Search",
            placeholder="Name / Student ID / Course",
        )

    with c2:

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
            "⬇️ Download CSV",
            data=csv_data,
            file_name="attendance_report.csv",
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
