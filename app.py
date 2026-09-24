"""
Face Recognition Attendance System
Streamlit Web Application

Features:
- Student registration
- Browser camera face capture
- Face recognition
- Daily attendance
- Attendance search and filtering
- CSV report export
- SQLite database
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import Database
from recognition import (
    RecognitionError,
    capture_encoding_from_image,
    recognize_from_image,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

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
        font-size: 25px;
        font-weight: 600;
        color: #12304a;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .status-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7f9;
        border: 1px solid #e1e5e8;
        margin-bottom: 15px;
    }

    .success-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #eaf8f2;
        border: 1px solid #b7e4d1;
        color: #146c43;
    }

    .warning-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        color: #7a5b00;
    }

    .danger-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #fff0f0;
        border: 1px solid #f2b8b8;
        color: #a33a3a;
    }

    div[data-testid="stMetric"] {
        background-color: #f7f9fb;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5e8eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

@st.cache_resource
def get_database() -> Database:
    return Database()


db = get_database()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 Face Recognition Attendance System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Register students, capture face data, mark attendance, and generate attendance reports."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

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

students = db.students()
attendance_records = db.attendance()

st.sidebar.metric(
    "Registered Students",
    len(students),
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
        '<div class="section-title">Dashboard</div>',
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
            "📝 Attendance Records",
            len(attendance_records),
        )

    students_with_face = sum(
        1
        for student in students
        if student["face_encoding"]
    )

    with col3:
        st.metric(
            "🙂 Face Data",
            students_with_face,
        )

    st.markdown("---")

    st.subheader("System Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.markdown(
            """
            ### 👨‍🎓 Student Registration

            - Add student ID
            - Add student name
            - Add course/class
            - Add email
            - Capture face using browser camera
            - Store face encoding securely in SQLite
            """
        )

    with feature_col2:

        st.markdown(
            """
            ### 📷 Attendance

            - Open browser camera
            - Capture student's face
            - Compare with registered students
            - Automatically identify student
            - Mark attendance once per day
            """
        )

    st.markdown("---")

    st.subheader("📋 Recent Attendance")

    recent = db.attendance(limit=10)

    if recent:

        data = []

        for row in recent:

            data.append(
                {
                    "Date": row["attended_on"],
                    "Time": row["attended_at"].split("T")[-1],
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

        st.info("No attendance records available yet.")


# =========================================================
# STUDENT REGISTRATION
# =========================================================

elif page == "👨‍🎓 Student Registration":

    st.markdown(
        '<div class="section-title">👨‍🎓 Student Registration</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Enter the student information and then capture one clear face image."
    )

    col1, col2 = st.columns(2)

    with col1:

        student_id = st.text_input(
            "Student ID *",
            placeholder="Example: STU001",
        )

        student_name = st.text_input(
            "Full Name *",
            placeholder="Example: Ritesh Patwa",
        )

        course = st.text_input(
            "Course / Class *",
            placeholder="Example: B.Sc IT",
        )

    with col2:

        email = st.text_input(
            "Email",
            placeholder="student@example.com",
        )

        st.markdown("### 📷 Capture Face")

        face_image = st.camera_input(
            "Take a clear photo of the student's face"
        )

    st.markdown("---")

    if st.button(
        "💾 Register Student",
        type="primary",
        use_container_width=True,
    ):

        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

        if not student_id.strip():
            st.warning("Student ID is required.")
            st.stop()

        if not student_name.strip():
            st.warning("Student name is required.")
            st.stop()

        if not course.strip():
            st.warning("Course / class is required.")
            st.stop()

        if face_image is None:
            st.warning(
                "Please capture the student's face before registering."
            )
            st.stop()

        # ---------------------------------------------
        # CHECK DUPLICATE
        # ---------------------------------------------

        existing = db.student_by_student_id(
            student_id.strip()
        )

        if existing:
            st.error(
                f"Student ID '{student_id}' is already registered."
            )
            st.stop()

        # ---------------------------------------------
        # FACE ENCODING
        # ---------------------------------------------

        try:

            with st.spinner("Processing face..."):

                encoding = capture_encoding_from_image(
                    face_image.getvalue()
                )

        except RecognitionError as error:

            st.error(
                f"Face capture failed: {error}"
            )

            st.stop()

        # ---------------------------------------------
        # SAVE STUDENT
        # ---------------------------------------------

        try:

            db.add_student(
                student_id=student_id.strip(),
                name=student_name.strip(),
                course=course.strip(),
                email=email.strip(),
                face_encoding=encoding,
            )

            st.success(
                f"✅ {student_name} registered successfully!"
            )

            st.balloons()

            st.info(
                "The student's face data has been saved."
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
        '<div class="section-title">📷 Mark Attendance</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Allow camera access and capture the face of a registered student."
    )

    students = db.students()

    students_with_faces = [
        student
        for student in students
        if student["face_encoding"]
    ]

    if not students_with_faces:

        st.warning(
            "No students with face data are registered yet."
        )

        st.info(
            "Go to 'Student Registration' and register a student first."
        )

    else:

        st.write(
            f"Registered face profiles available: "
            f"**{len(students_with_faces)}**"
        )

        attendance_image = st.camera_input(
            "Capture face for attendance"
        )

        if attendance_image is not None:

            try:

                with st.spinner(
                    "Recognizing face..."
                ):

                    matched_id = recognize_from_image(
                        attendance_image.getvalue(),
                        students_with_faces,
                    )

                if matched_id is None:

                    st.error(
                        "❌ Face not recognized."
                    )

                    st.info(
                        "Make sure the student is registered and "
                        "the face is clearly visible."
                    )

                else:

                    student = db.student(
                        matched_id
                    )

                    if student is None:

                        st.error(
                            "Recognized student could not be found."
                        )

                    else:

                        marked = db.mark_attendance(
                            matched_id
                        )

                        if marked:

                            st.success(
                                f"✅ Attendance marked for "
                                f"**{student['name']}**"
                            )

                            st.write(
                                f"Student ID: **{student['student_id']}**"
                            )

                            st.write(
                                f"Course: **{student['course']}**"
                            )

                        else:

                            st.warning(
                                f"⚠️ {student['name']} "
                                "has already been marked present today."
                            )

            except RecognitionError as error:

                st.error(
                    f"Recognition error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Unexpected error: {error}"
                )


# =========================================================
# ATTENDANCE REPORT
# =========================================================

elif page == "📊 Attendance Report":

    st.markdown(
        '<div class="section-title">📊 Attendance Report</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        search = st.text_input(
            "🔍 Search",
            placeholder="Name, Student ID or Course",
        )

    with col2:

        selected_date = st.date_input(
            "📅 Attendance Date",
            value=None,
        )

    date_filter = ""

    if selected_date:

        date_filter = selected_date.isoformat()

    records = db.attendance(
        search=search.strip(),
        date=date_filter,
    )

    if records:

        report_data = []

        for record in records:

            report_data.append(
                {
                    "Date": record["attended_on"],
                    "Time": record["attended_at"].split("T")[-1],
                    "Student ID": record["student_id"],
                    "Name": record["name"],
                    "Course": record["course"],
                }
            )

        dataframe = pd.DataFrame(
            report_data
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------
        # CSV EXPORT
        # ---------------------------------------------

        csv_data = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download CSV Report",
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
    "Streamlit + Python + SQLite"
)
