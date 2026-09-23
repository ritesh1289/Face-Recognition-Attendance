# Face Recognition Attendance System

A desktop college attendance prototype built with Python, Tkinter, SQLite, OpenCV, and `face_recognition`.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

   On Windows, `face-recognition` may require the Visual C++ Build Tools because of its `dlib` dependency.

3. Start the application:

   ```powershell
   py app.py
   ```

The SQLite database is created automatically as `attendance.db` in the project directory.

## Workflow

1. Enter student details and save the student.
2. Choose to capture face data. In the camera window, press Space when one face is visible, or Escape to cancel.
3. Open **Mark attendance** and start webcam recognition.
4. Review, filter, and export records from **Attendance report**.

Use biometric data only with informed consent and protect the database file appropriately.
