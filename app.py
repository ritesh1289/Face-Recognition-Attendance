"""Face Recognition Attendance System desktop application."""

from __future__ import annotations

import csv
import tkinter as tk
from datetime import date
from tkinter import filedialog, messagebox, ttk

from database import Database
from recognition import RecognitionError, capture_encoding, recognize_from_webcam


class AttendanceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Recognition Attendance")
        self.geometry("1050x680")
        self.minsize(900, 600)
        self.db = Database()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        self._build_ui()
        self.refresh_students()
        self.refresh_attendance()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#12304a")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#5a6872")
        style.configure("Accent.TButton", background="#087f5b", foreground="white", padding=9)
        style.configure("Treeview", rowheight=30)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(28, 22, 28, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Attendance Desk", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Register students, capture face data, and keep daily records in one place.", style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        self.registration_tab = ttk.Frame(self.notebook, padding=22)
        self.attendance_tab = ttk.Frame(self.notebook, padding=22)
        self.report_tab = ttk.Frame(self.notebook, padding=22)
        self.notebook.add(self.registration_tab, text="Student registration")
        self.notebook.add(self.attendance_tab, text="Mark attendance")
        self.notebook.add(self.report_tab, text="Attendance report")
        self._build_registration()
        self._build_attendance()
        self._build_report()

    def _build_registration(self) -> None:
        form = ttk.LabelFrame(self.registration_tab, text="New student", padding=18)
        form.pack(fill="x")
        self.student_fields: dict[str, ttk.Entry] = {}
        labels = [("Student ID", "student_id"), ("Full name", "name"), ("Course / class", "course"), ("Email", "email")]
        for row, (label, key) in enumerate(labels):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=7)
            entry = ttk.Entry(form, width=42)
            entry.grid(row=row, column=1, sticky="ew", pady=7)
            self.student_fields[key] = entry
        form.columnconfigure(1, weight=1)
        ttk.Button(form, text="Save student and capture face", style="Accent.TButton", command=self.register_student).grid(row=4, column=1, sticky="w", pady=(15, 2))

        list_frame = ttk.LabelFrame(self.registration_tab, text="Registered students", padding=12)
        list_frame.pack(fill="both", expand=True, pady=(20, 0))
        columns = ("student_id", "name", "course", "email", "face")
        self.students_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        headings = {"student_id": "Student ID", "name": "Name", "course": "Course", "email": "Email", "face": "Face data"}
        for column in columns:
            self.students_tree.heading(column, text=headings[column])
            self.students_tree.column(column, width=140)
        self.students_tree.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(list_frame, orient="vertical", command=self.students_tree.yview).pack(side="right", fill="y")

    def _build_attendance(self) -> None:
        ttk.Label(self.attendance_tab, text="Use the webcam to identify one registered student and mark today's attendance.", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 18))
        ttk.Button(self.attendance_tab, text="Start webcam recognition", style="Accent.TButton", command=self.mark_attendance).pack(anchor="w")
        self.attendance_status = ttk.Label(self.attendance_tab, text="Ready", padding=(0, 22), font=("Segoe UI", 13))
        self.attendance_status.pack(anchor="w")
        ttk.Label(self.attendance_tab, text="Attendance is recorded once per student per day. Recognition requires a webcam and the packages in requirements.txt.", style="Subtitle.TLabel", wraplength=700).pack(anchor="w", pady=(10, 0))

    def _build_report(self) -> None:
        controls = ttk.Frame(self.report_tab)
        controls.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Search").pack(side="left")
        self.search_entry = ttk.Entry(controls, width=28)
        self.search_entry.pack(side="left", padx=(8, 16))
        ttk.Label(controls, text="Date (YYYY-MM-DD)").pack(side="left")
        self.date_entry = ttk.Entry(controls, width=15)
        self.date_entry.insert(0, date.today().isoformat())
        self.date_entry.pack(side="left", padx=8)
        ttk.Button(controls, text="Filter", command=self.refresh_attendance).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear", command=self.clear_filters).pack(side="left")
        ttk.Button(controls, text="Export CSV", command=self.export_csv).pack(side="right")
        frame = ttk.Frame(self.report_tab)
        frame.pack(fill="both", expand=True)
        columns = ("date", "time", "student_id", "name", "course")
        self.attendance_tree = ttk.Treeview(frame, columns=columns, show="headings")
        headings = {"date": "Date", "time": "Time", "student_id": "Student ID", "name": "Name", "course": "Course"}
        for column in columns:
            self.attendance_tree.heading(column, text=headings[column])
            self.attendance_tree.column(column, width=150)
        self.attendance_tree.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(frame, orient="vertical", command=self.attendance_tree.yview).pack(side="right", fill="y")

    def register_student(self) -> None:
        values = {key: entry.get().strip() for key, entry in self.student_fields.items()}
        if not values["student_id"] or not values["name"] or not values["course"]:
            messagebox.showwarning("Incomplete form", "Student ID, name, and course are required.")
            return
        try:
            student_id = self.db.add_student(**values)
            if messagebox.askyesno("Capture face", "Student saved. Capture face data now?"):
                encoding = capture_encoding()
                self.db.update_face_encoding(student_id, encoding)
            for entry in self.student_fields.values():
                entry.delete(0, tk.END)
            self.refresh_students()
            messagebox.showinfo("Saved", "Student registration completed.")
        except (ValueError, RecognitionError) as error:
            messagebox.showerror("Could not complete registration", str(error))
        except Exception as error:
            if "UNIQUE constraint failed" in str(error):
                messagebox.showerror("Duplicate student", "That student ID is already registered.")
            else:
                messagebox.showerror("Could not save student", str(error))

    def mark_attendance(self) -> None:
        students = [dict(row) for row in self.db.students()]
        try:
            matched_id = recognize_from_webcam(students)
            if matched_id is None:
                self.attendance_status.configure(text="Recognition cancelled.")
                return
            marked = self.db.mark_attendance(matched_id)
            student = self.db.student(matched_id)
            name = student["name"] if student else "Student"
            self.attendance_status.configure(text=f"{name}: " + ("attendance marked." if marked else "already marked today."))
            self.refresh_attendance()
        except RecognitionError as error:
            messagebox.showerror("Recognition unavailable", str(error))

    def refresh_students(self) -> None:
        if not hasattr(self, "students_tree"):
            return
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        for student in self.db.students():
            self.students_tree.insert("", "end", values=(student["student_id"], student["name"], student["course"], student["email"], "Ready" if student["face_encoding"] else "Not captured"))

    def refresh_attendance(self) -> None:
        if not hasattr(self, "attendance_tree"):
            return
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        search = self.search_entry.get().strip() if hasattr(self, "search_entry") else ""
        selected_date = self.date_entry.get().strip() if hasattr(self, "date_entry") else ""
        for record in self.db.attendance(search, selected_date):
            self.attendance_tree.insert("", "end", values=(record["attended_on"], record["attended_at"].split("T")[-1], record["student_id"], record["name"], record["course"]))

    def clear_filters(self) -> None:
        self.search_entry.delete(0, tk.END)
        self.date_entry.delete(0, tk.END)
        self.refresh_attendance()

    def export_csv(self) -> None:
        records = self.db.attendance(self.search_entry.get().strip(), self.date_entry.get().strip())
        if not records:
            messagebox.showinfo("Nothing to export", "No attendance records match the current filters.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="attendance_report.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Student ID", "Name", "Course"])
            writer.writerows((row["attended_on"], row["attended_at"].split("T")[-1], row["student_id"], row["name"], row["course"]) for row in records)
        messagebox.showinfo("Export complete", f"Report exported to {path}")

    def close(self) -> None:
        self.db.close()
        self.destroy()


if __name__ == "__main__":
    AttendanceApp().mainloop()
