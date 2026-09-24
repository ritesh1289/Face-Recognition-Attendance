"""
SQLite database layer for the
Face Recognition Attendance System.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:

    def __init__(
        self,
        path: str = "attendance.db",
    ) -> None:

        self.path = Path(path)

        self.connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self.connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self.initialize()

    # -----------------------------------------------------
    # INITIALIZE DATABASE
    # -----------------------------------------------------

    def initialize(self) -> None:

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                student_id TEXT NOT NULL UNIQUE,

                name TEXT NOT NULL,

                course TEXT NOT NULL,

                email TEXT NOT NULL DEFAULT '',

                face_encoding BLOB,

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

        self.connection.commit()

    # -----------------------------------------------------
    # ADD STUDENT
    # -----------------------------------------------------

    def add_student(
        self,
        student_id: str,
        name: str,
        course: str,
        email: str = "",
        face_encoding: bytes | None = None,
    ) -> int:

        cursor = self.connection.execute(
            """
            INSERT INTO students
            (
                student_id,
                name,
                course,
                email,
                face_encoding,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                name,
                course,
                email,
                face_encoding,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    # -----------------------------------------------------
    # UPDATE FACE
    # -----------------------------------------------------

    def update_face_encoding(
        self,
        student_id: int,
        encoding: bytes,
    ) -> None:

        self.connection.execute(
            """
            UPDATE students
            SET face_encoding = ?
            WHERE id = ?
            """,
            (
                encoding,
                student_id,
            ),
        )

        self.connection.commit()

    # -----------------------------------------------------
    # GET ALL STUDENTS
    # -----------------------------------------------------

    def students(self) -> list[sqlite3.Row]:

        return list(
            self.connection.execute(
                """
                SELECT *
                FROM students
                ORDER BY name COLLATE NOCASE
                """
            )
        )

    # -----------------------------------------------------
    # GET STUDENT BY DATABASE ID
    # -----------------------------------------------------

    def student(
        self,
        student_id: int,
    ) -> sqlite3.Row | None:

        return self.connection.execute(
            """
            SELECT *
            FROM students
            WHERE id = ?
            """,
            (student_id,),
        ).fetchone()

    # -----------------------------------------------------
    # GET STUDENT BY STUDENT ID
    # -----------------------------------------------------

    def student_by_student_id(
        self,
        student_id: str,
    ) -> sqlite3.Row | None:

        return self.connection.execute(
            """
            SELECT *
            FROM students
            WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()

    # -----------------------------------------------------
    # MARK ATTENDANCE
    # -----------------------------------------------------

    def mark_attendance(
        self,
        student_id: int,
    ) -> bool:

        now = datetime.now()

        cursor = self.connection.execute(
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

        self.connection.commit()

        return cursor.rowcount == 1

    # -----------------------------------------------------
    # GET ATTENDANCE
    # -----------------------------------------------------

    def attendance(
        self,
        search: str = "",
        date: str = "",
        limit: int | None = None,
    ) -> list[sqlite3.Row]:

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

        values: list[Any] = []

        # Search
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

        # Date
        if date:

            query += """
                AND a.attended_on = ?
            """

            values.append(date)

        query += """
            ORDER BY a.attended_at DESC
        """

        if limit is not None:

            query += """
                LIMIT ?
            """

            values.append(limit)

        return list(
            self.connection.execute(
                query,
                values,
            )
        )

    # -----------------------------------------------------
    # CLOSE DATABASE
    # -----------------------------------------------------

    def close(self) -> None:

        self.connection.close()
