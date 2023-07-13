import face_recognition
import tkinter as tk
import cv2
import sqlite3
import numpy as np
import os
from tkinter import simpledialog


class Webcam:
    """Webcam class
    Implemented with OpenCV
    """
    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)

    def read_frame(self) -> np.ndarray:
        """Read a frame from the webcam"""
        _, frame = self.video_capture.read()
        self.video_capture.release()
        return frame


class Database:
    """Database class to handle the registered faces
    Using SQLite
    """

    _instance = None

    def __new__(self):
        # Create a database if it does not exist
        if not os.path.exists("database.db"):
            conn = sqlite3.connect("database.db")
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE faces(
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    face_encoding BLOB NOT NULL,
                                    name TEXT
                                )"""
            )
            conn.commit()
            conn.close()

        if self._instance is None:
            self._instance = super(Database, self).__new__(self)

            # SQLite initialization
            self._instance.conn = sqlite3.connect("database.db")
            self._instance.cursor = self._instance.conn.cursor()

        return self._instance

    def __del__(self):
        """Close the database connection"""
        self.conn.close()

    def register(self, face_encoding, name=""):
        """Register a face in the database"""
        self.cursor.execute(
            "INSERT INTO faces(face_encoding, name) VALUES(?, ?)", (face_encoding, name)
        )
        self.conn.commit()

    def get_face_encodings(self):
        """Get all the registered face encodings"""
        self.cursor.execute("SELECT face_encoding FROM faces")
        return self.cursor.fetchall()

    def get_face_name(self, face_encoding):
        """Get the name of the face"""
        self.cursor.execute("SELECT name FROM faces WHERE face_encoding = ?", (face_encoding,))
        return self.cursor.fetchone()[0]

    def check_if_face_registered(self, face_encoding):
        """Check if a face is registered"""
        self.cursor.execute("SELECT face_encoding FROM faces")
        face_encodings_from_database = self.cursor.fetchall()

        for face_encoding_from_database in face_encodings_from_database:
            # Convert the face encoding from database to numpy array
            face_encoding_from_database = np.frombuffer(
                face_encoding_from_database[0], dtype=np.float64
            )

            if True in face_recognition.compare_faces(
                    [face_encoding], face_encoding_from_database
            ):
                return self.get_face_name(face_encoding_from_database)

        return None


class FaceRecognition:
    """Face Recognition class"""

    database = Database()

    def __init__(self) -> None:
        # Create a window
        self.window = tk.Tk()
        self.window.title("Face Recognition")
        self.window.geometry("500x250")
        self.window.resizable(False, False)

        # Background color
        self.window.config(background="#ffffff")

        # Label
        self.label = tk.Label(self.window, text="Face Recognition")
        self.label.config(font=("Fira Code", 28), bg="#ffffff", fg="#000000")
        self.label.pack(padx=50, pady=20)

        # Login button
        self.login_button = tk.Button(self.window, text="Login", width=10, height=2)
        self.login_button.config(font=("Fira Code", 14), command=self.login)
        self.login_button.pack(side=tk.LEFT, padx=50, pady=50)

        # Register button
        self.register_button = tk.Button(
            self.window, text="Register", width=10, height=2
        )
        self.register_button.config(font=("Fira Code", 14), command=self.register)
        self.register_button.pack(side=tk.RIGHT, padx=50, pady=50)

    def run(self) -> None:
        """Run the application"""
        self.window.mainloop()

    def login(self) -> None:
        """Login
        1. Capture a frame from the webcam
        3. Compare the detected faces with the registered faces
        4. If the detected face is registered, login
        """
        webcam = Webcam()

        # Capture a frame from the webcam
        frame = webcam.read_frame()

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)

        # If there is no face, raise a popup window
        if len(face_locations) == 0:
            self.raise_popup_window("No face detected")
            return

        # Get the face encodings
        face_encodings = face_recognition.face_encodings(frame)

        # Check if the face is registered 
        name = self.database.check_if_face_registered(face_encodings[0])
        if name is not None:
            self.raise_popup_window("Login successful " + name)
            print("Login successful", name)
            return

        # If the face is not registered, raise a popup window 
        self.raise_popup_window("Face not registered")

    def register(self) -> None:
        """Register
        1. Capture a frame from the webcam
        3. If there is more than one face, raise a popup window
        4. If there is no face, raise a popup window
        5. If the face is registered, raise a popup window
        6. If there is only one face, register the face
        """
        webcam = Webcam()

        # Capture a frame from the webcam
        frame = webcam.read_frame()

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)

        # If there is no face, raise a popup window
        if len(face_locations) == 0:
            self.raise_popup_window("No face detected")
            return

        # If there is more than one face, raise a popup window
        if len(face_locations) > 1:
            self.raise_popup_window("More than one face detected")
            return

        # Get the face encodings
        face_encodings = face_recognition.face_encodings(frame)

        # Check if the face is registered
        if self.database.check_if_face_registered(face_encodings[0]) is not None:
            self.raise_popup_window("Face already registered")
            return

        # Get the name of the person
        name = simpledialog.askstring("Name", "Enter your name")

        # Register the face
        self.database.register(face_encodings[0], name)
        self.raise_popup_window("Face registered")

    def raise_popup_window(self, message) -> None:
        # Create a popup window
        self.popup_window = tk.Toplevel()
        self.popup_window.title("Alert")
        self.popup_window.geometry("300x200")
        self.popup_window.resizable(False, False)

        # Background color
        self.popup_window.config(background="#ffffff")

        # Label
        self.label = tk.Label(self.popup_window, text=message)
        self.label.pack(padx=50, pady=20)

        # Button
        self.button = tk.Button(
            self.popup_window, text="OK", command=self.popup_window.destroy
        )
        self.button.pack(padx=50, pady=20)


def main():
    face_recognition = FaceRecognition()  # Create an instance of FaceRecognition
    face_recognition.run()


if __name__ == "__main__":
    main()
