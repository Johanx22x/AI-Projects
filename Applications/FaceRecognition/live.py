import face_recognition
import cv2
import sqlite3
import numpy as np
from time import perf_counter


def get_database_faces() -> list:
    # Connect to database 
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Get all faces from database 
    c.execute("SELECT * FROM faces")
    faces = c.fetchall()

    # Close connection to database 
    conn.close()

    return faces


def convert_to_dict(faces: list) -> dict:
    face_dict = {}
    for face in faces:
        face_dict[face[2]] = np.frombuffer(face[1], dtype=np.float64)

    return face_dict


def draw_fps(frame, t1_start, t1_stop) -> None:
    cv2.putText(
        frame,
        f"FPS: {int(1/(t1_stop-t1_start))}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def live() -> None:
    # Get faces from database
    faces = convert_to_dict(get_database_faces())

    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Read the first frame
    success, frame = cap.read()

    # While there are frames to be read
    while success:
        # Get the start time of the fps counter
        t1_start = perf_counter()

        # Get the face locations from the frame
        face_locations = face_recognition.face_locations(frame)

        # For each face in the frame
        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the face
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 0, 255),
                2,
            )

            # Get the face encoding of the face 
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]

            # Compare the face encoding to the faces in the database 
            matches = face_recognition.compare_faces(list(faces.values()), face_encoding) 

            # If there is a match 
            if True in matches:
                # Get the index of the match 
                match_index = matches.index(True)

                # Get the name of the match 
                match_name = list(faces.keys())[match_index]

                # Draw the name of the match on the frame
                cv2.putText(
                    frame,
                    match_name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Get the end time of the fps counter
        t1_stop = perf_counter()

        # Draw the FPS on the frame
        draw_fps(frame, t1_start, t1_stop)

        # Show the frame
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read the next frame
        success, frame = cap.read()


def main() -> None:
    live()


if __name__ == "__main__":
    main()
