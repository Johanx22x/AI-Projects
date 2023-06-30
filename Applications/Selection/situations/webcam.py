import cv2
from time import perf_counter
from algorithms.emotion.emotion import get_emotion
from utils.utils import draw_emotion


def draw_fps(frame, t1_start, t1_stop):
    """This function draws the FPS on the frame

    Args:
        frame (numpy.ndarray): The frame to draw the FPS on
        t1_start (float): The start time of the frame
        t1_stop (float): The stop time of the frame
    """
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


def start(face_algorithm, emotion_model, dataset) -> None:
    """This function starts the webcam situation

    Args:
        face_algorithm (FaceAlgorithm): The face recognition algorithm
        emotion_algorithm (EmotionAlgorithm): The emotion recognition algorithm
    """
    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Read the first frame
    success, frame = cap.read()

    # While there are frames to be read
    while success:
        # Get the start time of the fps counter
        t1_start = perf_counter()

        # Detect the faces in the frame
        faces = face_algorithm.detect(frame)

        # For each face detected
        for face in faces:
            # Detect the emotion of the face
            cropped_face = frame[
                    face[1] : face[3], face[0] : face[2]
                ]
            emotion = get_emotion(emotion_model, cropped_face, dataset)

            # Draw the emotion on the frame
            draw_emotion(frame, face, emotion)

        # Get the end time of the fps counter
        t1_stop = perf_counter()

        # Draw the FPS on the frame
        draw_fps(frame, t1_start, t1_stop)

        # Show the frame
        cv2.imshow("frame", frame)

        # If the user presses 'q', break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Read the next frame
        success, frame = cap.read()
