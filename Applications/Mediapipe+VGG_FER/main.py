import argparse
import cv2
import numpy as np
import mediapipe as mp
from emotion_utils import *


def init() -> tuple:
    """Initialize mediapipe face detection model and parse arguments 
    Returns:
        mp_face_detection: face detection model 
        args: parsed arguments 
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--mode", type=str, default="webcam", help="mode to run the script"
    )
    args.add_argument(
        "--file_path", type=str, default="./data/people.jpg", help="path to image"
    )
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection, args.parse_args()


def show_img(img: np.ndarray) -> None:
    """Show image in a window
    Args:
        img: image to be shown
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_webcam(face_detection: mp.solutions.face_detection.FaceDetection,
                emotion_model: tf.keras.Model
                ) -> None:
    """Show webcam in a window
    Args:
        face_detection: face detection model
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        frame = get_prediction(frame, face_detection, emotion_model)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def show_video(
    face_detection: mp.solutions.face_detection.FaceDetection,
    args: argparse.ArgumentParser,
    emotion_model: tf.keras.Model
) -> None:
    """Show video in a window
    Args:
        face_detection: face detection model
    """
    cap = cv2.VideoCapture(args.file_path)
    ret, frame = cap.read()
    while ret:
        frame = get_prediction(frame, face_detection, emotion_model)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def get_prediction(
    img: np.ndarray, face_detection: mp.solutions.face_detection.FaceDetection, emotion_model: tf.keras.Model
) -> np.ndarray:
    """Blur faces in an image
    Args:
        img: image to be processed
        face_detection: face detection model
    Returns:
        img: image with blurred faces
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * img_width)
            y1 = int(y1 * img_height)
            w = int(w * img_width)
            h = int(h * img_height)

            # Adjust the region of interest if it exceeds the image boundaries
            if y1 < 0:
                h += y1  # Reduce the height by the excess amount
                y1 = 0  # Set y1 to 0 to start from the top

            if x1 < 0:
                w += x1  # Reduce the width by the excess amount
                x1 = 0  # Set x1 to 0 to start from the left

            if y1 + h > img_height:
                h = img_height - y1  # Reduce the height if it exceeds the image height

            if x1 + w > img_width:
                w = img_width - x1  # Reduce the width if it exceeds the image width

            # Cropped image
            cropped_img = img[y1 : y1 + h, x1 : x1 + w]

            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

            # Get the emotion 
            emotion = get_emotion(emotion_model, cropped_img)

            img = cv2.putText(
                img,
                emotion,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return img


def main() -> None:
    """Main function"""
    # Initialize mediapipe face detection model
    mp_face_detection, args = init()
    mp_face_detection.use_gpu = True

    # Load emotion model 
    emotion_model_path = "./models/VGG_FER/"
    emotion_model = load_model(emotion_model_path)

    # Run the script
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        if args.mode in ["webcam"]:
            show_webcam(face_detection, emotion_model)
        elif args.mode in ["video"]:
            show_video(face_detection, args, emotion_model)
        elif args.mode in ["image"]:
            img = cv2.imread(args.file_path)
            img = get_prediction(img, face_detection, emotion_model)
            show_img(img)


if __name__ == "__main__":
    main()
