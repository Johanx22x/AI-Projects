from mtcnn import MTCNN
import cv2


def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    detector = MTCNN()
    faces = detector.detect_faces(img)
    faces = [(x, y, w+x, h+y) for (x, y, w, h) in [face['box'] for face in faces]]
    return faces
