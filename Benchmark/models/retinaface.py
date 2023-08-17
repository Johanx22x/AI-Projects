from retinaface import RetinaFace
import cv2


def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Get bounding boxes of faces
    faces = RetinaFace.detect_faces(img)

    # Convert bounding boxes to list
    try:
        faces = [face[1]['facial_area'] for face in faces.items()]
        return faces
    except:
        return []
