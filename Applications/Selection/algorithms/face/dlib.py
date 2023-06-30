import dlib
import cv2

def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Load the pre-trained face detector from Dlib
    detector = dlib.get_frontal_face_detector()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector(gray)

    # Convert the resulting Dlib rectangle objects to bounding boxes
    faces = [(face.left(), face.top(), face.left() + face.width(), face.top() + face.height()) for face in faces]

    # Return the bounding boxes
    return faces
