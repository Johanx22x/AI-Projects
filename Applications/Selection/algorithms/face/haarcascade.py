import cv2


def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Load the Haar Cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier("./algorithms/face/haarcascade_frontalface_alt.xml")

    # Convert the image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the bounding boxes to a list of tuples
    faces = [ (x, y, x + w, y + h) for (x, y, w, h) in faces ]

    return faces
