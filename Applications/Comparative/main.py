import tkinter as tk 
from tkinter import filedialog 
import cv2
import mediapipe as mp
import dlib
from mtcnn import MTCNN
from retinaface import RetinaFace
from yoloface import face_analysis
import numpy
from PIL import Image


def get_data_path() -> str:
    ''' This function returns the path of the 
    file selected by the user

    Returns:
        str: The path of the file selected by the user
    '''
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def yolo(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    confidence = 0.5

    face=face_analysis()        #  Auto Download a large weight files from Google Drive.
                                #  only first time.
                                #  Automatically  create folder .yoloface on cwd.

    _, box, conf = face.face_detection(frame_arr=img,frame_status=True,model='tiny')

    # Based on the previous comment, draw a rectangle on the faces
    faces = []
    for i in range(len(box)):
        if conf[i] < confidence:
            continue
        faces.append((box[i][0], box[i][1], box[i][3] + box[i][0], box[i][2] + box[i][1]))

    return faces


def retinaface(img) -> list:
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


def mtcnn(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    detector = MTCNN()
    faces = detector.detect_faces(img)
    faces = [(x, y, w+x, h+y) for (x, y, w, h) in [face['box'] for face in faces]]
    return faces


def mediapipe(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Initialize the face detection module
    mp_face_detection = mp.solutions.face_detection

    # Use the GPU if available
    mp_face_detection.use_gpu = True

    # Initialize the face detection module
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        # Convert the image to RGB
        img_rgb = img

        # Get the image dimensions
        img_height, img_width, _ = img.shape
        
        # Process the image
        out = face_detection.process(img_rgb)

        faces = []
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
                    h = (
                        img_height - y1
                    )  # Reduce the height if it exceeds the image height

                if x1 + w > img_width:
                    w = img_width - x1  # Reduce the width if it exceeds the image width

                faces.append([x1, y1, x1 + w, y1 + h])
    return faces


def dlib(img) -> list:
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


def haarcascade(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Load the Haar Cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    # Convert the image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the bounding boxes to a list of tuples
    faces = [ (x, y, x + w, y + h) for (x, y, w, h) in faces ]

    return faces


def main() -> None:
    ''' This function is the main function of the program

    This program reads an image from the user and detects faces in it 
    using different face detection algorithms. It then draws a rectangle 
    around the detected faces and displays a grid with the 6 resulting images.
    '''
    # Get the path of the file selected by the user
    file_path = get_data_path()
    # Read the image
    img = cv2.imread(file_path)

    # Convert the image to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the bounding boxes of the faces 
    faces = {
        "Haar Cascade": haarcascade(img),
        # "Dlib": dlib(img),
        "MediaPipe": mediapipe(img),
        "MTCNN": mtcnn(img),
        "RetinaFace": retinaface(img),
        "YOLO": yolo(img)
    }

    # Draw a rectangle around the detected faces in a copy of the image
    images = []
    images.append(img)
    for name, face in faces.items():
        image = img.copy()
        cv2.putText(image, name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
        for x1, y1, x2, y2 in face:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        images.append(image)

    cv2.putText(img, "Original", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

    # Display the images in a grid
    grid = Image.new("RGB", (3 * img.shape[1], 2 * img.shape[0]), "white")

    for i in range(6):
        grid.paste(Image.fromarray(images[i]), (i % 3 * img.shape[1], i // 3 * img.shape[0]))

    # Change the color from BGR to RGB 
    grid = grid.convert("RGB")

    grid.show()


if __name__ == '__main__':
    main()
