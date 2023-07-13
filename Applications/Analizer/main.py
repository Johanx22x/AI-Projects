import os
import cv2
import dlib
import numpy as np
from yoloface import face_analysis
from retinaface import RetinaFace
import mediapipe as mp
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import model_from_json


class Analysis:
    n_faces = 0
    bbox_faces = []
    image_name = ''
    model_name = ''
    vgg_emotions = []

    def __init__(self, image_name, model_name):
        self.image_name = image_name
        self.model_name = model_name

    def set_n_faces(self, n_faces):
        self.n_faces = n_faces

    def get_n_faces(self):
        return self.n_faces

    def set_bbox_faces(self, bbox_faces):
        self.bbox_faces = bbox_faces

    def get_bbox_faces(self):
        return self.bbox_faces

    def set_vgg_emotions(self, vgg_emotions):
        self.vgg_emotions = vgg_emotions

    def get_vgg_emotions(self): 
        return self.vgg_emotions


def detect_dlib(img) -> list:
    # Load the pre-trained face detector from Dlib
    detector = dlib.get_frontal_face_detector()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = detector(gray)

    # Convert the resulting Dlib rectangle objects to bounding boxes
    faces = [(face.left(), face.top(), face.left() + face.width(), face.top() + face.height()) for face in faces]

    # Return the bounding boxes
    return len(faces), faces


def detect_haar_cascade(img) -> list:
    # Load the Haar Cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    # Convert the image to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the bounding boxes to a list of tuples
    faces = [ (x, y, x + w, y + h) for (x, y, w, h) in faces ]

    return len(faces), faces


def detect_mediapipe(img) -> list:
    # Initialize the face detection module
    mp_face_detection = mp.solutions.face_detection

    # Use the GPU if available
    mp_face_detection.use_gpu = True

    # Initialize the face detection module
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    return len(faces), faces


def detect_mtcnn(img) -> list:
    detector = MTCNN()
    faces = detector.detect_faces(img)
    faces = [(x, y, w+x, h+y) for (x, y, w, h) in [face['box'] for face in faces]]
    return len(faces), faces


def detect_retinaface(img) -> list:
    # Get bounding boxes of faces
    faces = RetinaFace.detect_faces(img)

    # Convert bounding boxes to list
    try:
        faces = [face[1]['facial_area'] for face in faces.items()]
        return len(faces), faces
    except:
        return 0, []


def detect_yolo(img) -> list:
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

    return len(faces), faces


# def get_emotion(model: tf.keras.Model, frame: np.ndarray, dataset: str) -> str:
#     """Get emotion from frame 
#     Args:
#         model: emotion detection model
#         frame: frame to be processed 
#     Returns:
#         emotion: emotion detected
#     """
#     # Resize the frame to 48x48
#     frame = cv2.resize(frame,(48,48))

#     # Convert the captured frame into Gray scale
#     gray_frame = frame
#     if dataset == "fer":
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert the 2D image to 1D array
#     image_pixels = tf.keras.preprocessing.image.img_to_array(gray_frame) 

#     # Reshaping the image to support our model input
#     image_pixels = np.expand_dims(image_pixels, axis = 0) 

#     # Normalize the image
#     image_pixels /= 255

#     # Predicting the emotion
#     predictions = model.predict(image_pixels)

#     # Find max indexed array
#     max_index = np.argmax(predictions[0])

#     # Emotions list
#     emotions = ('contempt', 'angry', 'neutral', 'fear', 'sadness', 'surprise', 'disgust', 'happiness')
#     if dataset == "fer":
#         emotions = ('angry', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
#     elif dataset == "ck":
#         emotions = ('sadness', 'angry', 'happiness', 'contempt', 'disgust', 'fear', 'surprise')

#     # Get corresponding emotion
#     emotion = emotions[max_index]

#     return emotion


# def get_emotions_vgg(img, faces):
#     # Load the emotion model
#     emotion_model = model_from_json(
#         open("./vgg_fer.json", "r").read()
#     )
#     emotion_model.load_weights('./vgg_fer.h5')
#     return [get_emotion(emotion_model, img[y1:y2, x1:x2], "fer") for (x1, y1, x2, y2) in faces]


def draw_faces(img, faces):
    # Draw a rectangle around each face
    for face in faces:
        cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (255, 0, 255), 4)


def create_folder(model_name):
    # Create folder
    if not os.path.exists('data/' + model_name):
        os.makedirs('data/' + model_name)


def create_csv_file(model_name):
    # Create csv file
    csv_file = open('data/' + model_name + '/results.csv', 'w')
    csv_file.write('image_name,n_faces,bbox_faces\n')
    # csv_file.write('image_name,n_faces,bbox_faces,emotion_vgg_fer\n')
    csv_file.close()


def write_csv_file(model_name, analysis):
    # Write csv file
    csv_file = open('data/' + model_name + '/results.csv', 'a')
    csv_file.write(analysis.image_name + ',' + str(analysis.n_faces) + ',' + str(analysis.bbox_faces) + '\n')
    # csv_file.write(analysis.image_name + ',' + str(analysis.n_faces) + ',' + str(analysis.bbox_faces) + ',' + str(analysis.vgg_emotions) + '\n')
    csv_file.close()


def write_image(model_name, analysis, img):
    # Write image
    cv2.imwrite('data/' + model_name + '/' + analysis.image_name, img)


def main():
    models = {
        'Dlib': detect_dlib,
        'Haar Cascade': detect_haar_cascade,
        'MediaPipe': detect_mediapipe,
        'MTCNN': detect_mtcnn,
        'RetinaFace': detect_retinaface,
        'YOLO': detect_yolo
    }

    for model_name, model in models.items():
        create_folder(model_name)
        create_csv_file(model_name)

        for image_name in os.listdir('images'):
            print('Processing image ' + image_name + ' with ' + model_name)

            # Read image
            img = cv2.imread('images/' + image_name)

            # Detect faces
            n_faces, bbox_faces = model(img)

            # for bbox in bbox_faces:
            #     bbox = list(bbox)
            #     # Adjust the region of interest if it exceeds the image boundaries
            #     bbox[0] = max(bbox[0], 0)
            #     bbox[1] = max(bbox[1], 0)
            #     bbox[2] = min(bbox[2], img.shape[1])
            #     bbox[3] = min(bbox[3], img.shape[0])

            # Create analysis object
            analysis = Analysis(image_name, model_name)
            analysis.set_n_faces(n_faces)
            analysis.set_bbox_faces(bbox_faces)

            # # Get Emotions
            # emotions = get_emotions_vgg(img, bbox_faces)

            # # Add emotions to analysis 
            # analysis.set_vgg_emotions(emotions)

            # Write csv file
            write_csv_file(model_name, analysis)

            # Draw faces 
            draw_faces(img, bbox_faces)

            # Write image 
            write_image(model_name, analysis, img)


if __name__ == '__main__':
    main()
