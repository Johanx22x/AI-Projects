from yoloface import face_analysis
import numpy
import cv2


def detect(img) -> list:
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
