from yoloface import face_analysis
import numpy
import cv2

confidence = 0.5

face=face_analysis()        #  Auto Download a large weight files from Google Drive.
                            #  only first time.
                            #  Automatically  create folder .yoloface on cwd.

img, box, conf = face.face_detection(image_path='../data/people.jpg',model='tiny')

# Based on the previous comment, draw a rectangle on the faces
for i in range(len(box)):
    if conf[i] < confidence:
        continue
    x, y, w, h = box[i][0], box[i][1], box[i][2], box[i][3]
    cv2.rectangle(img, (x, y), (x+h, y+w), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
