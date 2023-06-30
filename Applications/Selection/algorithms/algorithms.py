''' 
This file contains the algorithms 
that will be used to compare the 
performance of different face and 
emotion recognition algorithms.
'''
import algorithms.face.mediapipe as mediapipe
import algorithms.face.haarcascade as haar_cascade
import algorithms.face.dlib as dlib 
import algorithms.face.yolo as yolo
import algorithms.face.mtcnn as mtcnn 
import algorithms.face.retinaface as retinaface 

face_algorithms = {
        'Mediapipe': mediapipe,
        'Haar Cascade': haar_cascade,
        'Dlib': dlib,
        'MTCNN': mtcnn,
        'RetinaFace': retinaface,
        'Yolo': yolo 
    }

emotion_algorithms = {
        'VGG_FER': 'vgg_fer',
        'RESNET_FER': 'resnet_fer',
        'VGG_CK': 'vgg_ck',
        'RESNET_CK': 'resnet_ck',
        'VGG_CK_FER_KDEF': 'vgg_ck_fer_kdef',
        'RESNET_CK_FER_KDEF': 'resnet_ck_fer_kdef'
    }
