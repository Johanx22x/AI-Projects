''' 
This file contains the situations 
that will be used to compare the
performance of different face and
emotion recognition algorithms.
'''
from situations.webcam import start as webcam
from situations.image import start as image 
from situations.video import start as video

situations = {
        'Webcam': webcam,
        'Image': image,
        'Video': video
    }

