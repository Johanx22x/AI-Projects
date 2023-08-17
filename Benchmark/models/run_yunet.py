import numpy as np
import cv2 as cv

def detect(image, model):
    h, w, _ = image.shape
    model.setInputSize([w, h])
    result = model.infer(image)
    if result is None:
        return []
    return result
