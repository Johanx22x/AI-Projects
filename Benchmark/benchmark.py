import cv2
import time
import csv
import os

from models import retinaface, mediapipe, yolo, mtcnn, haarcascade, run_yunet 
from models.yunet import YuNet


def dump_data(data):
    """Dump data to csv file"""
    with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'model', 'time', 'num_faces'])
        for img in data:
            for model in data[img]:
                writer.writerow([img, model, data[img][model]['time'], data[img][model]['num_faces']])


def main():
    data = {}
    models = {
            "retinaface": retinaface,
            "mediapipe": mediapipe,
            "yolo": yolo,
            "mtcnn": mtcnn,
            "haarcascade": haarcascade,
            "yunet": run_yunet
            }

    yunet_model = YuNet(modelPath="models/face_detection_yunet_2023mar.onnx",
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                  targetId=cv2.dnn.DNN_TARGET_CPU)

    images_dir = "images/"
    images_subdir = os.listdir(images_dir)
    for subdir in images_subdir:
        images = os.listdir(images_dir + subdir)
        for img_name in images:
            img = cv2.imread(os.path.join(images_dir, subdir, img_name))
            data[img_name] = {}
            for model in models:
                start = time.time()
                if model == "yunet":
                    results = models[model].detect(img, yunet_model)
                else:
                    results = models[model].detect(img)
                end = time.time()
                data[img_name][model] = {}
                data[img_name][model]['time'] = end - start
                data[img_name][model]['results'] = results
                data[img_name][model]['num_faces'] = len(results)

    dump_data(data)


if __name__ == '__main__':
    main()
