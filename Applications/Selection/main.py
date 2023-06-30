""" 
This program is used to do a comparative analysis between different face and
emotion recognition algorithms and their performance on different situations.
"""
import GUI.menu as menu
import algorithms.algorithms as alg
import situations.situations as sit
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


def main() -> None:
    """Main function of the program"""
    dataset = "fer"
    emotion_model_path = "algorithms/emotion/models"
    face_algorithm, emotion_algorithm, situation = menu.start()

    # Check if the input is valid
    if face_algorithm not in alg.face_algorithms.keys():
        raise Exception("Invalid face recognition algorithm")
    if emotion_algorithm not in alg.emotion_algorithms.keys():
        raise Exception("Invalid emotion recognition algorithm")
    if situation not in sit.situations.keys():
        raise Exception("Invalid situation")

    # Get the face and emotion recognition algorithms
    face_algorithm = alg.face_algorithms[face_algorithm]
    emotion_algorithm = alg.emotion_algorithms[emotion_algorithm]

    if "kdef" in emotion_algorithm:
        dataset = "kdef"
    elif "ck" in emotion_algorithm:
        dataset = "ck"

    # Load the emotion model
    emotion_model = model_from_json(
        open(emotion_model_path + "/" + emotion_algorithm + ".json", "r").read()
    )
    emotion_model.load_weights(emotion_model_path + "/" + emotion_algorithm + ".h5")

    # Run the situation
    situation = sit.situations[situation](face_algorithm, emotion_model, dataset)


if __name__ == "__main__":
    main()
