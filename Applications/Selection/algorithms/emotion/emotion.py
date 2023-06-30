import cv2 
import numpy as np 
import tensorflow as tf


def get_emotion(model: tf.keras.Model, frame: np.ndarray, dataset: str) -> str:
    """Get emotion from frame 
    Args:
        model: emotion detection model
        frame: frame to be processed 
    Returns:
        emotion: emotion detected
    """
    # Resize the frame to 48x48
    frame = cv2.resize(frame,(48,48))

    # Convert the captured frame into Gray scale
    gray_frame = frame
    if dataset == "fer":
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the 2D image to 1D array
    image_pixels = tf.keras.preprocessing.image.img_to_array(gray_frame) 

    # Reshaping the image to support our model input
    image_pixels = np.expand_dims(image_pixels, axis = 0) 

    # Normalize the image
    image_pixels /= 255

    # Predicting the emotion
    predictions = model.predict(image_pixels)

    # Find max indexed array
    max_index = np.argmax(predictions[0])

    # Emotions list
    emotions = ('contempt', 'angry', 'neutral', 'fear', 'sadness', 'surprise', 'disgust', 'happiness')
    if dataset == "fer":
        emotions = ('angry', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
    elif dataset == "ck":
        emotions = ('sadness', 'angry', 'happiness', 'contempt', 'disgust', 'fear', 'surprise')

    # Get corresponding emotion
    emotion = emotions[max_index]

    return emotion
