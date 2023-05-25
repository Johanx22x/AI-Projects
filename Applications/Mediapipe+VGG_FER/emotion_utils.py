import cv2
import tensorflow as tf
from keras.models import model_from_json
import numpy as np

def load_model(model_path: str) -> tf.keras.Model:
    """Load model from path 
    Args:
        model_path: path to model 
    Returns:
        model: loaded model
    """
    # load json and create model
    model = model_from_json(open(model_path+"/model.json", "r").read())

    # load weights into new model
    model.load_weights(model_path+"/model.h5")

    return model


def get_emotion(model: tf.keras.Model, frame: np.ndarray) -> str:
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
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    # Get corresponding emotion
    emotion = emotions[max_index]

    return emotion
