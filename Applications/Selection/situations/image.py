from utils.utils import get_data_path
import cv2
from algorithms.emotion.emotion import get_emotion
from utils.utils import draw_emotion


def start(face_algorithm, emotion_model, dataset) -> None:
    ''' This function starts the image situation 

    Args:
        face_algorithm (FaceAlgorithm): The face recognition algorithm
        emotion_algorithm (EmotionAlgorithm): The emotion recognition algorithm
    '''
    # Get the path of the image
    image_path = get_data_path()

    # Load the image 
    image = cv2.imread(image_path) 

    # Detect the faces in the image 
    faces = face_algorithm.detect(image)

    # For each face detected
    for face in faces:
        # Detect the emotion of the face
        cropped_face = image[
            face[1] : face[1] + face[3], face[0] : face[0] + face[2]
        ]
        emotion = get_emotion(emotion_model, cropped_face, dataset)

        # Draw the emotion on the frame
        draw_emotion(image, face, emotion)

    # Show the image 
    cv2.imshow("Image", image)

    # Wait for a key press to exit 
    cv2.waitKey(0)

    # Close all windows 
    cv2.destroyAllWindows()
