from utils.utils import get_data_path
import cv2
from algorithms.emotion.emotion import get_emotion
from utils.utils import draw_emotion

def start(face_algorithm, emotion_model, dataset) -> None:
    ''' This function starts the video situation

    Args:
        face_algorithm (FaceAlgorithm): The face recognition algorithm
        emotion_algorithm (EmotionAlgorithm): The emotion recognition algorithm
    '''
    video_path = get_data_path()
    
    # Load the video 
    video = cv2.VideoCapture(video_path)

    # Read the first frame 
    success, frame = video.read()

    # While there are frames to be read 
    while success:
        # Detect the faces in the frame
        faces = face_algorithm.detect(frame)

        # For each face detected
        for face in faces:
            # Detect the emotion of the face
            cropped_face = frame[
                face[1] : face[1] + face[3], face[0] : face[0] + face[2]
            ]
            emotion = get_emotion(emotion_model, cropped_face, dataset)

            # Draw the emotion on the frame
            draw_emotion(frame, face, emotion)

        # Show the frame 
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == ord("q"):
            break

        # Read the next frame
        success, frame = video.read()

    # Release the video 
    video.release()

    # Close all windows 
    cv2.destroyAllWindows()
