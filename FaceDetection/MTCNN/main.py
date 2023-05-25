from mtcnn import MTCNN
import cv2

# Load the MTCNN face detection model
detector = MTCNN()

# Load the input image
image = cv2.imread("../data/people.jpg")

# Perform face detection
faces = detector.detect_faces(image)

# Iterate over detected faces
for face in faces:
    # Extract the bounding box coordinates for the face
    x, y, w, h = face['box']

    # Draw a rectangle around the face on the original image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

