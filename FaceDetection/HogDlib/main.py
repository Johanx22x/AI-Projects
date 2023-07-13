import dlib
import cv2

# Load the pre-trained face detector from Dlib
detector = dlib.get_frontal_face_detector()

# Load the input image
image = cv2.imread("../data/people.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    # Extract the bounding box coordinates for the face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Draw a rectangle around the face on the original image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
