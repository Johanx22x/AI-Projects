import cv2

# Load the Haar Cascade XML file for face detection
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

# Load the input image
image = cv2.imread("../data/people.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over the detected faces
for (x, y, w, h) in faces:
    # Draw a rectangle around each face on the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

