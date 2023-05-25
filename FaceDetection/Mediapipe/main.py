import cv2
import mediapipe as mp

# Load mediapipe face detection model
mp_face_detection = mp.solutions.face_detection

# Load image 
img = cv2.imread("../data/people.jpg")

with mp_face_detection.FaceDetection( model_selection=1, min_detection_confidence=0.5) as face_detection:

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape
    out = face_detection.process(img_rgb)

    if out.detections:
        # Draw the face detection annotations on the image.
        for detection in out.detections:
            # Get the coordinates of the bounding box
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Convert relative coordinates to absolute coordinates
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1, y1, w, h = int(x1 * img_width), int(y1 * img_height), int(w * img_width), int(h * img_height)

            # Draw bounding box
            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
