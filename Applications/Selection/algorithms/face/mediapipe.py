import mediapipe as mp
import cv2


def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Initialize the face detection module
    mp_face_detection = mp.solutions.face_detection

    # Use the GPU if available
    mp_face_detection.use_gpu = True

    # Initialize the face detection module
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the image dimensions
        img_height, img_width, _ = img.shape
        
        # Process the image
        out = face_detection.process(img_rgb)

        faces = []
        if out.detections:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * img_width)
                y1 = int(y1 * img_height)
                w = int(w * img_width)
                h = int(h * img_height)

                # Adjust the region of interest if it exceeds the image boundaries
                if y1 < 0:
                    h += y1  # Reduce the height by the excess amount
                    y1 = 0  # Set y1 to 0 to start from the top

                if x1 < 0:
                    w += x1  # Reduce the width by the excess amount
                    x1 = 0  # Set x1 to 0 to start from the left

                if y1 + h > img_height:
                    h = (
                        img_height - y1
                    )  # Reduce the height if it exceeds the image height

                if x1 + w > img_width:
                    w = img_width - x1  # Reduce the width if it exceeds the image width

                faces.append([x1, y1, x1 + w, y1 + h])
    return faces
