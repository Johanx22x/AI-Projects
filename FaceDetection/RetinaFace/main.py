from retinaface import RetinaFace
import cv2


def detect(img) -> list:
    """Detects faces in a frame and returns a list of bounding boxes

    Args:
        frame (numpy.ndarray): Image frame to detect faces in
    """
    # Get bounding boxes of faces
    faces = RetinaFace.detect_faces(img)

    # Convert bounding boxes to list
    try:
        faces = [face[1]['facial_area'] for face in faces.items()]
        return faces
    except:
        return []


def main():
    # Load image
    img = cv2.imread("../data/selfie.png")

    # Detect faces
    faces = detect(img)

    # Draw bounding boxes
    for face in faces:
        x1, y1, x2, y2 = face
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()