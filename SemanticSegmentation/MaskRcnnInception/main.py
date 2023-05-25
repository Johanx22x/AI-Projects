import cv2
import numpy as np
import os
from utils import get_detection


def show_webcam(net: cv2.dnn_Net, mirror: bool = False) -> None:
    """Show the webcam feed"""
    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    while ret_val:
        if mirror:
            img = cv2.flip(img, 1)
        img = get_image_without_background(img, net)
        cv2.imshow("Webcam", img)
        ret_val, img = cam.read()
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def get_image_without_background(img: np.ndarray, net: cv2.dnn_Net) -> np.ndarray:
    """Get the image without background
    Args:
        img (np.ndarray): The image
        net (cv2.dnn_Net): The network
    Returns:
        np.ndarray: The image without background
    """
    img_height, img_width, img_channels = img.shape

    # Convert the image to blob format
    blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

    # Get mask
    boxes, masks = get_detection(net, blob)

    # Draw the mask
    empty_img = np.zeros((img_height, img_width, img_channels))

    # Define the threshold
    detection_th = 0.5

    for i in range(len(masks)):
        bbox = boxes[0, 0, i]

        # Get the class id and score
        class_id = bbox[1]
        score = bbox[2]

        # Check if the score is greater than the threshold
        if score < detection_th:
            continue

        # Get the mask
        mask = masks[i]

        # Get the bounding box coordinates
        x1 = int(bbox[3] * img_width)
        y1 = int(bbox[4] * img_height)
        x2 = int(bbox[5] * img_width)
        y2 = int(bbox[6] * img_height)

        # Get the mask
        mask = mask[int(class_id)]

        # Resize the mask
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))

        # Threshold the mask
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        mask = mask * 255

        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask

        # Visualization

        # Remove the background
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (empty_img[:, :, c] / 255)

    return img


def main() -> None:
    """Main function for the Mask R-CNN model"""
    # Define the model paths
    cfg_path = (
        "./models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    )
    weights_path = "./models/mask_rcnn_inception/frozen_inference_graph.pb"
    class_names = "./models/mask_rcnn_inception/class.names"

    # Load the model
    net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

    # Use GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    show_webcam(net)


if __name__ == "__main__":
    main()
