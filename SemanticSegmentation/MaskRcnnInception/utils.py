import cv2
import numpy as np


def get_detection(net: cv2.dnn_Net, blob: np.ndarray) -> tuple:
    """ Get the detection from the frame using the net 
    Args:
        net (cv2.dnn_Net): The net to use for the detection 
        blob (np.ndarray): The blob to use for the detection
    Returns:
        tuple: The boxes and masks from the detection
    """
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks
