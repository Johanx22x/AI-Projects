''' This file is used to get the path 
of a file selected by the user '''
import tkinter as tk 
from tkinter import filedialog 
import cv2


def get_data_path() -> str:
    ''' This function returns the path of the 
    file selected by the user

    Returns:
        str: The path of the file selected by the user
    '''
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def draw_emotion(frame, box, emotion):
    """This function draws the emotion on the frame

    Args:
        frame (numpy.ndarray): The frame to draw the emotion on
        box (list): The bounding box of the face
        emotion (str): The emotion to draw
    """
    cv2.rectangle(
        frame,
        (box[0], box[1]),
        (box[2], box[3]),
        (255, 0, 255),
        4,
    )
    cv2.putText(
        frame,
        emotion,
        (box[0], box[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        4,
        cv2.LINE_AA,
    )
