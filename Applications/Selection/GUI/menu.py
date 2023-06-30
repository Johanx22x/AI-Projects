''' 
This file is used to implement 
the main menu of the program,
where the user can choose 
between different algorithms
'''
import tkinter as tk
import algorithms.algorithms as alg
import situations.situations as sit

def start() -> list:
    ''' This function is used to
    display the main menu of the
    program and return the choice
    of the user

    Returns:
        list: The choice of the user
    '''
    root = tk.Tk()
    root.title("Face and Emotion Recognition")
    root.geometry("500x550")
    root.resizable(False, False)

    # Create the main menu of the program
    tk.Label(root, text="Face and Emotion Recognition", font=("Ubuntu", 20)).pack(pady=10)

    face_algorithms = alg.face_algorithms 
    emotion_algorithms = alg.emotion_algorithms

    # Create a dropdown menu for the face recognition algorithms 
    tk.Label(root, text="Face Recognition", font=("Ubuntu", 15)).pack(pady=10)
    face_var = tk.StringVar(root)
    face_var.set("Choose an algorithm")
    face_dropdown = tk.OptionMenu(root, face_var, *face_algorithms.keys())
    face_dropdown.pack(pady=10)

    # Create a dropdown menu for the emotion recognition algorithms 
    tk.Label(root, text="Emotion Recognition", font=("Ubuntu", 15)).pack(pady=10) 
    emotion_var = tk.StringVar(root) 
    emotion_var.set("Choose an algorithm") 
    emotion_dropdown = tk.OptionMenu(root, emotion_var, *emotion_algorithms.keys()) 
    emotion_dropdown.pack(pady=10) 

    situations = sit.situations

    # Create a dropdown menu for the situation
    tk.Label(root, text="Situation", font=("Ubuntu", 15)).pack(pady=10)
    situation_var = tk.StringVar(root)
    situation_var.set("Choose a situation")
    situation_dropdown = tk.OptionMenu(root, situation_var, *situations)
    situation_dropdown.pack(pady=10)

    # Create a button to start the program
    tk.Button(root, text="Start", font=("Ubuntu", 15), command=root.destroy).pack(pady=10)

    # Create a button to exit the program
    tk.Button(root, text="Exit", font=("Ubuntu", 15), command=lambda: exit()).pack(pady=10)

    # X button of the window 
    root.protocol("WM_DELETE_WINDOW", exit)

    # Start the main loop of the program
    root.mainloop()

    # Choice is the face algorithm, emotion algorithm and situation chosen by the user 
    choice = [face_var.get(), emotion_var.get(), situation_var.get()]
    return choice
