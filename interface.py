import tkinter as tk
import os
from PIL import Image, ImageTk
import subprocess
import threading
import time

# Define a list of program commands and their names
program_commands = [
    ("python LPB.py", "Entrenamiento"),  # Change these to your program commands and names
    ("python distance.py", "Verificacion")
]

current_program_index = -1
process = None

# Function to start or kill a Python program
def toggle_program():
    global process, current_program_index
    if process is not None:
        process.kill()
        process = None
        toggle_button.config(text="Start Program")
        program_name_label.config(text="No program running")
    else:
        current_program_index = (current_program_index + 1) % len(program_commands)
        program_command, program_name = program_commands[current_program_index]
        process = subprocess.Popen(program_command, shell=True)
        toggle_button.config(text="Kill Program")
        program_name_label.config(text=f"Running: {program_name}")

# Function to delete a txt file in the directory
def delete_txt_file():
    file_to_delete = "output.txt"  # Change this to the name of your txt file
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)

# Function to refresh the image
def refresh_image():
    global image, photo, image_label
    # Replace "Frame.jpg" with the name of your image file
    image = Image.open("Frame.jpg")
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create a function to periodically check for changes in the image
def image_refresh_task():
    while True:
        time.sleep(5)  # Check every 5 seconds
        current_image_time = os.path.getmtime("Frame.jpg")
        if current_image_time != image_time:
            refresh_image()
            image_time = current_image_time

# Create the main window
window = tk.Tk()
window.title("Image Viewer and Program Controller")

# Load and display an image
image = Image.open("Frame.jpg")  # Change "Frame.jpg" to your image file
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(window, image=photo)
image_label.pack()

# Create a label to display the currently running program
program_name_label = tk.Label(window, text="No program running", font=("Helvetica", 12))
program_name_label.pack()

# Create a button to toggle the program
toggle_button = tk.Button(window, text="Start Program", command=toggle_program)
toggle_button.pack()

# Create a button to delete a txt file
delete_button = tk.Button(window, text="Delete TXT File", command=delete_txt_file)
delete_button.pack()

# Create a button to refresh the image
refresh_button = tk.Button(window, text="Refresh Image", command=refresh_image)
refresh_button.pack()

# Start the image refresh thread
image_time = os.path.getmtime("Frame.jpg")
image_refresh_thread = threading.Thread(target=image_refresh_task)
image_refresh_thread.daemon = True
image_refresh_thread.start()

# Start the Tkinter main loop
window.mainloop()
