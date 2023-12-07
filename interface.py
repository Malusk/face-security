import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk
import subprocess
import threading
import time

# Define a list of program commands and their names
class CustomWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Configure window
        self.title("Sistema de Videovigilancia")
        self.geometry("400x300")
        self.configure(bg="#2E2E2E")  # Set background color

        # Configure title bar
        self.title_frame = ttk.Frame(self, style="Custom.TFrame")
        self.title_frame.pack(side="top", fill="x")

        # Configure content frame
        self.content_frame = ttk.Frame(self, style="Custom.TFrame")
        self.content_frame.pack(side="top", fill="both", expand=True)

        # Add content to the window
        label = ttk.Label(self.content_frame, text="Sistema de Videovigilancia", style="Custom.TLabel")
        label.pack(padx=20, pady=20)

        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background="#2E2E2E", borderwidth=2, relief="solid")
        self.style.configure("Custom.TLabel", background="#2E2E2E", foreground="white", font=("Helvetica", 20, "bold"))
        self.style.configure("Custom.TButton", background="#2E2E2E", foreground="#2E2E2E", borderwidth=20)

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
    image = image.resize((image.width // 2, image.height // 2))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create a function to periodically check for changes in the image
def image_refresh_task():
    while True:
        time.sleep(1)  # Check every 5 seconds
        current_image_time = os.path.getmtime("Frame.jpg")
        if current_image_time != image_time:
            refresh_image()
            image_time = current_image_time

# Create the main window
#window = tk.Tk()
custom_window = CustomWindow()
#custom_window.title("Image Viewer and Program Controller")

# Load and display an image
image = Image.open("Frame.jpg")  # hange "Frame.jpg" to your image file
#image = image.convert('RGB')
image = image.resize((image.width // 2, image.height // 2))
photo = ImageTk.PhotoImage(image)
image_label = ttk.Label(custom_window, image=photo)
image_label.pack()

# Create a label to display the currently running program
program_name_label = ttk.Label(custom_window, text="No program running", style="Custom.TLabel")
program_name_label.pack(pady=5)

# Create a button to toggle the program
toggle_button = ttk.Button(custom_window, text="Start Program", command=toggle_program, style="Custom.TButton")
toggle_button.pack(pady=5)

# Create a button to delete a txt file
delete_button = ttk.Button(custom_window, text="Delete TXT File", command=delete_txt_file, style="Custom.TButton")
delete_button.pack(pady=5)

# Create a button to refresh the image
refresh_button = ttk.Button(custom_window, text="Refresh Image", command=refresh_image, style="Custom.TButton")
refresh_button.pack(pady=5)

# Start the image refresh thread
image_time = os.path.getmtime("Frame.jpg")
image_refresh_thread = threading.Thread(target=image_refresh_task)
image_refresh_thread.daemon = True
image_refresh_thread.start()

# Start the Tkinter main loop
#custom_window.pack(fill=tk.BOTH, expand=True)
custom_window.mainloop()
