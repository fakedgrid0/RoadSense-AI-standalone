import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
from PIL import Image, ImageTk


class GUI:
    def __init__(self, backend):
        self.backend = backend

        self.root = tk.Tk()
        self.root.title("RoadSense AI")

        # Set the width and height of the main window
        window_width = 800
        window_height = 500
        self.root.geometry(f"{window_width}x{window_height}")

        # Create a frame for the sidebar on the left
        sidebar_width = 250
        self.sidebar = tk.Frame(self.root, bg="black", width=sidebar_width)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)  # Prevent the sidebar from resizing

        self.first_sidebar()

        # Create the main content frame on the right
        main_content_width = window_width - sidebar_width
        self.main_content = tk.Frame(self.root, bg="grey", width=main_content_width)
        self.main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_content.pack_propagate(False)  # Prevent the main content frame from resizing

        # Add content to the main content frame
        self.video_label = tk.Label(self.main_content)
        self.video_label.pack(pady=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_program)


    def first_sidebar(self):
        """Setup the sidebar when the user first loads up the program"""
        self.clear_frame(self.sidebar)

        start_tracking_btn = tk.Button(self.sidebar, text="Start Tracking", command=self.on_start_tracking_click)
        start_tracking_btn.pack(pady=10)

        callibrate_btn = tk.Button(self.sidebar, text="Callibrate", command=self.on_callibrate_click)
        callibrate_btn.pack(pady=10)


    def second_sidebar(self):
        """setup the sidebar when start tracking button is clicked"""
        self.clear_frame(self.sidebar)

        pause_tracking_btn = tk.Button(self.sidebar, text="Pause Tracking", command=self.on_pause_tracking_click)
        pause_tracking_btn.pack(pady=10)

        stop_tracking_btn = tk.Button(self.sidebar, text="Stop Tracking", command=self.on_stop_tracking_click)
        stop_tracking_btn.pack(pady=10)

        # callibrate_again_btn = tk.Button(self.sidebar, text="Callibrate Again", command=self.on_callibrate_click)
        # callibrate_again_btn.pack(pady=10)


    def third_sidebar(self):
        """setup the sidebar when pause tracking button is clicked"""
        self.clear_frame(self.sidebar)

        resume_tracking_button = tk.Button(self.sidebar, text="Resume Tracking", command=self.on_resume_tracking_click)
        resume_tracking_button.pack(pady=10)

        stop_tracking_btn = tk.Button(self.sidebar, text="Stop Tracking", command=self.on_stop_tracking_click)
        stop_tracking_btn.pack(pady=10)

        # callibrate_again_btn = tk.Button(self.sidebar, text="Callibrate Again", command=self.on_callibrate_click)
        # callibrate_again_btn.pack(pady=10)


    def on_start_tracking_click(self):
        self.backend.start_tracking = True
        self.backend.start_webcam = True
        self.second_sidebar()

    def on_stop_tracking_click(self):
        self.backend.start_tracking = False
        self.backend.start_webcam = True
        self.first_sidebar()

    def on_pause_tracking_click(self):
        self.backend.start_tracking = False
        self.backend.start_webcam = False
        self.third_sidebar()

    def on_resume_tracking_click(self):
        self.backend.start_tracking = True
        self.backend.start_webcam = True
        self.second_sidebar()

    def on_callibrate_click(self):
        self.backend.callibrate()


    def clear_frame(self, frame):
        """delete all widgets of a frame"""
        for widget in frame.winfo_children():
            widget.destroy()


    def show_webcam_feed(self, cap):
        while True:
            if self.backend.exit:
                print("webcam thread exited")
                break
            
            if self.backend.start_webcam:
                ret, frame = cap.read()

                # Convert to PIL format to display in Tkinter Label
                image = Image.fromarray(frame)
                image = ImageTk.PhotoImage(image=image)

                self.video_label.config(image=image)
                self.video_label.image = image

                time.sleep(0.03)


    def on_close_program(self):
        self.backend.terminate_threads()
        self.root.destroy()
        

    def run(self):
        self.root.mainloop()


#TODO Create binaries
#TODO Port to Android if possible