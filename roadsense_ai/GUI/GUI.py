import tkinter as tk
import cv2
from PIL import Image, ImageTk

class GUI:
    def __init__(self, backend):
        self.main_options_logic = {
            "start": [
                ("Start Tracking", self.on_start_tracking_click),
                ("Callibrate", self.on_callibrate_click),
            ],
            "tracking": [
                ("Pause Tracking", self.on_pause_tracking_click),
                ("Stop Tracking", self.on_stop_tracking_click),
            ],
            "paused": [
                ("Resume Tracking", self.on_resume_tracking_click),
                ("Stop Tracking", self.on_stop_tracking_click),
            ],
        }

        self.raw_view = True
        self.tracking_view = False
        self.can_open_settings = True
        self.webcam_queue = []
        self.backend = backend

        self.root = tk.Tk()
        self.root.title("RoadSense AI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_program)
        window_width = 800
        window_height = 500
        self.root.geometry(f"{window_width}x{window_height}")

        # Create a frame for the sidebar on the left
        sidebar_width = 300
        self.sidebar = tk.Frame(self.root, bg="grey", width=sidebar_width)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)  # Prevent the sidebar from resizing
        
        # Create the main content frame on the right
        self.main_content_width = window_width - sidebar_width
        self.main_content = tk.Frame(self.root, bg="white", width=self.main_content_width)
        self.main_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_content.pack_propagate(False)  # Prevent the main content frame from resizing

        self.video_label = tk.Label(self.main_content)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.create_sidebar("start")

    def create_sidebar(self, state):
        """Setup the sidebar based on the state, (start, tracking, paused)"""
        self.clear_frame(self.sidebar)

        # fill main options frame in sidebar
        main_options_label = tk.Label(self.sidebar, text="Main Options")
        main_options_label.pack(fill=tk.X)
        for text, command in self.main_options_logic[state]:
            btn = tk.Button(self.sidebar, text=text, command=command)
            btn.pack(pady=10)

        view_options_label = tk.Label(self.sidebar, text="View Options")
        view_options_label.pack(fill=tk.X)
        raw_view_btn = tk.Button(self.sidebar, text="Raw", command=self.on_raw_view_click)
        raw_view_btn.pack(pady=10)
        tracking_view_btn = tk.Button(self.sidebar, text="Tracker", command=self.on_tracking_view_click)
        tracking_view_btn.pack(pady=10)

        config_options_label = tk.Label(self.sidebar, text="Configuration")
        config_options_label.pack(fill=tk.X)
        settings_button = tk.Button(self.sidebar, text="Settings", command=self.on_open_settings)
        settings_button.pack(pady=10)
        help_btn = tk.Button(self.sidebar, text="Help", command=self.open_help)
        help_btn.pack(pady=10)


    #* Improve
    def update_vars(self):
        self.backend.drowsiness_detector.update_wait_time(float(self.d_w_t_spinbox.get()))
        self.backend.drowsiness_detector.update_ear_thresh(float(self.d_e_t_spinbox.get()))
        self.backend.head_pose_detector.update_wait_time(float(self.h_w_t_spinbox.get()))
        self.backend.head_pose_detector.update_offset(float(self.h_offset_spinbox.get()))

    def open_settings(self):
        """Open the settings window"""
        self.can_open_settings = False
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_close_settings)
        self.settings_window.geometry("400x600")
        self.settings_window.title("Settings")        
        
        tk.Label(self.settings_window, text="Drowsiness Detection", bg="grey").pack(fill=tk.X)
        tk.Label(self.settings_window, text="Wait Time (ms)").pack(pady=5)
        self.d_w_t_spinbox = tk.Spinbox(self.settings_window, from_=1, to=100)
        self.d_w_t_spinbox.delete(0, tk.END)
        self.d_w_t_spinbox.insert(0, str(self.backend.drowsiness_detector.WAIT_TIME))
        self.d_w_t_spinbox.pack()

        tk.Label(self.settings_window, text="EAR Threshold").pack(pady=5)
        self.d_e_t_spinbox = tk.Spinbox(self.settings_window, from_=0.1, to=0.30)
        self.d_e_t_spinbox.delete(0, tk.END)
        self.d_e_t_spinbox.insert(0, str(self.backend.drowsiness_detector.EAR_THRESH))
        self.d_e_t_spinbox.pack(pady=5)

        tk.Label(self.settings_window, text="Head Pose Detection", bg="grey").pack(fill=tk.X)
        tk.Label(self.settings_window, text="Wait Time (ms)").pack(pady=5)
        self.h_w_t_spinbox = tk.Spinbox(self.settings_window, from_=1, to=100)
        self.h_w_t_spinbox.delete(0, tk.END)
        self.h_w_t_spinbox.insert(0, str(self.backend.head_pose_detector.WAIT_TIME))
        self.h_w_t_spinbox.pack()

        tk.Label(self.settings_window, text="Head Offset").pack(pady=5)
        self.h_offset_spinbox = tk.Spinbox(self.settings_window, from_=1, to=50)
        self.h_offset_spinbox.delete(0, tk.END)
        self.h_offset_spinbox.insert(0, str(self.backend.head_pose_detector.OFFSET))
        self.h_offset_spinbox.pack()

        update_button = tk.Button(self.settings_window, text="Update Variables", command=self.update_vars)
        update_button.pack(pady=20)

        self.settings_window.mainloop()

    def open_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.geometry("800x800")
        help_window.title("Help")

        help_str = """
        Steps To start using RoadSense AI:
        1. Click on Callibrate, to let the the program know what your front is
        2. Click on Start Tracking, to start tracking using your webcam

        Button Uses:
        Raw -> Use video which is not processed by AI, to show to you
        Tracker -> Use video which is processed by AI, which the AI uses to do various detections
        Settings -> You can use this to change the variables which the AI uses for detections

        Important Notes:

        1. There is a problem that before starting tracking, the camera is a bit slow (bug)
        2. The Tracker button will only work if the Start Tracking button is pressed
        3. Full explanation on how the program works soon
        4. All of the directions will be defined based on the forward direction, which must be callibrated
        """

        help_text = tk.Text(help_window, font=("Arial", 12))
        help_text.insert(tk.END, help_str)
        help_text.pack()

    def on_start_tracking_click(self):
        self.backend.start_tracking = True
        self.backend.start_webcam = True
        self.create_sidebar("tracking")

    def on_stop_tracking_click(self):
        self.backend.start_tracking = False
        self.backend.start_webcam = True
        self.raw_view = True
        self.tracking_view = False
        self.create_sidebar("start")

    def on_pause_tracking_click(self):
        self.backend.start_tracking = False
        self.backend.start_webcam = False
        self.create_sidebar("paused")

    def on_resume_tracking_click(self):
        self.backend.start_tracking = True
        self.backend.start_webcam = True
        self.create_sidebar("tracking")

    def on_callibrate_click(self):
        self.backend.callibrate()

    def on_raw_view_click(self):
        self.raw_view = True
        self.tracking_view = False

    def on_tracking_view_click(self):
        if self.backend.start_tracking:
            self.raw_view = False
            self.tracking_view = True

    def on_open_settings(self):
        if self.can_open_settings:
            self.open_settings()

    def on_close_settings(self):
        self.update_vars()
        self.settings_window.destroy()
        self.can_open_settings = True

    def on_close_program(self):
        self.backend.terminate_threads()
        self.root.destroy()


    def update_webcam_feed(self, cap):
        while True:
            if not self.tracking_view and self.webcam_queue:
                self.webcam_queue.clear()

            if self.backend.start_webcam:

                if self.raw_view:
                    _, frame = cap.read()
                    frame = cv2.flip(frame, 1)

                elif self.tracking_view:
                    if self.webcam_queue:
                        frame = self.webcam_queue[0]
                        self.webcam_queue.pop(0)

                # Convert to PIL format to display in Tkinter Label
                image = Image.fromarray(frame)
                image = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=image)
                self.video_label.image = image

    def clear_frame(self, frame):
        """delete all widgets of a frame"""
        for widget in frame.winfo_children():
            widget.destroy()
        
    def run(self):
        try:
            self.root.mainloop()
        except AttributeError:
            pass