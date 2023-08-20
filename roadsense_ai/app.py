import sys
import random
import threading
import time
import cv2
import mediapipe as mp
import winsound

from .GUI import GUI
from .voice_engine import voice_engine
from .drowsiness_detection import drowsiness_detection
from .head_pose_estimation import head_pose_estimation
from . import responses


def get_audio(detection_type: str):
    return random.choice(responses.responses[detection_type])

def get_face_mesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,)


class DriverAidSystem:
    EAR_THRESH = 0.24
    DROWSINESS_WAIT_TIME = 0.9
    HEAD_POSE_WAIT_TIME = 0.7
    HEAD_POSE_OFFSET = 10


    def __init__(self) -> None:
        self.alpha = 0.5
        self.start_tracking = False
        self.start_webcam = True
        self.driver_not_visible = False
        self.drowsiness_detect_bool = False
        self.head_pose_detect_bool = False
        self.audio_lock = threading.Lock()
        self.audio_list = []

        self.frame1 = None
        self.frame2 = None
        self.bool_1 = False
        self.bool_2 = False

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = get_face_mesh()

        self.gui = GUI.GUI(self)
        self.voice_engine = voice_engine.VoiceEngine()
        self.drowsiness_detector = drowsiness_detection.DrowsinessDetector(self.face_mesh)
        self.head_pose_detector = head_pose_estimation.HeadPoseEstimator(self.face_mesh)

        self.drowsiness_detector.EAR_THRESH = DriverAidSystem.EAR_THRESH
        self.drowsiness_detector.WAIT_TIME = DriverAidSystem.DROWSINESS_WAIT_TIME
        self.head_pose_detector.WAIT_TIME = DriverAidSystem.HEAD_POSE_WAIT_TIME
        self.head_pose_detector.OFFSET = DriverAidSystem.HEAD_POSE_OFFSET


    def start(self):
        self.tracking_thread = threading.Thread(target=self.track, daemon=True)
        self.webcam_thread = threading.Thread(target=self.gui.update_webcam_feed, args=(self.cap,), daemon=True)
        self.tracking_thread.start()
        self.webcam_thread.start()
        self.gui.run()

    def track(self):
        counter = 0
        max_counter_limit = 10  # Adjust this value to set the time limit
        
        while True:
            if self.start_tracking:
                try:
                    _, inp_frame = self.cap.read()

                    frame_start_time = time.time()

                    # run detections on inp_frame
                    self.frame1, self.bool_1 = self.drowsiness_detector.run(inp_frame)
                    try:
                        self.frame2, self.bool_2 = self.head_pose_detector.run(inp_frame)
                        counter = 0
                        self.driver_not_visible = False
                    
                    except TypeError:
                        counter += 1
                        if counter >= max_counter_limit:
                            self.driver_not_visible = True
    
                except AttributeError:
                    return None

                try:
                    if self.gui.tracking_view:    
                        # Combine the two frames using alpha blending            
                        combined_frame = cv2.addWeighted(self.frame1, 1 - self.alpha, self.frame2, self.alpha, 0)
                        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                        self.gui.webcam_queue.append(combined_frame)
                except cv2.error:
                    pass

                # check for detections
                if self.bool_1:
                    self.add_audio("drowsiness_detect")
                if self.bool_2:
                    self.add_audio("head_pose_detect")  
                if self.driver_not_visible:
                    self.add_audio("driver_not_visible")               

                if self.audio_list:
                    # define audio and speak
                    audio = " and ".join(self.audio_list)
                    voice_thread = threading.Thread(target=self.speak, args=(audio,), daemon=True)
                    voice_thread.start()
                    
                    
                frame_end_time = time.time()
                fps = 1.0 / (frame_end_time - frame_start_time)
                print(fps)
                

    def speak(self, audio):  
        self.voice_engine.speak(audio)
        self.audio_list.clear()
        return None
    
    def add_audio(self, detection_type: str):
        with self.audio_lock:
            self.audio_list.append(get_audio(detection_type))

    def callibrate(self):
        self.voice_engine.speak("The callibration is starting, please look forward.")

        try:
            self.voice_engine.engine.endLoop()
        except RuntimeError:
            pass

        self.voice_engine.speak("In 3,, 2,, 1")
        _, c_frame = self.cap.read()
        self.head_pose_detector.callibrate(c_frame)
        self.voice_engine.speak("The calibration was successful, you can now start tracking.")
    

    def terminate_threads(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()