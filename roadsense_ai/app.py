import sys
import random
import threading
import cv2

from .GUI import GUI
from .voice_engine import voice_engine
from .drowsiness_detection import drowsiness_detection
from .head_pose_estimation import head_pose_estimation
from . import responses


def get_audio(detection_type: str):
    return random.choice(responses.responses[detection_type])


class DriverAidSystem:
    EAR_THRESH = 0.21
    DROWSINESS_WAIT_TIME = 0.9
    HEAD_POSE_WAIT_TIME = 0.7
    HEAD_POSE_OFFSET = 15

    def __init__(self) -> None:
        self.alpha = 0.5
        self.start_tracking = False
        self.start_webcam = True
        self.drowsiness_detect_bool = False
        self.head_pose_detect_bool = False
        self.audio_lock = threading.Lock()
        self.audio_list = []

        self.cap = cv2.VideoCapture(0)

        self.gui = GUI.GUI(self)
        self.voice_engine = voice_engine.VoiceEngine()
        self.drowsiness_detector = drowsiness_detection.DrowsinessDetector()
        self.head_pose_detector = head_pose_estimation.HeadPoseEstimator()

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
        while True:
            if self.start_tracking:
                try:
                    _, inp_frame = self.cap.read()

                    # run detections on inp_frame
                    frame1, self.drowsiness_detect_bool = self.drowsiness_detector.run(inp_frame)
                    frame2, self.head_pose_detect_bool = self.head_pose_detector.run(inp_frame)

                except AttributeError:
                    break

                if self.gui.tracking_view:    
                    # Combine the two frames using alpha blending            
                    combined_frame = cv2.addWeighted(frame1, 1 - self.alpha, frame2, self.alpha, 0)
                    self.gui.webcam_queue.append(combined_frame)

                # check for detections
                if self.drowsiness_detect_bool:
                    print("Drowsy detected")
                    self.add_audio("drowsiness_detect")
                if self.head_pose_detect_bool:
                    print("Head pose detected")
                    self.add_audio("head_pose_detect")

                if self.audio_list:
                    # define audio and speak
                    audio = " and ".join(self.audio_list)
                    with self.audio_lock:
                        self.voice_engine.speak(audio)
                        self.audio_list.clear()

    def add_audio(self, detection_type: str):
        with self.audio_lock:
            self.audio_list.append(get_audio(detection_type))


    def callibrate(self):
        self.voice_engine.speak("The callibration is starting, please look forward.")

        try:
            self.voice_engine.engine.endLoop()
        except RuntimeError:
            print("runtime endloop error")

        self.voice_engine.speak("In 3,, 2,, 1")

        _, c_frame = self.cap.read()
        self.head_pose_detector.callibrate(c_frame)
        self.voice_engine.speak("The calibration was successful")
    

    def terminate_threads(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()
