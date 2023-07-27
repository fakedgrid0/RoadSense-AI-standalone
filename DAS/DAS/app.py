import sys
import time
from GUI import GUI
from voice_engine import voice_engine
from drowsiness_detection import drowsiness_detection
from head_pose_estimation import head_pose_estimation
from responses import responses
import cv2
import random
import threading


def get_audio(detection_type: str):
    audio = random.choice(responses[detection_type])
    return audio


class DriverAidSystem:
    def __init__(self) -> None:
        self.audio_list = []
        self.start_tracking = False
        self.start_webcam = True
        self.exit = False
        
        self.cap = cv2.VideoCapture(0)

        self.gui = GUI.GUI(self)
        self.voice_engine = voice_engine.VoiceEngine()
        self.drowsiness_detector = drowsiness_detection.DrowsinessDetector()
        self.head_pose_detector = head_pose_estimation.HeadPoseEstimator()

        self.drowsiness_detector.EAR_THRESH = 0.21
        self.drowsiness_detector.WAIT_TIME = 0.9
        self.head_pose_detector.WAIT_TIME = 0.7

        self.drowsiness_detect_bool = False
        self.head_pose_detect_bool = False
        self.sleep_reminder_bool = False

        self.tracking_thread = threading.Thread(target=self.track, daemon=True)
        self.webcam_thread = threading.Thread(target=self.gui.show_webcam_feed, args=(self.cap,), daemon=True)
        self.start()

    
    def start(self):
        self.tracking_thread.start()
        self.webcam_thread.start()
        self.gui.run()
        

    def track(self):
        while True:
            if self.exit:
                print("tracking thread exited")
                break
                
            if self.start_tracking:
                ret, inp_frame = self.cap.read()

                # run detections on inp_frame
                image_drowsiness_detection, self.drowsiness_detect_bool = self.drowsiness_detector.run(inp_frame)
                image_head_pose_detection, self.head_pose_detect_bool = self.head_pose_detector.run(inp_frame)

                # check for detections
                if self.drowsiness_detect_bool:
                    print("Drowsy detected")
                    self.audio_list.append(get_audio("drowsiness_detect"))
                if self.head_pose_detect_bool:
                    print("Head pose detected")
                    self.audio_list.append(get_audio("head_pose_detect"))
                
                # define audio and speak
                audio = " and ".join(self.audio_list)
                
                self.voice_engine.speak(audio)
                self.audio_list.clear()       


    def callibrate(self):
        self.countdown()
        c_ret, c_frame = self.cap.read()
        self.head_pose_detector.callibrate(c_frame)
        self.voice_engine.speak("callibration complete")

    def countdown(self):
        try:
            self.voice_engine.engine.endLoop()
        except RuntimeError:
            print("runtime endloop error")

        self.voice_engine.speak("3,, 2,, 1")


    def terminate_threads(self):
        self.exit = True
        self.start_tracking = False
        self.start_webcam = False

        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()