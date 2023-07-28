import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def get_face_mesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,)

def process_image(image):
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    return image

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, img_w, img_h):
    # Calculate Eye Aspect Ratio for one eye.
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, img_w, img_h)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, img_w, img_h):
    # Calculate average Eye aspect ratio
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, img_w, img_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, img_w, img_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(image, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(image, coord, 2, color, -1)

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    cv2.putText(image, text, origin, font, fntScale, color, thickness)


class DrowsinessDetector:
    def __init__(self):
        self.EAR_THRESH = 0.20
        self.WAIT_TIME = 1.0  # secs

        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        self.start_time = time.perf_counter()
        self.d_time = 0.0
        self.play_alarm = False
        self.EAR_txt_pos = (10, 30)

        self.facemesh_model = get_face_mesh()


    def run(self, image: np.array):
        # This function is used to implement our Drowsy detection algorithm
        image = process_image(image)

        img_h, img_w, _ = image.shape    
        DROWSY_TIME_txt_pos = (10, int(img_h // 2 * 1.7))
        ALM_txt_pos = (10, int(img_h // 2 * 1.85))

        results = self.facemesh_model.process(image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], img_w, img_h)
            plot_eye_landmarks(image, coordinates[0], coordinates[1], (0, 255, 0))

            if EAR < self.EAR_THRESH:

                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()
                self.d_time += end_time - self.start_time
                self.start_time = time.perf_counter()

                if self.d_time >= self.WAIT_TIME:
                    self.play_alarm = True
                    plot_text(image, "WAKE UP! WAKE UP", ALM_txt_pos, (255, 0 , 0))

            else:
                self.start_time = time.perf_counter()
                self.d_time = 0.0
                self.play_alarm = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.d_time, 3)} Secs"
            plot_text(image, EAR_txt, self.EAR_txt_pos, (0, 255, 0))
            plot_text(image, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, (0, 255, 0))

        else:
            self.start_time = time.perf_counter()
            self.d_time = 0.0
            self.play_alarm = False


        return image, self.play_alarm