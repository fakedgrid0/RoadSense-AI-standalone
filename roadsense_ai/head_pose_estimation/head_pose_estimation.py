import cv2
import mediapipe as mp
import numpy as np
import time


def get_face_mesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    face_mesh    =  mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_num_faces, 
                    min_detection_confidence=min_detection_confidence, 
                    min_tracking_confidence=min_tracking_confidence)
    
    return face_mesh

def process_image(image):
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = False 
    return image

def get_camera_matrix(img_h, img_w):
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1] ])
    return cam_matrix


def get_rotvec(face_3d, face_2d, cam_matrix, dist_matrix):
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    return rot_vec

def get_rmat(rot_vec):
    rmat, jac = cv2.Rodrigues(rot_vec)
    return rmat

def get_angles(img_h, img_w, face_3d, face_2d):
    cam_matrix = get_camera_matrix(img_h, img_w)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    rot_vec = get_rotvec(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat = get_rmat(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles

def get_direction(fx, fy, x, y, offset:int):
    if y-fy < -offset:
        return "Left"
    elif y-fy > offset:
        return "Right"
    elif x-fx < -offset:
        return "Down"
    elif x-fx > offset:
        return "Up"
    else:
        return "Forward"

def plot_nose_line(image, nose_2d, x, y):
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
    cv2.line(image, p1, p2, (255, 0, 0), 3)

def plot_text(image, head_direction, x, y, z):
    cv2.putText(image, head_direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


class HeadPoseEstimator():
    def __init__(self) -> None:
        self.WAIT_TIME = 1.0
        self.OFFSET = 15
        
        self.face_mesh = get_face_mesh()
        self.target_landmarks = [33, 263, 1, 61, 291, 199]
        self.good_directions = ["Forward"]
        self.play_alarm = False
        self.forward_x = 0.0
        self.forward_y = 0.0
        self.d_time = 0.0
        self.start_time = time.perf_counter()

    def update_wait_time(self, new_value):
        self.WAIT_TIME = new_value
    
    def update_offset(self, new_value):
        self.OFFSET = new_value

    def callibrate(self, image):
        image = process_image(image)
        direction_str, x, y = self.get_head_direction(image)
        self.forward_x = x
        self.forward_y = y

    def get_head_direction(self, image):
        results = self.face_mesh.process(image)
        
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for idx, lm in enumerate(landmarks):
                if idx in self.target_landmarks:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            angles = get_angles(img_h, img_w, face_3d, face_2d)

            # scale the angles, I guess
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
        
            head_direction = get_direction(self.forward_x, self.forward_y, x, y, offset=self.OFFSET)
            
            plot_nose_line(image, nose_2d, x, y)
            # plot_text(image, head_direction, x, y, z)

        #? How to handle no face detected errors more elegantly
        try:
            return head_direction, x, y
        except UnboundLocalError:
            return None, None, None
        
        
    def run(self, image):
        image = process_image(image)
        head_direction, x, y = self.get_head_direction(image)
        
        if head_direction:
            if head_direction not in self.good_directions:
                end_time = time.perf_counter()
                self.d_time += end_time - self.start_time
                self.start_time = time.perf_counter()

                if self.d_time >= self.WAIT_TIME:
                    self.play_alarm = True
            else:
                self.start_time = time.perf_counter()
                self.d_time = 0.0
                self.play_alarm = False
        else:
            self.start_time = time.perf_counter()
            self.d_time = 0.0
            self.play_alarm = False
            
    
        return image, self.play_alarm
        return self.play_alarm