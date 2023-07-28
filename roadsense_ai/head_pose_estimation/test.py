from head_pose_estimation import HeadPoseEstimator
import cv2

app = HeadPoseEstimator()

vid = cv2.VideoCapture(0)
c_ret, c_frame = vid.read()
app.callibrate(c_frame)
print("callibrated")

while True:
    ret, frame = vid.read()
    
    frame, play_alarm = app.run(frame)
    if play_alarm:
        print("Look Forward")

    cv2.imshow("Head Pose Estimation", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()


