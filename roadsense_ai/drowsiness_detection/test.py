from drowsiness_detection import DrowsinessDetector
import cv2

thresholds = {"WAIT_TIME":2.0,
              "EAR_THRESH":0.18
}


app = DrowsinessDetector()
app.WAIT_TIME = 1.0
app.EAR_THRESH = 0.20

vid = cv2.VideoCapture(0)


while True:    
    success, frame = vid.read()
    frame, play_alarm = app.run(frame)

    print(play_alarm)
    
    cv2.imshow('drowsiness detection', frame)
      
    if cv2.waitKey(1) == ord('q'):
        break
  

vid.release()
cv2.destroyAllWindows()