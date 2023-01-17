# attempt to realize player detection with background subtraction on a video
import cv2
import numpy as np

lower_blue = np.array([52,0,0])
upper_blue = np.array([154,160,245])
lower_red = np.array([0,0,0])
upper_red = np.array([33,255,255])

cap = cv2.VideoCapture("res/cutvideo.mp4")
backsub = cv2.createBackgroundSubtractorMOG2()

out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

while True:
    success, frame = cap.read()
    if success:
        bsMask = backsub.apply(frame)
        mask_fil = cv2.medianBlur(bsMask, 5)

        cont, hierarchy = cv2.findContours(mask_fil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kernel = np.ones((7,7),np.uint8)
        thresh = cv2.morphologyEx(mask_fil, cv2.MORPH_CLOSE, kernel)

        cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cont:
            x, y, w, h = cv2.boundingRect(cnt)
            if(h>=(1.5)*w):
                if(w>15 and h>= 15):
                    if cv2.contourArea(cnt) > 50 and cv2.contourArea(cnt) < 5000:
                        person = frame[y:y+h, x:x+w] 
                        person = cv2.GaussianBlur(person, (7,7), 3)
                        person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)

                        mask_blue = cv2.inRange(person_hsv, lower_blue, upper_blue)
                        nonzero_blue = cv2.countNonZero(mask_blue)
                        mask_red = cv2.inRange(person_hsv, lower_red, upper_red)
                        nonzero_red = cv2.countNonZero(mask_red)
                        if nonzero_red > 650:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)
                            cv2.putText(frame, "Team red", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                        else:
                            if nonzero_blue > 500:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 1)
                                cv2.putText(frame, "Team Blue", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                            else:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 1)
                                cv2.putText(frame, "Unknown", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)



        cv2.imshow("Video", frame)
        out.write(frame)
    
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
