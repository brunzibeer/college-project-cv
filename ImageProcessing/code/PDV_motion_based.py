import cv2
import numpy as np

#cap = cv2.VideoCapture("res/soccer.mp4")

cap = cv2.VideoCapture(0)

baseline_img = None

while True:
    success, frame = cap.read()
    if success:
        frame_to_use = frame.copy()
        frame_to_use = cv2.GaussianBlur(frame_to_use, (13,13), 2)
        gray_frame = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (13,13),2)

        if baseline_img is None:
            baseline_img = gray_frame
            continue

        delta = cv2.absdiff(baseline_img, gray_frame)
        thresh_frame = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
        cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
    
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("Difference", delta)
        cv2.imshow("Thresh", thresh_frame)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
