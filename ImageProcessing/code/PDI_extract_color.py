# player detection with green extraction and team classification with color extraction

import cv2
import numpy as np 
from CVfunctions import *

img_01 = cv2.imread("res/img_01.jpg")
field = cv2.imread("res/field2.jpg")
img = img_01.copy()

img_c = img_01.copy()
img_corner = img_01.copy()

# Color ranges
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30, 255, 255])

lower_blue = np.array([52,50,50])
upper_blue = np.array([130,255,255])

lower_white = np.array([0,0,0])
upper_white = np.array([84,58,255])

# Homography matrix
pts_src = np.array([[649, 165], [468, 250], [106, 207], [144, 127]])
pts_dst = np.array([[118, 115], [119, 176], [67, 156], [37, 115]])

H, status = cv2.findHomography(pts_src, pts_dst)

# Image processing
img_blurred = cv2.GaussianBlur(img, (13,13), 7)
img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

res = cv2.bitwise_or(img, img, mask=green_mask)
res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)

kernel = np.ones((13,13),np.uint8)
thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

thresh = cv2.medianBlur(thresh, 5) # median filtering to remove some noise

cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cont:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 450 and cv2.contourArea(cnt) < 5000:
        person = img[y:y+h, x:x+w] # player founded
        person = cv2.GaussianBlur(person, (7,7), 3)
        person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(person_hsv, lower_yellow, upper_yellow)
        nonzero_yellow = cv2.countNonZero(mask_yellow)

        mask_blue = cv2.inRange(person_hsv, lower_blue, upper_blue)
        nonzero_blue = cv2.countNonZero(mask_blue)
        

        if nonzero_yellow > 50:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 1)
            cv2.putText(img, "Referee", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)

            point = np.array([[x+w//2, y+h]], np.float32)
            point=np.array([point])

            outPoint = cv2.perspectiveTransform(point, H)
            outX = round(outPoint[0][0][0])
            outY = round(outPoint[0][0][1])

            cv2.circle(field, (outX, outY), 2, (0,255,255), 5)

        else:
            if nonzero_blue > 50:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
                cv2.putText(img, "Team blue", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)

                point = np.array([[x+w//2, y+h]], np.float32)
                point = np.array([point])

                outPoint = cv2.perspectiveTransform(point, H)
                outX = round(outPoint[0][0][0])
                outY = round(outPoint[0][0][1])

                cv2.circle(field, (outX, outY), 2, (255,0,0), 5)

            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
                cv2.putText(img, "Team white", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)
                
                point = np.array([[x+w//2, y+h]], np.float32)
                point=np.array([point])

                outPoint = cv2.perspectiveTransform(point, H)
                outX = round(outPoint[0][0][0])
                outY = round(outPoint[0][0][1])

                cv2.circle(field, (outX, outY), 2, (0,0,255), 5)


# Ball detection
template = cv2.imread("res/ball_template02.jpg", 0)
w, h = template.shape[::-1]

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

threshold_ball = 0.95

loc = np.where( res >= threshold_ball)

for pt in zip(*loc[::-1]):
    #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,0), 1)
    #cv2.putText(img, "Ball", (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    #img = cv2.circle(img, center=(pt[0] + w//2, pt[1] + h), radius=2, color=(255,255,255), thickness=1)

    point = np.array([[pt[0] + w//2, pt[1] + h]], np.float32)
    point=np.array([point])

    outPoint = cv2.perspectiveTransform(point, H)
    outX = round(outPoint[0][0][0])
    outY = round(outPoint[0][0][1])

    cv2.circle(field, (outX, outY), 2, (0,0,0), 5)

stackTemp = stackImages(0.5, [[img_01, img_blurred, img_hsv], [green_mask, res_gray, thresh]])
#cv2.namedWindow("Img stack", cv2.WINDOW_NORMAL)
#cv2.imshow("Img stack", stackTemp)
cv2.imwrite("C:/Users/checc/Desktop/img.jpg", img)

cv2.imshow("Players", img)
cv2.imshow("Field", field)


cv2.waitKey(0)