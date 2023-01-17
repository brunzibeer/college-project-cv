# attempt to find interesting points in an automatic way
import cv2
import numpy as np
from CVfunctions import *

img = cv2.imread("res/img_01.jpg")

# green range
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

img_blurred = cv2.GaussianBlur(img, (13,13), 7)
img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

res = cv2.bitwise_and(img, img, mask=green_mask)
res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(res_gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(canny, 1, np.pi/180, 100)

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 1)

for i in range(0, len(lines)):
    for j in range(0, len(lines)):
        if i != j:
            points = intersection(lines[i], lines[j])
            if points is not None:
                cv2.circle(img, (points[0], points[1]), 2, (0,0,255),2)
        

cv2.imshow("Img", img)
cv2.waitKey(0)
