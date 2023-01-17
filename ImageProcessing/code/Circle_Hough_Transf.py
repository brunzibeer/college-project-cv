import cv2 
import numpy as np

img = cv2.imread("res/img_01.jpg")
output = img.copy()
img = cv2.GaussianBlur(img, (7,7), 2)


edges = cv2.Canny(img, 100, 200)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30,  minRadius=5, maxRadius=50)


if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0, :]:
        cv2.circle(output, (c[0], c[1]), c[2], (0,255,0), 2)

cv2.imshow("Circles", output)
cv2.imwrite("C:/Users/checc/Desktop/cerchi.jpg", output)
cv2.waitKey(0)