import cv2
import numpy as np

img = cv2.imread("res/img_01.jpg")
field = cv2.imread("res/field2.jpg")

cv2.circle(img, (649,165), 2, (0,0,255),2)
cv2.putText(img, "1", (649+10,165), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (468,250), 2, (0,0,255),2)
cv2.putText(img, "2", (468+10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (106,207), 2, (0,0,255),2)
cv2.putText(img, "3", (106+10,207), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (144,127), 2, (0,0,255),2)
cv2.putText(img, "4", (144+10,127), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

cv2.circle(field, (118,115), 2, (255,0,0),2)
cv2.putText(field, "1", (118+10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (119, 176), 2, (255,0,0),2)
cv2.putText(field, "2", (119+10, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (67, 156), 2, (255,0,0),2)
cv2.putText(field, "3", (67+10, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (37,115), 2, (255,0,0),2)
cv2.putText(field, "4", (37+10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

pts_src = np.array([[649, 165], [468, 250], [106, 207], [144, 127]])
pts_dst = np.array([[118, 115], [119, 176], [67, 156], [37, 115]])

H, status = cv2.findHomography(pts_src, pts_dst)

point = np.array([[480, 320]], np.float32)
point = np.array([point])

point_in_field = cv2.perspectiveTransform(point, H)
outX = round(point_in_field[0][0][0])
outY = round(point_in_field[0][0][1])

cv2.circle(img, (480, 320), 2, (255,0,255),2)
cv2.putText(img, "P", (480+10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)
cv2.circle(field, (outX, outY), 2, (255,0,255),2)
cv2.putText(field, "P", (outX+10,outY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)

cv2.imshow("Img", img)
cv2.imshow("Field", field)
cv2.waitKey(0)