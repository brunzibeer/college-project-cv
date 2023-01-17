import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

original = cv2.imread("res/calcio3.jpg")
field = cv2.imread("res/field2.jpg")
img = original.copy()
nms = original.copy()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

filtered = cv2.bilateralFilter(img, 5, 120, 120)

regions, weights = hog.detectMultiScale(filtered, winStride=(2,2), padding=(10,10), scale=1.05)

print(regions)
print(len(regions))

for (x,y,w,h) in regions:
    cv2.rectangle(original, (x,y), (x+w,y+h), (0,0,255), 1)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)

for (xA, yA, xB, yB) in pick:
	cv2.rectangle(nms, (xA, yA), (xB, yB), (0, 255, 0), 2)

print("\nNMS:")
print(pick)
print(len(pick))

cv2.imshow("Hog detection", original)
cv2.imwrite("C:/Users/checc/Desktop/before.jpg", original)
cv2.imwrite("C:/Users/checc/Desktop/after.jpg", nms)
cv2.waitKey(0)