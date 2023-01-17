# attempt to find correspondence (at least four) between match image and soccer field
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

MIN_MATCH_COUNT = 3

img = cv2.imread("res/img_01.jpg")
field = cv2.imread("res/field.jpg")

sift = cv2.SIFT_create()

kp_img, des_img = sift.detectAndCompute(img, None)
kp_fld, des_fld = sift.detectAndCompute(field, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_img, des_fld, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print(f"Number of good matches founded: {len(good)}")

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_fld[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel(). tolist()
    
    h,w,d = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(field,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img, kp_img, field, kp_fld, good, None, **draw_params)
plt.imshow(img3, 'gray'),plt.show()

path = r'C:\Users\checc\Desktop\Ingegneria Informatica\Computer Vision\Project\ransac_05.jpg'
cv2.imwrite(path, img3)