# Yolo, homographic transform with six points and color extraction
import cv2
import numpy as np

field = cv2.imread("res/field2.jpg")
img = cv2.imread("res/img_02.jpg")
copy = img.copy()

lower = np.array([0,21,0])
upper = np.array([43,255,255])

cv2.circle(copy, (316, 76), 2, (0,0,255),2)
cv2.putText(copy, "1", (316+10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(copy, (708, 170), 2, (0,0,255),2)
cv2.putText(copy, "2", (708+10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(copy, (225,407), 2, (0,0,255),2)
cv2.putText(copy, "3", (225+10,407), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(copy, (43,290), 2, (0,0,255),2)
cv2.putText(copy, "4", (43+10,290), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(copy, (723, 47), 2, (0,0,255),2)
cv2.putText(copy, "5", (723+10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(copy, (862, 68), 2, (0,0,255),2)
cv2.putText(copy, "6", (862+10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)


cv2.circle(field, (313 , 32), 2, (255,0,0),2)
cv2.putText(field, "1", (313+10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (313 , 259), 2, (255,0,0),2)
cv2.putText(field, "2", (313+10, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (118 , 237), 2, (255,0,0),2)
cv2.putText(field, "3", (118+10, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (119,175), 2, (255,0,0),2)
cv2.putText(field, "4", (119+10,175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (508, 176), 2, (255,0,0),2)
cv2.putText(field, "5", (508+10, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (508, 240), 2, (255,0,0),2)
cv2.putText(field, "6", (508+10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

pts_src = np.array([[316, 76], [708, 170], [225,407], [43,290], [723, 47], [862, 68]])
pts_dst = np.array([[313 , 32], [313 , 259], [118 , 237], [119,175], [508, 176], [508,240]])

H, status = cv2.findHomography(pts_src, pts_dst)

# thresholds
conf_th = 0.2
nms_th = 0.4

classes = []
classesFile = 'res/YOLOcfg/coco.names'
with open(classesFile, 'r') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

Cfg = 'res/YOLOcfg/yolov3.cfg'
Weights = 'res/YOLOcfg/yolov3.weights'

CfgTiny = 'res/YOLO-tiny/yolov3-tiny.cfg'
WeightsTiny = 'res/YOLO-tiny/yolov3-tiny.weights'

# Load yolo
net = cv2.dnn.readNet(Weights, Cfg)
print("Net loaded")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(output_layers)

# Load image
height, width, ch = img.shape

scaling = 1/255
blob = cv2.dnn.blobFromImage(img, scaling, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outputs:
    for detection in out:
        scores = detection[5:]
        id_class = np.argmax(scores)
        confidence = scores[id_class]
        if confidence > conf_th:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(id_class)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        conf = str(round(confidences[i], 2))
        if label == 'person':
            person = img[y:y+h, x:x+w]
            person = cv2.GaussianBlur(person, (7,7), 3)
            person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(person_hsv, lower, upper)
            nonzero = cv2.countNonZero(mask)

            if nonzero > 150:  
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 2)
                img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)
                point = np.array([[x+w//2, y+h]], np.float32)
                point=np.array([point])
                outPoint = cv2.perspectiveTransform(point, H)
                outX = round(outPoint[0][0][0])
                outY = round(outPoint[0][0][1])
                cv2.circle(field, (outX, outY), 2, (0,255,255), 5)

            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)
                point = np.array([[x+w//2, y+h]], np.float32)
                point=np.array([point])
                outPoint = cv2.perspectiveTransform(point, H)
                outX = round(outPoint[0][0][0])
                outY = round(outPoint[0][0][1])
                cv2.circle(field, (outX, outY), 2, (0,0,255), 5)

cv2.imshow("Copy", copy)
cv2.imshow("Img", img)
cv2.imshow("Field", field)

cv2.imwrite("C:/Users/checc/Desktop/img.jpg", copy)
cv2.imwrite("C:/Users/checc/Desktop/field.jpg", field)
cv2.imwrite("C:/Users/checc/Desktop/detection.jpg", img)

cv2.waitKey(0)