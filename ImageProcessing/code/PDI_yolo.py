import cv2
import numpy as np

# thresholds
conf_th = 0.6
nms_th = 0.4

# color ranges
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30, 255, 255])

lower_blue = np.array([52,50,50])
upper_blue = np.array([130,255,255])

# Homography matrix
pts_src = np.array([[649, 165], [468, 250], [106, 207], [144, 127]])
pts_dst = np.array([[118, 115], [119, 176], [67, 156], [37, 115]])

H, status = cv2.findHomography(pts_src, pts_dst)

# COCO classes
classes = []
classesFile = 'res/YOLOcfg/coco.names'
with open(classesFile, 'r') as f:
    classes = f.read().rstrip('\n').split('\n')

#colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Weights and configurations
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

# Load images
img = cv2.imread("res/img_01.jpg")
field = cv2.imread("res/field2.jpg")
copy = img.copy()
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
            # rectangle coords
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
        conf = str(round(confidences[i], 2))
        if label == 'person':
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
                if nonzero_blue > 70:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
                    cv2.putText(img, "Team blue", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                    img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)

                    point = np.array([[x+w//2, y+h]], np.float32)
                    point=np.array([point])

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
        elif label == 'sports ball':
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.putText(img, "Ball", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            img = cv2.circle(img, center=(x+w//2, y+h), radius=2, color=(255,255,255), thickness=1)

            point = np.array([[x+w//2, y+h]], np.float32)
            point=np.array([point])

            outPoint = cv2.perspectiveTransform(point, H)
            outX = round(outPoint[0][0][0])
            outY = round(outPoint[0][0][1])

            cv2.circle(field, (outX, outY), 2, (0,0,0), 5)
            


cv2.imshow("Img", img)
cv2.imshow("Field", field)
cv2.waitKey(0)