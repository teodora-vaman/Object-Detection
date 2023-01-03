import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pybboxes as pbx


def drawWindow(img):
    h,w,l = img.shape
    x_win1 = w // 2 - int(w * 0.35)
    y_win1 = h // 2 - int(h * 0.35)

    x_win2 = w // 2 + int(w * 0.35)
    y_win2 = h // 2 + int(h * 0.35)
    start_point = (x_win1, y_win1)
    end_point = (x_win2, y_win2)

    cv2.rectangle(img, start_point, end_point, (125, 200, 0), 2)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def verifyIntersect(img, bboxes):
    h,w,l = img.shape
    x_win1 = w // 2 - int(w * 0.35)
    y_win1 = h // 2 - int(h * 0.35)
    x_win2 = w // 2 + int(w * 0.35)
    y_win2 = h // 2 + int(h * 0.35)
    drawWindow(img)
    intersect = 0
    prag = 500
    for box in bboxes:

        tmp_box = (box[0], box[1], box[2], box[3])
        x1 = int( float(tmp_box[0]) * w )
        y1 = int( float(tmp_box[1]) * h )
        xw = int( float(tmp_box[2]) * w /2)
        yw = int( float(tmp_box[3]) * h /2)

        rectInters = [0, 0, 0, 0]

        rectInters[0] = max(x1 - xw, x_win1) # x1
        rectInters[1] = max(y1 - yw, y_win1) # y1

        rectInters[2] = min(x1 + xw, x_win2) # x2
        rectInters[3] = min(y1 + yw, y_win2) # y2


        rectIntersArea = max(0, rectInters[2] - rectInters[0] + 1) * max(0, rectInters[3] - rectInters[1] + 1)

        boxArea = (y1 - yw - (x1 - xw) + 1) * (y1 + yw - (x1 + xw) + 1)
        if rectIntersArea > prag and boxArea > 50000:
            start_point = (x1 - xw, y1 - yw )
            end_point   = (x1 + xw, y1 + yw )
            cv2.rectangle(img, start_point, end_point, (0, 0, 255), 3)
            cv2.putText(img, text='SLOW DOWN', org=(x_win1 - 5, y_win1 - 5), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(122, 200, 0),thickness=3)
            cv2.putText(img, text="Obstacle: {} - confidence: {}".format(box[5], box[4]), org=(x_win1 + 15, y_win2 + 15), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(122, 200, 0),thickness=1)
            print("Obstacle: {} - confidence: {}".format(box[5], box[4]))
    
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture("E:\Lucru\ANUL II MASTER\SCCV\Proiect\Object-Detection\Road.mp4")

if (cap.isOpened()== False):
    print("Error opening video file")
 
size = (360, 640, 3)
# save the video
out = cv2.VideoWriter('obstacleDetection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,360))
i = 0
detected_frames = []
while(cap.isOpened()):
     
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame

        # cv2.imshow('Frame', frame)
        results = model(frame)

        boxes = results.xywhn[0].cpu()
        # print(boxes)
        
        verifyIntersect(frame, boxes)
        detected_frames.append(frame)
        # print("Frame number:", i)
        # i += 1
        cv2.imshow('Frame', frame)
        # results.show()
        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    else:
        break

for img in detected_frames:
        out.write(img)

out.release()
cap.release()
 
