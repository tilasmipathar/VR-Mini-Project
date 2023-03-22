import cv2
import torch
import torchvision.models.detection as tv
import torchvision.transforms as T
import numpy as np
from sort.sort import Sort

# Load FasterRCNN_ResNet50_FPN_V2 model
model = tv.fasterrcnn_resnet50_fpn_v2(weights=tv.faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# Create SORT tracker
tracker = Sort(max_age=10000, min_hits=1)

# Load video
cap = cv2.VideoCapture('test2.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

frame_count = 0
cars_count=0
car_ids=[]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count+=1

    # Apply FRCNN
    with torch.no_grad():
        results = model(frame)

    # change to SORT format
    boxes = []
    for b, s, l in zip(np.array(results[0]['boxes'].to('cpu')), np.array(results[0]['scores'].to('cpu')), np.array(results[0]['labels'].to('cpu'))):
        if(l==3): #Check for Car class
            b = [b[0], b[1], b[2], b[3]]
            boxes.append(b)
    boxes = np.array(boxes)

    # update SORT tracker
    ids = tracker.update(boxes)
    
    cars_curr_count=0
    car_curr_ids=[]
    # Draw the tracking results on the frame
    for t in ids:
        xmin, ymin, xmax, ymax, track_id = t
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.putText(frame, str(track_id), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if(track_id not in car_ids and int(ymin)>300): # filtering out the range with int(ymin)>300 due to clustering of cars
            cars_count+=1
            car_ids.append(track_id)
        if(track_id not in car_curr_ids ):
            cars_curr_count+=1
            car_curr_ids.append(track_id)
    cv2.line(frame, (20,65+ (40)), (227,65+ (40)), [85,45,255], 30)
    cv2.putText(frame,"total Cars: " + str(cars_count), (20,100),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.5,color=(255,255,255),thickness=4)
    
    # Write into output video
    wr_arr = np.asarray(frame)
    for i in range(len(wr_arr)):
        output.write(wr_arr[i])
    output.release()

    # Show the individual frames with counts
    cv2.imshow('output', frame)
    if cv2.waitKey(30) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
