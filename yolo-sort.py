import cv2
import torch
import numpy as np
from sort.sort import Sort

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create SORT tracker
tracker = Sort()

# load video
cap = cv2.VideoCapture('test6.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#Initialise counter for cars
cars_count=0
car_ids=[]
# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLOv5
    results = model(frame)
    bboxes = results.xyxy[0].cpu().numpy()
    
    # Filter the detected boxes to only include cars
    car_bboxes = []
    for bbox in bboxes:
        if bbox[5] == 2: # Class 2 means cars as per COCO dataset
            car_bboxes.append(bbox)
    
    # update SORT tracker
    tracks = tracker.update(np.array(car_bboxes))
    
    cars_curr_count=0
    car_curr_ids=[]
    # Draw the tracking results on the frame
    for track in tracks:
        xmin, ymin, xmax, ymax, track_id = track
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if(track_id not in car_ids and int(ymin)>300): # filtering out the range with int(ymin)>300 due to clustering of cars
            cars_count+=1
            car_ids.append(track_id)
        if(track_id not in car_curr_ids ):
            cars_curr_count+=1
            car_curr_ids.append(track_id)
    cv2.putText(frame,"total Cars: " + str(cars_count), (20,100),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.5,color=(0,0,0),thickness=4)
    cv2.putText(frame,"Cars currently in frame: " + str(cars_curr_count), (20,150),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.5,color=(0,0,0),thickness=4)
    
    wr_arr = np.asarray(frame)
    for i in range(len(wr_arr)):
        output.write(wr_arr[i])
    output.release()

    # Show the individual frames with counts
    cv2.imshow('Tracking', frame)
    

cap.release()
cv2.destroyAllWindows()
