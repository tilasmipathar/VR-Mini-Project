import cv2
import numpy as np
import torch
import torchvision.models.detection as tv
import torchvision.transforms as T
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections
from deep_sort.application_util import preprocessing

# Load the model
model = tv.fasterrcnn_resnet50_fpn_v2(weights=tv.faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()

# Set up the tracker
max_cosine_distance=0.5
nn_budget=None
nms_max_overlap=1.5

weights_path = 'mars-small128.pb'
encoder = generate_detections.create_box_encoder(weights_path, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Loading the video
video_path = 'test3.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Initialising the counters
count_cars=0
prev_ids=[]

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # use FRCNN model
    with torch.no_grad():
        results = model(frame)

    detections = []
    boxes = [] 
    scores = []
    for b, s, l in zip(np.array(results[0]['boxes']), np.array(results[0]['scores']), np.array(results[0]['labels'])):
        if(l==3): #check for car class
            boxes.append([b[0], b[1], b[2]-b[0], b[3]-b[1]])
            scores.append(s)

    boxes = np.array(boxes) 
    scores = np.array(scores)
    features = encoder(frame, boxes)
    features = np.array(features)
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]
    
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])

    #Applying non-maximum supression to eliminate multiple boxes for same image
    filtered_index = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in filtered_index]
    
    # Update the tracker
    tracker.predict()
    tracker.update(detections)

    # Draw the tracking results on the frame
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        if(track.track_id not in prev_ids):
            count_cars+=1
            prev_ids.append(track.track_id)
    cv2.putText(frame,"Cars: " + str(count_cars), (20,100),cv2.FONT_HERSHEY_COMPLEX,fontScale=1.5,color=(200,0,255),thickness=4)

    # Write into output video
    wr_arr = np.asarray(frame)
    for i in range(len(wr_arr)):
        out.write(wr_arr[i])
    out.release()
    
    # Display the output frame
    cv2.imshow('output', frame)
    if cv2.waitKey(30) & 0xFF==27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()