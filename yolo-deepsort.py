import cv2
import numpy as np
import torch
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections
from deep_sort.application_util import preprocessing

# Load YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create DeepSORT tracker
max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 1.5

model_filename = 'mars-small128.pb'
encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Load video
video_path = 'test2.mp4'
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = int(video.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)


# Initialise car tracker
prev_ids=[]
count_cars=0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, size)
    
    # Apply yolo model
    results = model(frame)
    yolores = results.pandas().xyxy[0]

    # Change to DeepSORT format
    detections = []
    b, s, n = [], [], []
    for i, entry in yolores.iterrows():
        objClass = entry['name']
        if(objClass == "car"):
            n.append(objClass)
            scores.append(entry['confidence'])
            bbox = [int(entry['xmin']), int(entry['ymin']), int(entry['xmax'])-int(entry['xmin']), int(entry['ymax'])-int(entry['ymin'])]
            boxes.append(bbox)

    boxes = np.array(boxes) 
    names = np.array(names)
    scores = np.array(scores)
    features = np.array(encoder(frame, boxes))
    detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    
    #Applying non-maximum supression to eliminate multiple boxes for same image
    filtered_index = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in filtered_index]
    
    # Update tracker
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
    cv2.putText(frame, "Cars: " + str(count_cars), (20,100),cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,0),thickness=4)
    
    # Write into output video
    out.write(np.asarray(frame))

    # Display the output frame
    cv2.imshow('FRAME', frame)

video.release()
cv2.destroyAllWindows()