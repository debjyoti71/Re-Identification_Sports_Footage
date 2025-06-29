from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Load YOLOv8 model
print("Loading YOLO model...")
model = YOLO("model/best.pt")
print("Model loaded.")

# Setup video paths
video_paths = {
    "broadcast": "videos/broadcast.mp4",
    "tacticam": "videos/tacticam.mp4"
}

# Deep SORT trackers
tracker_dict = {
    "broadcast": DeepSort(max_age=30),
    "tacticam": DeepSort(max_age=30)
}

# Output folder
os.makedirs("output", exist_ok=True)

# model summary
print("Model Summary:")
print(model.model)  

print("Class labels in the model:")
print(model.names)

