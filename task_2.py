from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
SKIP_FRAMES = 2  # Process every nth frame

# Load YOLOv8 model
print("Loading YOLO model...")
model = YOLO("model/best.pt")
print("Model loaded.")

class_names = model.names  # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
print("Detected classes:", class_names)

# Assign color for each class
class_colors = {
    0: (0, 0, 255),      # Ball - Red
    1: (255, 0, 0),      # Goalkeeper - Blue
    2: (0, 255, 0),      # Player - Green
    3: (0, 255, 255)     # Referee - Yellow
}

# Define per-class score thresholds
class_score_thresholds = {
    0: 0.2,   # ball
    1: 0.4,   # goalkeeper
    2: 0.40,  # player
    3: 0.3    # referee
}

# Video input paths
video_paths = {
    "15sec_input": "videos/15sec_input_720p.mp4",
    
}

# DeepSORT tracker per video
tracker_dict = {
    "15sec_input": DeepSort(max_age=10),
    
}

# Output folder
os.makedirs("output", exist_ok=True)

for cam_name, video_path in video_paths.items():
    print(f"\nProcessing: {cam_name}")
    cap = cv2.VideoCapture(video_path)
    crop_base_dir = f"cropped_players_{cam_name}"
    os.makedirs(crop_base_dir, exist_ok=True)

    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(
        f"output/{cam_name}_players.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker = tracker_dict[cam_name]
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % SKIP_FRAMES != 0:
            frame_num += 1
            continue

        results = model(frame, verbose=False)[0]
        detections = []
        class_ids = []
        bboxes_xywh = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            class_id = int(class_id)
            score = float(score)

            if score < class_score_thresholds.get(class_id, 0.9):
                continue

            w, h = int(x2 - x1), int(y2 - y1)
            bbox = [int(x1), int(y1), w, h]
            detections.append((bbox, score, None))  # (xywh, conf, None)
            class_ids.append(class_id)
            bboxes_xywh.append(bbox)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Match class with detection bbox
            matched_class = None
            matched_score = 0
            for i, (x, y, w, h) in enumerate(bboxes_xywh):
                if abs(x - l) < 20 and abs(y - t) < 20:
                    matched_class = class_ids[i]
                    matched_score = detections[i][1]
                    break

            if matched_class is None:
                matched_class = 2  # fallback to player
                matched_score = 0.0

            class_name = class_names[matched_class].replace(" ", "_")
            color = class_colors.get(matched_class, (255, 255, 255))
            label = f"{class_name} {track_id} ({matched_score:.2f})"

            # Draw label on frame
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Crop and save
            crop = frame[max(0, t):min(b, height), max(0, l):min(r, width)]
            if crop.size == 0:
                continue

            save_dir = os.path.join(crop_base_dir, class_name, f"id_{track_id}")
            os.makedirs(save_dir, exist_ok=True)
            crop_filename = f"{class_name}_{frame_num:05d}_{matched_score:.2f}.jpg"
            cv2.imwrite(os.path.join(save_dir, crop_filename), crop)

        # Show and write
        cv2.imshow(f"{cam_name} view", frame)
        writer.write(frame)
        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠️ Interrupted by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyWindow(f"{cam_name} view")
    print(f"✅ Saved: output/{cam_name}_players.mp4")

cv2.destroyAllWindows()
print("\n🎯 All videos processed. Crops saved in class-wise ID folders.")