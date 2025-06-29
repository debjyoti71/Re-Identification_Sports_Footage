import os
import cv2
import torch
import numpy as np
import random
import torchreid
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import models, transforms
from deep_sort_realtime.deepsort_tracker import DeepSort


class YOLODeepSortTracker:
    def __init__(self, model_path="model/best.pt"):
        print("🔍 Loading YOLOv8 model...")
        self.yolo = YOLO(model_path)
        self.class_names = self.yolo.names
        print("✅ Model loaded.")

        self.class_colors = {
            0: (0, 0, 255),     # Ball - Red
            1: (255, 0, 0),     # Goalkeeper - Blue
            2: (0, 255, 0),     # Player - Green
            3: (0, 255, 255)    # Referee - Yellow
        }

        self.class_score_thresholds = {
            0: 0.2,
            1: 0.5,   # Goalkeeper
            2: 0.7,   # Player
            3: 0.3
        }

        self.trackers = {
            "broadcast": DeepSort(max_age=12),
            "tacticam": DeepSort(max_age=12)
        }

        # Memory to store stable class for a track ID
        self.track_role_memory = {}

    def process_video(self, cam_name, video_path):
        print(f"\n🎥 Processing: {cam_name}")
        cap = cv2.VideoCapture(video_path)
        crop_base_dir = f"cropped_players_{cam_name}"
        os.makedirs(crop_base_dir, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = cv2.VideoWriter(
            f"output/{cam_name}_players.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        tracker = self.trackers[cam_name]
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.yolo(frame, verbose=False)[0]
            detections, class_ids, bboxes_xywh = [], [], []

            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                class_id = int(class_id)
                score = float(score)

                if score < self.class_score_thresholds.get(class_id, 0.9):
                    continue

                w, h = int(x2 - x1), int(y2 - y1)
                bbox = [int(x1), int(y1), w, h]
                detections.append((bbox, score, None))
                class_ids.append(class_id)
                bboxes_xywh.append(bbox)

            tracks = tracker.update_tracks(detections, frame=frame)

            # 💡 Prepare for distance calculation
            player_centroids = []
            player_ids = []

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())

                # --- Match class based on detection box ---
                matched_class, matched_score = None, 0
                for i, (x, y, w, h) in enumerate(bboxes_xywh):
                    if abs(x - l) < 20 and abs(y - t) < 20:
                        matched_class = class_ids[i]
                        matched_score = detections[i][1]
                        break

                # Fallback class
                if matched_class is None:
                    matched_class = 2
                    matched_score = 0.0

                # Position-based goalkeeper correction
                if matched_class == 2:
                    if l < width * 0.1 or r > width * 0.9:
                        matched_class = 1

                # Memory override
                if track_id in self.track_role_memory:
                    matched_class = self.track_role_memory[track_id]
                else:
                    self.track_role_memory[track_id] = matched_class

                class_name = self.class_names[matched_class].replace(" ", "_")
                color = self.class_colors.get(matched_class, (255, 255, 255))
                label = f"{class_name} {track_id} ({matched_score:.2f})"

                # Draw on frame
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Save cropped player image
                crop = frame[max(0, t):min(b, height), max(0, l):min(r, width)]
                if crop.size != 0:
                    save_dir = os.path.join(crop_base_dir, class_name, f"id_{track_id}")
                    os.makedirs(save_dir, exist_ok=True)
                    crop_filename = f"{class_name}_{frame_num:05d}_{matched_score:.2f}.jpg"
                    cv2.imwrite(os.path.join(save_dir, crop_filename), crop)

                # Collect centroid if player
                if matched_class == 2:
                    cx = (l + r) // 2
                    cy = (t + b) // 2
                    player_centroids.append((cx, cy))
                    player_ids.append(track_id)

            # 🧮 Compute pairwise distances between players
            if len(player_centroids) >= 2:
                distances = []
                for i in range(len(player_centroids)):
                    for j in range(i + 1, len(player_centroids)):
                        id1, id2 = player_ids[i], player_ids[j]
                        (x1, y1), (x2, y2) = player_centroids[i], player_centroids[j]
                        dist = np.linalg.norm(np.array([x1 - x2, y1 - y2]))
                        distances.append((id1, id2, dist))

                distance_dir = os.path.join(crop_base_dir, "distances")
                os.makedirs(distance_dir, exist_ok=True)
                distance_file = os.path.join(distance_dir, f"player_distances_frame_{frame_num:05d}.csv")
                with open(distance_file, "w") as f:
                    f.write("Player1_ID,Player2_ID,Distance\n")
                    for id1, id2, dist in distances:
                        f.write(f"{id1},{id2},{dist:.2f}\n")

            # Show and save frame
            cv2.imshow(f"{cam_name} view", frame)
            writer.write(frame)
            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⛔ Interrupted by user.")
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()



class FeatureExtractor:
    def __init__(self):
        print("🧠 Loading torchreid model...")
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',  # use default pretrained
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ Ready.")

    def extract_features(self, player_dir):
        features_dict = {}

        for player_id in os.listdir(player_dir):
            id_path = os.path.join(player_dir, player_id)
            if not os.path.isdir(id_path):
                continue

            player_features = []

            for img_name in os.listdir(id_path):
                img_path = os.path.join(id_path, img_name)
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                try:
                    feat = self.extractor(img_path).squeeze()
                    player_features.append(feat.cpu().numpy())

                except Exception as e:
                    print(f"⚠️ Error on {img_path}: {e}")

            features_dict[player_id] = player_features

        return features_dict
    
class CrossCameraMatcher:
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.global_id_counter = 0
        self.global_tracks = {}  # global_id: {'features': [...], 'appearances': [(cam, id)]}

    def match_and_update(self, cam_name, features_dict):
        id_to_global = {}

        for local_id, features in features_dict.items():
            mean_feature = np.mean(features, axis=0, keepdims=True)
            matched = False

            for global_id, data in self.global_tracks.items():
                global_features = np.mean(data['features'], axis=0, keepdims=True)
                sim = cosine_similarity(mean_feature, global_features)[0][0]

                if sim > self.similarity_threshold:
                    self.global_tracks[global_id]['features'].extend(features)
                    self.global_tracks[global_id]['appearances'].append((cam_name, local_id))
                    id_to_global[local_id] = global_id
                    matched = True
                    break

            if not matched:
                new_id = self.global_id_counter
                self.global_id_counter += 1
                self.global_tracks[new_id] = {
                    'features': features,
                    'appearances': [(cam_name, local_id)]
                }
                id_to_global[local_id] = new_id

        return id_to_global

    def print_summary(self):
        print("\n🧾 Global ID Assignment Summary:")
        for gid, data in self.global_tracks.items():
            print(f"Global ID {gid}: {data['appearances']}")


class MultiCamVisualizer:
    def __init__(self, video_paths, global_map, crop_dirs):
        self.video_paths = video_paths
        self.global_map = global_map  # {"broadcast": {"id_1": global_id, ...}, ...}
        self.crop_dirs = crop_dirs

    def visualize(self):
        caps = {cam: cv2.VideoCapture(path) for cam, path in self.video_paths.items()}
        widths = [int(caps[cam].get(cv2.CAP_PROP_FRAME_WIDTH)) for cam in caps]
        heights = [int(caps[cam].get(cv2.CAP_PROP_FRAME_HEIGHT)) for cam in caps]
        max_height = max(heights)
        total_width = sum(widths)

        writer = cv2.VideoWriter(
            "output/global_tracking_comparison.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (total_width, max_height)
        )

        while True:
            frames = []
            for cam in self.video_paths:
                ret, frame = caps[cam].read()
                if not ret:
                    return

                label_img_dir = self.crop_dirs[cam]
                overlay = frame.copy()
                if os.path.exists(label_img_dir):
                    for id_dir in os.listdir(label_img_dir):
                        local_id = id_dir.split("_")[-1]
                        global_id = self.global_map.get(cam, {}).get(local_id, None)
                        if global_id is not None:
                            label = f"GID: {global_id}"
                            cv2.putText(overlay, label, (10, 30 + 30 * int(local_id)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frames.append(overlay)

            combined = np.hstack(frames)
            writer.write(combined)
            cv2.imshow("Global Tracking", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in caps.values():
            cap.release()
        writer.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    video_paths = {
        "broadcast": "videos/broadcast.mp4",
        "tacticam": "videos/tacticam.mp4"
    }

    crop_dirs = {
        "broadcast": "cropped_players_broadcast/Player",
        "tacticam": "cropped_players_tacticam/Player"
    }

    # Step 1: Run detection and tracking
    tracker = YOLODeepSortTracker()
    for cam_name, path in video_paths.items():
        tracker.process_video(cam_name, path)

    # Step 2: Extract features
    extractor = FeatureExtractor()

    all_features = {}
    for cam_name in crop_dirs:
        print(f"\n🔍 Extracting features from: {crop_dirs[cam_name]}")
        features = extractor.extract_features(crop_dirs[cam_name])
        all_features[cam_name] = features
        np.save(f"output/features_{cam_name}.npy", features)

    # Step 3: Cross-camera matching
    matcher = CrossCameraMatcher(similarity_threshold=0.8)
    global_map = {}
    for cam_name in all_features:
        id_map = matcher.match_and_update(cam_name, all_features[cam_name])
        global_map[cam_name] = {k.replace("id_", ""): v for k, v in id_map.items()}  # Clean id names

    matcher.print_summary()

    # Step 4: Global visualization
    visualizer = MultiCamVisualizer(video_paths, global_map, crop_dirs)
    print("\n🎬 Showing global tracking video. Press 'q' to quit.")
    visualizer.visualize()

