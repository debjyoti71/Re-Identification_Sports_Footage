# 🏟️ Player Re-Identification in Sports Footage

This project addresses the challenge of **player re-identification** in football videos. Specifically, the goal is to ensure that **the same player maintains a consistent ID** even when:

* Appearing from **different camera views**, or
* Re-entering the frame after leaving in a **single moving camera feed**.

---

## 🌟 Task 1: Cross-Camera Player Mapping

### 📌 Question

> Given two clips — `broadcast.mp4` and `tacticam.mp4` — of the same game from different camera angles, **match the players** so that each retains a consistent `player_id` across both views.

### 📅 Input Videos

* `broadcast.mp4`
* `tacticam.mp4`

### ⚠️ Problem Faced

* **All players wear the same jersey**, making **appearance-based re-ID extremely difficult**.
* Torch-based features often yield **high similarity between different players**.

### 💡 My Solution

* Instead of relying only on appearance, I used:

  * **Player location** over time (centroids and movement trajectories).
  * Matching based on **frame-wise proximity and motion similarity**.

### 🧠 Code

* `task_1.py`: Detect players with YOLOv8 → track with DeepSort → extract features with Torchreid → match across feeds using location & feature similarity.

### 🎨 Output

* `output/global_tracking_comparison.mp4`: A **side-by-side video** showing both camera views with synchronized player IDs.

---

## 🌟 Task 2: Re-Identification in a Single Moving Camera Feed

### 📌 Question

> Given a 15-second football video (`15sec_input_720p.mp4`), assign consistent IDs to each player, even if they leave and **re-enter** the frame later.

### 📅 Input Video

* `15sec_input_720p.mp4`

### ⚠️ Problem Faced

* The **camera is moving**, which **breaks DeepSort’s tracking**.
* Player re-entry leads to inconsistent ID assignments.

### 💡 My Solution

* Processed the video with **less frame density**, reducing motion blur.
* Allowed DeepSort to **better preserve identities** by spacing detections.
* Did **not apply any external ID re-matching** for simplicity.

### 🧠 Code

* `task_2.py`: Runs YOLOv8 + DeepSort and saves output video with IDs.

### 🎨 Output

* `output/15sec_input_players.mp4`: Video with player tracking using DeepSort.

---

## 📆 Repository Structure

```
📁 Re-Identification_Sports_Footage/
├── .gitignore
├── cropped_players_broadcast/
│   ├── ball/
│   ├── distances/
│   ├── goalkeeper/
│   ├── player/
│   └── referee/
├── cropped_players_tacticam/
│   ├── ball/
│   ├── distances/
│   ├── goalkeeper/
│   ├── player/
│   └── referee/
├── model/                 # (ignored, stores saved models)
├── model_summery.py
├── output/
│   ├── 15sec_input_players.mp4
│   ├── broadcast_players.mp4
│   ├── features_broadcast.npy
│   ├── features_tacticam.npy
│   ├── global_tracking_comparison.mp4
│   ├── player_id_mapping.txt
│   └── tacticam_players.mp4
├── player_id_mapping.csv
├── player_id_mapping.json
├── task_1.py
├── task_1_2.py
├── task_2.py
└── videos/
    ├── 15sec_input_720p.mp4
    ├── broadcast.mp4
    └── tacticam.mp4
```

---

## 🔧 Setup Instructions

1. **Clone this repository**

```bash
git clone https://github.com/debjyoti71/Re-Identification_Sports_Footage.git
cd Re-Identification_Sports_Footage
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install torch torchvision ultralytics opencv-python tqdm torchreid deep_sort_realtime
```

3. **Download input model and videos**

Place the following in the `/videos/` directory:

* `broadcast.mp4`
* `tacticam.mp4`
* `15sec_input_720p.mp4`

Place the model `yolov8.pt` in the root directory:

* [📅 YOLOv8 Model](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAM_ScrePVcMD/view)

Assignment Folder:

* [📁 Assignment Files](https://drive.google.com/drive/folders/1Nx6Hn0UUI6L-6i8WknXd4Cv2c3VjZTP?usp=sharing)

4. **Run Task 1**

```bash
python task_1.py
```

5. **Run Task 2**

```bash
python task_2.py
```

Output videos will be saved in the `/output/` directory.

---

## 📩 Contact

**Debjyoti Ghosh**
B.Tech, UEM Kolkata
📧 Email: [debjyoti1ghosh@gmail.com](mailto:debjyoti1ghosh@gmail.com)
🔗 GitHub: [github.com/debjyoti71](https://github.com/debjyoti71)

---

