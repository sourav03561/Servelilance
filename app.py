from flask import Flask, Response, render_template, request, jsonify
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO models
gun_model_path = "best.pt"
human_model_path = "yolov5l.pt"
gun_model = YOLO(gun_model_path)  # Load YOLO gun detection model
human_model = torch.hub.load('ultralytics/yolov5', 'custom', path=human_model_path)
human_model.conf = 0.5

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
human_model.to(device)

# Zone and alert parameters
zone = [100, 60, 400, 400]  # Default zone
people_count_threshold = 1  # Default crowd alert threshold
time_threshold_frames = 100
tracks = {}
next_id = 0
max_missed = 30  # Remove track if not seen for 30 frames

# Helper functions
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_in_zone(box, zone):
    if not zone or len(zone) != 4:
        return False  # Or handle the error as appropriate
    bx_cx, bx_cy = box_center(box)
    return (zone[0] <= bx_cx <= zone[2]) and (zone[1] <= bx_cy <= zone[3])

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)

def associate_detections_to_tracks(detections, tracks):
    track_ids = list(tracks.keys())
    if len(track_ids) == 0 or len(detections) == 0:
        return [], list(range(len(detections))), track_ids

    cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
    for i, tid in enumerate(track_ids):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1 - iou(tracks[tid]['box'], det)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_dets, unmatched_tracks = [], [], []

    for j in range(len(detections)):
        if j not in col_ind:
            unmatched_dets.append(j)
    for i in range(len(track_ids)):
        if i not in row_ind:
            unmatched_tracks.append(track_ids[i])

    for r, c in zip(row_ind, col_ind):
        if 1 - cost_matrix[r, c] < 0.3:  # IOU < 0.3
            unmatched_dets.append(c)
            unmatched_tracks.append(track_ids[r])
        else:
            matches.append((track_ids[r], c))
    return matches, unmatched_dets, unmatched_tracks

def detect_guns(frame):
    results = gun_model(frame)
    annotated_frame = results[0].plot()
    return results, annotated_frame

# Video processing
def process_frame():
    global tracks, next_id
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Gun detection
        gun_results, annotated_frame = detect_guns(frame)

        # Human detection
        results = human_model(frame)
        detections = []
        knife = []  # Detect knives
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if cls == 0:  # Person class
                detections.append([int(x1), int(y1), int(x2), int(y2)])
            if cls == 43:  # Knife class
                knife.append([int(x1), int(y1), int(x2), int(y2)])

        # Handle knife detection
        for k in knife:
            x1, y1, x2, y2 = k
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for knife
            cv2.putText(annotated_frame, "KNIFE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Associate detections with tracks
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(detections, tracks)

        for tid, det_idx in matches:
            tracks[tid]['box'] = detections[det_idx]
            if box_in_zone(detections[det_idx], zone):
                tracks[tid]['frames_in_zone'] += 1
            else:
                tracks[tid]['frames_in_zone'] = 0
            tracks[tid]['missed'] = 0

        for det_idx in unmatched_dets:
            new_id = next_id
            next_id += 1
            tracks[new_id] = {
                'box': detections[det_idx],
                'frames_in_zone': 1 if box_in_zone(detections[det_idx], zone) else 0,
                'missed': 0
            }

        for ut in unmatched_tracks:
            tracks[ut]['missed'] += 1

        remove_ids = [t for t in tracks if tracks[t]['missed'] > max_missed]
        for rid in remove_ids:
            del tracks[rid]

        # Alerts
        in_zone_count = sum(1 for t in tracks if box_in_zone(tracks[t]['box'], zone))
        stay_alert = any(tracks[t]['frames_in_zone'] > time_threshold_frames for t in tracks)
        crowd_alert = in_zone_count > people_count_threshold

        # Draw human detections and zone
        for tid, data in tracks.items():
            x1, y1, x2, y2 = data['box']
            color = (0, 255, 0) if box_in_zone(data['box'], zone) else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(annotated_frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"In zone: {in_zone_count}", (zone[0], zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if stay_alert:
            cv2.putText(annotated_frame, "ALERT: Person in zone too long!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if crowd_alert:
            cv2.putText(annotated_frame, "ALERT: Too many people in zone!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_zone', methods=['POST'])
def update_zone():
    global zone, people_count_threshold, time_threshold_frames
    try:
        data = request.json
        new_zone = data.get('zone')
        new_threshold = data.get('people_count_threshold')
        time_threshold_frames = data.get('time_count_threshold')
        # Validate zone values
        if new_zone and len(new_zone) == 4 and all(isinstance(i, (int, float)) for i in new_zone):
            zone = new_zone
        else:
            return jsonify({"error": "Invalid zone values"}), 400

        # Validate threshold value
        if isinstance(new_threshold, int) and new_threshold > 0:
            people_count_threshold = new_threshold
        else:
            return jsonify({"error": "Invalid people count threshold"}), 400

        return jsonify({"message": "Zone and threshold updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)