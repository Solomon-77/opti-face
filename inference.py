import cv2
import torch
import numpy as np
import threading
import torch.nn.functional as F
from utils.face_utils import preprocess_frame
from utils.model_utils import load_face_recognition_model

# Load model, device, and embeddings
model, device = load_face_recognition_model()
saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
saved_embeddings, saved_labels = saved_data['embeddings'], saved_data['labels']

# Shared recognition results
recognition_results = {}
lock = threading.Lock()
next_face_id = 0
recognition_counter = 0
RECOGNITION_INTERVAL = 5

# Recognize face by comparing embeddings
def recognize_face(embedding):
    embedding = embedding.to(device)
    similarities = [F.cosine_similarity(torch.tensor(e).to(device), embedding).item() for e in saved_embeddings]
    best_idx = np.argmax(similarities)
    best_match = saved_labels[best_idx] if similarities[best_idx] > 0.45 else "Unknown"
    return best_match, similarities[best_idx]

# Find the best-matching face by IOU
def find_face(new_box):
    return next((face_id for face_id, data in recognition_results.items() if iou(new_box, data['box']) > 0.5), None)

# IOU calculation
def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0

# Process recognition in a separate thread
def process_recognition(face_tensor, face_id):
    embedding = model(face_tensor.to(device)).detach()
    name, similarity = recognize_face(embedding)
    with lock:
        if face_id in recognition_results:
            recognition_results[face_id].update({'name': name, 'similarity': similarity, 'last_seen': recognition_counter})

# Main video capture and face detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    recognition_counter += 1
    current_faces = set()

    for face_tensor, (x1, y1, x2, y2) in preprocess_frame(frame):
        box = (x1, y1, x2, y2)
        face_id = find_face(box) or f"face_{next_face_id}"
        if face_id not in recognition_results:
            recognition_results[face_id] = {'name': "Unknown", 'similarity': 0.0, 'box': box, 'last_seen': recognition_counter}
            next_face_id += 1

        current_faces.add(face_id)
        recognition_results[face_id]['box'] = box

        if recognition_counter % RECOGNITION_INTERVAL == 0:
            threading.Thread(target=process_recognition, args=(face_tensor, face_id)).start()

        result = recognition_results[face_id]
        color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{result['name']}, score: {result['similarity']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Clean up old results
    if recognition_counter % 10 == 0:
        with lock:
            recognition_results = {fid: data for fid, data in recognition_results.items() if recognition_counter - data['last_seen'] <= 5}

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
