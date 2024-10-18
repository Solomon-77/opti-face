import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from face_alignment import align
from utils.face_recognition import get_model
from utils.face_detection import FaceDetector
import torch.nn.functional as F

# Load face detection and recognition models
face_detector = FaceDetector('./models/face_detection/RFB-320.mnn')
model = get_model("edgeface_xs_gamma_06")
model.load_state_dict(torch.load('models/face_recognition/edgeface_xs_gamma_06.pt', map_location='cpu'))
model.eval()

# Load saved embeddings and labels
saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
saved_embeddings, saved_labels = saved_data['embeddings'], saved_data['labels']

# Preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_frame(frame):
    boxes, _, _ = face_detector.detect(frame)
    faces = []
    for x1, y1, x2, y2 in boxes.astype(int):
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2)
        face_img = frame[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        aligned_face, _ = align.get_aligned_face(rgb_pil_image=pil_img)
        if aligned_face:
            face_tensor = transform(aligned_face).unsqueeze(0)
            faces.append((face_tensor, (x1, y1, x2, y2)))
    return faces

def recognize_face(embedding):
    best_match, highest_similarity = "Unknown", -1
    embedding = embedding.cpu()
    for saved_embedding, label in zip(saved_embeddings, saved_labels):
        similarity = F.cosine_similarity(torch.tensor(saved_embedding), embedding).item()
        if similarity > highest_similarity:
            highest_similarity, best_match = similarity, label
    return best_match if highest_similarity > 0.40 else "Unknown", highest_similarity

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = preprocess_frame(frame)
    for face_tensor, (x1, y1, x2, y2) in faces:
        embedding = model(face_tensor).detach()
        recognized_person, similarity = recognize_face(embedding)
        text, color = (f"{recognized_person}, Sim: {similarity:.2f}", (0, 255, 0)) if recognized_person != "Unknown" else ("Unknown", (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
