import cv2
import torch
import numpy as np
import torch.nn.functional as F
from utils.face_utils import preprocess_frame
from utils.model_utils import load_face_recognition_model

# Load the face recognition model
model = load_face_recognition_model()

# Load saved embeddings and labels
saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
saved_embeddings, saved_labels = saved_data['embeddings'], saved_data['labels']

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
        
        text, color = (f"{recognized_person}, score: {similarity:.2f}", (0, 255, 0)) if recognized_person != "Unknown" else ("Unknown", (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()