import cv2
import torch
import numpy as np
import os
import time
from torch.nn.functional import cosine_similarity
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform

class FaceRecognitionPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        face_database_dir = './face_database/'
        self.saved_embeddings = []
        self.saved_labels = []
        
        # FPS calculation variables
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
        # Load face database
        for filename in os.listdir(face_database_dir):
            if filename.endswith('.npz'):
                person_name = os.path.splitext(filename)[0]
                npz_path = os.path.join(face_database_dir, filename)
                data = np.load(npz_path)
                embeddings = data['embeddings']
                
                for embedding in embeddings:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    self.saved_labels.append(person_name)

    def recognize_face(self, aligned_face):
        """Recognize a face from an aligned image."""
        face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
        embedding = self.model(face_tensor).detach()
        
        similarities = [cosine_similarity(saved_emb, embedding).item() 
                       for saved_emb in self.saved_embeddings]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        name = self.saved_labels[best_idx] if best_score > 0.5 else "Unknown"
        return name, best_score

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        # Calculate FPS
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.fps = 0.9 * self.fps + 0.1 * fps  # Smooth FPS using exponential moving average
        self.prev_frame_time = self.curr_frame_time
        
        # Detect faces
        boxes, _ = detect_faces(frame)
        
        if len(boxes) == 0:
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
            
        # Process each detected face
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
                
            # Align and recognize face
            aligned_face = align_face(face_crop)
            if aligned_face is not None:
                name, similarity = self.recognize_face(aligned_face)
                
                # Draw results
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}, score: {similarity:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def cleanup(self):
        """Release resources."""
        cv2.destroyAllWindows()

def main():
    pipeline = FaceRecognitionPipeline()
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = pipeline.process_frame(frame)
            if processed is not None:
                cv2.imshow('Face Recognition', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.cleanup()
        cap.release()

if __name__ == "__main__":
    main()