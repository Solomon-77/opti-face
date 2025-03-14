import cv2
import torch
import numpy as np
import threading
import queue
import os
from torch.nn.functional import cosine_similarity
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform

class FaceRecognitionPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        face_database_dir = './face_database/'
        self.saved_embeddings = []
        self.saved_labels = []
        
        for filename in os.listdir(face_database_dir):
            if filename.endswith('.npz'):
                person_name = os.path.splitext(filename)[0]
                npz_path = os.path.join(face_database_dir, filename)
                data = np.load(npz_path)
                embeddings = data['embeddings']
                
                for embedding in embeddings:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    self.saved_labels.append(person_name)
                    
        self.align_queue = queue.Queue(maxsize=5)
        self.recog_queue = queue.Queue(maxsize=5)
        self.results = {}
        self.next_id = 0
        self.frame_count = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Start alignment and recognition worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def _alignment_worker(self):
        """Thread worker for aligning faces before recognition."""
        while self.running:
            try:
                frame, box, face_id = self.align_queue.get()
                if frame is None:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                    
                aligned_face = align_face(face_crop)
                
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))
                
                self.align_queue.task_done()
            except Exception as e:
                print(f"Alignment error: {e}")

    def _recognition_worker(self):
        """Thread worker for face recognition."""
        while self.running:
            try:
                aligned_face, face_id = self.recog_queue.get()
                if aligned_face is None:
                    continue
                
                face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
                embedding = self.model(face_tensor).detach()
                
                similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                with self.lock:
                    if face_id in self.results:
                        self.results[face_id].update({
                            'name': self.saved_labels[best_idx] if best_score > 0.5 else "Unknown",
                            'similarity': best_score,
                            'last_seen': self.frame_count
                        })
                
                self.recog_queue.task_done()
            except Exception as e:
                print(f"Recognition error: {e}")

    def process_frame(self, frame):
        """Detect, align, and recognize faces in a video frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        self.frame_count += 1
        boxes, _ = detect_faces(frame)
        
        if len(boxes) == 0:
            return frame
            
        for box in boxes:
            face_id = next(
                (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50), 
                f"face_{self.next_id}"
            )
            
            with self.lock:
                if face_id not in self.results:
                    self.results[face_id] = {
                        'name': "Unknown",
                        'similarity': 0.0,
                        'box': box,
                        'last_seen': self.frame_count
                    }
                    self.next_id += 1
                
                self.results[face_id].update({
                    'box': box,
                    'last_seen': self.frame_count
                })
            
            if self.frame_count % 5 == 0:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass
            
            result = self.results[face_id]
            color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{result['name']}, score: {result['similarity']:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if self.frame_count % 30 == 0:
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
        
        return frame

    def cleanup(self):
        """Stop the pipeline and release resources."""
        self.running = False
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