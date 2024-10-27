import cv2
import torch
import numpy as np
import threading
import queue
import torch.nn.functional as F
from torchvision import transforms
import traceback
from utils.face_utils import align_face
from utils.model_utils import load_face_recognition_model
from ultralight import UltraLightDetector

class FaceRecognitionPipeline:
    def __init__(self):
        self.face_detector = UltraLightDetector()
        self.model, self.device = load_face_recognition_model()
        saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
        self.saved_embeddings = [torch.tensor(e).to(self.device) for e in saved_data['embeddings']]  # Convert once
        self.saved_labels = saved_data['labels']
        
        self.align_queue = queue.Queue(maxsize=10)
        self.recog_queue = queue.Queue(maxsize=10)
        
        self.lock = threading.Lock()
        self.results = {}
        self.next_id = 0  # Needs to be accessed under lock to avoid race conditions
        self.frame_count = 0
        self.running = True
        
        # Start worker threads
        self.alignment_thread = threading.Thread(target=self._alignment_worker)
        self.recognition_thread = threading.Thread(target=self._recognition_worker)
        self.alignment_thread.daemon = True
        self.recognition_thread.daemon = True
        self.alignment_thread.start()
        self.recognition_thread.start()

    def _alignment_worker(self):
        while self.running:
            try:
                data = self.align_queue.get()  # Block until data is available
                if data is None:
                    continue
                    
                frame, box, face_id = data
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                aligned_face = align_face(face_img)
                
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))
                
                self.align_queue.task_done()
            except Exception as e:
                print(f"Alignment error: {e}")
                traceback.print_exc()

    def _recognition_worker(self):
        while self.running:
            try:
                data = self.recog_queue.get()
                if data is None:
                    continue
                    
                aligned_face, face_id = data
                face_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])(aligned_face).unsqueeze(0).to(self.device)
                
                embedding = self.model(face_tensor).detach()
                
                # Pre-converted embeddings are already on the device
                similarities = [F.cosine_similarity(saved_embedding, embedding).item() 
                                for saved_embedding in self.saved_embeddings]
                
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                with self.lock:
                    if face_id in self.results:
                        self.results[face_id].update({
                            'name': self.saved_labels[best_idx] if best_score > 0.41 else "Unknown",
                            'similarity': best_score,
                            'last_seen': self.frame_count
                        })
                
                self.recog_queue.task_done()
            except Exception as e:
                print(f"Recognition error: {e}")
                traceback.print_exc()

    def process_frame(self, frame):
        # Check for proper frame type
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        self.frame_count += 1
        result_frame = frame  # No need to copy unless we modify
        
        current_faces = set()
        
        boxes, _ = self.face_detector.detect_one(frame)
        for box in boxes:
            face_id = next((fid for fid, data in self.results.items() 
                          if np.linalg.norm(np.array(data['box']) - box) < 50), 
                         f"face_{self.next_id}")
            
            with self.lock:
                if face_id not in self.results:
                    self.results[face_id] = {
                        'name': "Unknown",
                        'similarity': 0.0,
                        'box': box,
                        'last_seen': self.frame_count
                    }
                    self.next_id += 1
                
                self.results[face_id]['box'] = box
                self.results[face_id]['last_seen'] = self.frame_count
            
            # Queue for processing every 5th frame
            if self.frame_count % 5 == 0:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass
            
            # Draw results
            result = self.results[face_id]
            color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, 
                       f"{result['name']}, score: {result['similarity']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Clean up old results
        if self.frame_count % 30 == 0:
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items()
                              if self.frame_count - data['last_seen'] <= 15}
        
        return result_frame

    def cleanup(self):
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