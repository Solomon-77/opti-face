import cv2
import torch
import numpy as np
import threading
import queue
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from utils.face_utils import align_face, load_face_recognition_model
from utils.scrfd import FaceDetector

class FaceRecognitionPipeline:
    def __init__(self):
        # Initialize face detector, recognition model, and load embeddings
        self.face_detector = FaceDetector(onnx_file='checkpoints/scrfd_500m.onnx')
        self.model, self.device = load_face_recognition_model()
        
        # Load saved embeddings and labels
        saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
        self.saved_embeddings = [torch.tensor(e).to(self.device) for e in saved_data['embeddings']]
        self.saved_labels = saved_data['labels']
        
        # Initialize queues, results storage, and threading locks
        self.align_queue = queue.Queue(maxsize=10)
        self.recog_queue = queue.Queue(maxsize=10)
        self.results = {}
        self.next_id = 0
        self.frame_count = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Start alignment and recognition worker threads
        for worker in [self._alignment_worker, self._recognition_worker]:
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()

    def _alignment_worker(self):
        """Thread worker for aligning faces before recognition."""
        while self.running:
            try:
                frame, box, face_id = self.align_queue.get()
                if frame is None:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                aligned_face = align_face(frame[y1:y2, x1:x2])
                
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))
                
                self.align_queue.task_done()
            except Exception as e:
                print(f"Alignment error: {e}")

    def _recognition_worker(self):
        """Thread worker for face recognition using aligned faces."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        
        while self.running:
            try:
                aligned_face, face_id = self.recog_queue.get()
                if aligned_face is None:
                    continue
                
                # Transform and get embedding
                face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
                embedding = self.model(face_tensor).detach()
                
                # Calculate cosine similarity with saved embeddings
                similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                
                # Update recognition result
                with self.lock:
                    if face_id in self.results:
                        self.results[face_id].update({
                            'name': self.saved_labels[best_idx] if best_score > 0.45 else "Unknown",
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
        det, _ = self.face_detector.detect(frame, thresh=0.5, input_size=(640, 640))
        
        if det is None or len(det) == 0:
            return frame
            
        boxes = det[:, :4]
        
        for box in boxes:
            # Assign unique face ID
            face_id = next(
                (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50), 
                f"face_{self.next_id}"
            )
            
            # Initialize or update face recognition result
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
            
            # Queue frame for alignment every 5 frames
            if self.frame_count % 5 == 0:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass
            
            # Draw detection box and label on frame
            result = self.results[face_id]
            color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{result['name']}, score: {result['similarity']:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Clean up stale results every 30 frames
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
    
    # optional
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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