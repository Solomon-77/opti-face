import cv2
import torch
import numpy as np
import threading
import queue
import time
import torch.nn.functional as F
from utils.face_utils import align_face, get_face_landmarks
from utils.model_utils import load_face_recognition_model
from ultralight import UltraLightDetector

class FaceRecognitionPipeline:
    def __init__(self, max_queue_size=10):
        # Initialize detectors and model
        self.face_detector = UltraLightDetector()
        self.model, self.device = load_face_recognition_model()
        
        # Load saved embeddings
        saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
        self.saved_embeddings = saved_data['embeddings']
        self.saved_labels = saved_data['labels']
        
        # Queues for pipeline stages
        self.alignment_queue = queue.Queue(maxsize=max_queue_size)
        self.recognition_queue = queue.Queue(maxsize=max_queue_size)
        
        # Shared state with thread safety
        self.lock = threading.Lock()
        self.recognition_results = {}
        self.next_face_id = 0
        self.frame_counter = 0
        
        # Start worker threads
        self.running = True
        self.alignment_thread = threading.Thread(target=self._alignment_worker)
        self.recognition_thread = threading.Thread(target=self._recognition_worker)
        self.alignment_thread.daemon = True  # Make threads daemon so they exit when main program exits
        self.recognition_thread.daemon = True
        self.alignment_thread.start()
        self.recognition_thread.start()

    def _validate_face_image(self, face_img):
        """Validate that the face image is valid for processing"""
        if face_img is None:
            return False
        if face_img.size == 0:  # Check if image is empty
            return False
        if len(face_img.shape) != 3:  # Check if image has proper dimensions
            return False
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:  # Check for zero dimensions
            return False
        return True

    def _extract_face_safely(self, frame, box):
        """Safely extract face region from frame"""
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame boundaries
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Check if resulting region is valid
            if x2 <= x1 or y2 <= y1:
                return None
                
            face_img = frame[y1:y2, x1:x2].copy()  # Make a copy to ensure memory safety
            
            if not self._validate_face_image(face_img):
                return None
                
            return face_img, (x1, y1, x2, y2)
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None

    def _alignment_worker(self):
        while self.running:
            try:
                face_data = self.alignment_queue.get(timeout=0.1)
                if face_data is None:
                    continue
                    
                frame, box, face_id = face_data
                extracted = self._extract_face_safely(frame, box)
                
                if extracted is None:
                    self.alignment_queue.task_done()
                    continue
                    
                face_img, adjusted_box = extracted
                
                try:
                    aligned_face = align_face(face_img)
                    if aligned_face is not None:
                        self.recognition_queue.put((aligned_face, face_id))
                except Exception as e:
                    print(f"Error in face alignment: {e}")
                
                self.alignment_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in alignment worker: {e}")
                continue

    def _recognition_worker(self):
        while self.running:
            try:
                recognition_data = self.recognition_queue.get(timeout=0.1)
                if recognition_data is None:
                    continue
                    
                aligned_face, face_id = recognition_data
                
                try:
                    # Convert to tensor and get embedding
                    face_tensor = self._preprocess_image(aligned_face).to(self.device)
                    embedding = self.model(face_tensor).detach()
                    
                    # Calculate similarities
                    similarities = [F.cosine_similarity(
                        torch.tensor(e).to(self.device), 
                        embedding
                    ).item() for e in self.saved_embeddings]
                    
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]
                    name = self.saved_labels[best_idx] if best_score > 0.41 else "Unknown"
                    
                    # Update results
                    with self.lock:
                        if face_id in self.recognition_results:
                            self.recognition_results[face_id].update({
                                'name': name,
                                'similarity': best_score,
                                'last_seen': self.frame_counter
                            })
                except Exception as e:
                    print(f"Error in recognition processing: {e}")
                
                self.recognition_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in recognition worker: {e}")
                continue

    def _preprocess_image(self, pil_image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(pil_image).unsqueeze(0)

    def _find_face(self, new_box):
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = ((box1[2] - box1[0]) * (box1[3] - box1[1]) + 
                    (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
            
            return inter / union if union > 0 else 0
            
        with self.lock:
            for face_id, data in self.recognition_results.items():
                if iou(new_box, data['box']) > 0.5:
                    return face_id
        return None

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return None
            
        self.frame_counter += 1
        processed_frame = frame.copy()
        current_faces = set()
        
        try:
            # Fast face detection
            boxes, _ = self.face_detector.detect_one(frame)
            
            # Process each detected face
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Find existing face or create new ID
                face_id = self._find_face(box) or f"face_{self.next_face_id}"
                if face_id not in self.recognition_results:
                    with self.lock:
                        self.recognition_results[face_id] = {
                            'name': "Unknown",
                            'similarity': 0.0,
                            'box': box,
                            'last_seen': self.frame_counter
                        }
                        self.next_face_id += 1
                
                # Update tracking
                current_faces.add(face_id)
                with self.lock:
                    self.recognition_results[face_id]['box'] = box
                
                # Queue for alignment and recognition
                if self.frame_counter % 5 == 0:  # Process every 5th frame
                    try:
                        self.alignment_queue.put_nowait((frame.copy(), box, face_id))
                    except queue.Full:
                        pass
                
                # Draw results
                result = self.recognition_results[face_id]
                color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, 
                           f"{result['name']}, score: {result['similarity']:.2f}", 
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.75, 
                           color, 
                           2)
            
            # Clean up old results
            if self.frame_counter % 30 == 0:
                with self.lock:
                    self.recognition_results = {
                        fid: data for fid, data in self.recognition_results.items()
                        if self.frame_counter - data['last_seen'] <= 15
                    }
            
            return processed_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

    def cleanup(self):
        self.running = False
        # No need to join threads since they're daemon threads now
        cv2.destroyAllWindows()

# Main loop
def main():
    pipeline = FaceRecognitionPipeline()
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = pipeline.process_frame(frame)
            if processed_frame is not None:
                cv2.imshow('Face Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        pipeline.cleanup()
        cap.release()

if __name__ == "__main__":
    main()