import cv2
import torch
import numpy as np
import threading
import queue
import os
import time
from torch.nn.functional import cosine_similarity
from src.backend.utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedLayout
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor

class FaceRecognitionPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        self.face_database_dir = './src/backend/face_database/' # Store path
        self.saved_embeddings = []
        self.saved_labels = []
        self.lock = threading.Lock() # Ensure lock is initialized before loading

        # Initial loading of embeddings
        self._load_all_embeddings()

        # FPS calculation variables
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
        self.align_queue = queue.Queue(maxsize=5)
        self.recog_queue = queue.Queue(maxsize=5)
        self.results = {}
        self.next_id = 0
        self.frame_count = 0
        self.running = True
        # self.lock = threading.Lock() # Moved up

        # Start alignment and recognition worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def _load_all_embeddings(self):
        """Loads all embeddings from the database directory."""
        print("Loading all known face embeddings...")
        loaded_embeddings = []
        loaded_labels = []
        try:
            for filename in os.listdir(self.face_database_dir):
                if filename.endswith('.npz'):
                    person_name = os.path.splitext(filename)[0]
                    npz_path = os.path.join(self.face_database_dir, filename)
                    try:
                        data = np.load(npz_path)
                        embeddings = data['embeddings']
                        for embedding in embeddings:
                            # Ensure embedding is correctly shaped (e.g., 1D array)
                            embedding_tensor = torch.tensor(embedding.squeeze()).to(self.device)
                            loaded_embeddings.append(embedding_tensor)
                            loaded_labels.append(person_name)
                        print(f"Loaded {len(embeddings)} embeddings for {person_name}")
                    except Exception as e:
                        print(f"Error loading embeddings from {filename}: {e}")
        except FileNotFoundError:
            print(f"Warning: Face database directory not found at {self.face_database_dir}")
        except Exception as e:
            print(f"An error occurred while listing database directory: {e}")

        with self.lock:
            self.saved_embeddings = loaded_embeddings
            self.saved_labels = loaded_labels
        print(f"Total loaded embeddings: {len(self.saved_embeddings)}")


    def load_new_person(self, person_name, npz_path):
        """Loads embeddings for a newly added person and appends them."""
        print(f"Dynamically loading embeddings for new person: {person_name}")
        try:
            data = np.load(npz_path)
            embeddings = data['embeddings']
            new_embeddings_count = 0
            with self.lock: # Ensure thread-safe update
                for embedding in embeddings:
                     # Ensure embedding is correctly shaped (e.g., 1D array)
                    embedding_tensor = torch.tensor(embedding.squeeze()).to(self.device)
                    self.saved_embeddings.append(embedding_tensor)
                    self.saved_labels.append(person_name)
                    new_embeddings_count += 1
            print(f"Successfully loaded {new_embeddings_count} new embeddings for {person_name}.")
            print(f"Total embeddings now: {len(self.saved_embeddings)}")
        except FileNotFoundError:
            print(f"Error: .npz file not found at {npz_path}")
        except Exception as e:
            print(f"Error loading new person embeddings from {npz_path}: {e}")

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
                with torch.no_grad(): # Ensure no gradients are calculated during inference
                    embedding = self.model(face_tensor).detach()

                with self.lock: # Acquire lock before accessing shared lists
                    if not self.saved_embeddings: # Check if embeddings list is empty
                         # Handle case with no known faces gracefully
                         best_score = -1.0
                         best_idx = -1
                    else:
                        # Calculate similarities only if there are saved embeddings
                        similarities = [cosine_similarity(saved_emb.unsqueeze(0), embedding).item() for saved_emb in self.saved_embeddings]
                        if similarities: # Ensure similarities list is not empty
                            best_idx = np.argmax(similarities)
                            best_score = similarities[best_idx]
                        else:
                            best_score = -1.0
                            best_idx = -1

                    # Update results (still under lock)
                    if face_id in self.results:
                        person_label = "Unknown"
                        if best_idx != -1 and best_score > self.recognition_threshold:
                             # Check index bounds just in case
                             if best_idx < len(self.saved_labels):
                                 person_label = self.saved_labels[best_idx]
                             else:
                                 print(f"Warning: best_idx {best_idx} out of bounds for saved_labels (len {len(self.saved_labels)})")


                        self.results[face_id].update({
                            'name': person_label,
                            'similarity': best_score if best_idx != -1 else 0.0,
                            'last_seen': self.frame_count
                        })
                    # Lock is released automatically when 'with' block exits

                self.recog_queue.task_done()
            except queue.Empty:
                 continue # Handle empty queue if timeout is used in get()
            except Exception as e:
                print(f"Recognition error: {e}")
                # Optionally put task_done even on error if appropriate
                # self.recog_queue.task_done()


    def process_frame(self, frame, recognition_threshold=0.6):
        """Detect, align, and recognize faces in a video frame."""
        self.recognition_threshold = recognition_threshold  # Store the threshold
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        self.frame_count += 1
        
        # Calculate FPS
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.fps = 0.9 * self.fps + 0.1 * fps  # Smooth FPS using exponential moving average
        self.prev_frame_time = self.curr_frame_time
        
        boxes, _ = detect_faces(frame)
        
        if len(boxes) == 0:
            # Still display FPS even when no faces are detected
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        
        # Display FPS
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.frame_count % 30 == 0:
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
        
        return frame

    def cleanup(self):
        """Stop the pipeline and release resources."""
        print("Cleaning up FaceRecognitionPipeline...")
        self.running = False
        # Add sentinel values to unblock worker threads waiting on queues
        try:
            self.align_queue.put_nowait((None, None, None))
        except queue.Full:
            pass
        try:
            self.recog_queue.put_nowait((None, None))
        except queue.Full:
            pass
        # Optionally join threads here if needed, though daemon=True helps
        cv2.destroyAllWindows()
        print("Cleanup complete.")


class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;") # Default background
        self.layout.addWidget(self.camera_label)

        self.pipeline = FaceRecognitionPipeline()
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # Do not start the timer here automatically
        # self.timer.start(30)

        # Placeholder text when feed is off
        self.placeholder_label = QLabel("Camera Feed Off")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("color: white; font-size: 16px;")
        self.placeholder_label.setMinimumSize(640, 480) # Match camera label size
        self.placeholder_label.hide() # Hide initially

        # Use a stacked layout to switch between camera feed and placeholder
        self.stacked_layout = QStackedLayout()
        self.stacked_layout.addWidget(self.camera_label)
        self.stacked_layout.addWidget(self.placeholder_label)
        self.layout.addLayout(self.stacked_layout) # Add stacked layout to main layout

        self.stacked_layout.setCurrentWidget(self.placeholder_label) # Show placeholder initially

        self.recognition_threshold = 0.6  # Add default threshold

    def set_recognition_threshold(self, threshold):
        """Sets the recognition threshold value."""
        self.recognition_threshold = threshold

    def start_feed(self):
        """Starts the camera feed update timer."""
        if not self.timer.isActive():
            self.stacked_layout.setCurrentWidget(self.camera_label) # Show camera label
            self.timer.start(30)

    def stop_feed(self):
        """Stops the camera feed update timer and shows placeholder."""
        if self.timer.isActive():
            self.timer.stop()
            # Clear the label or set a placeholder image/text
            # Create a black pixmap
            black_pixmap = QPixmap(self.camera_label.size())
            black_pixmap.fill(QColor('black'))
            self.camera_label.setPixmap(black_pixmap)
            self.stacked_layout.setCurrentWidget(self.placeholder_label) # Show placeholder

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            processed = self.pipeline.process_frame(frame, self.recognition_threshold)  # Pass threshold
            if processed is not None:
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Scale to fill the label while maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.width(),
                    self.camera_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        print("Closing CameraWidget...")
        self.stop_feed() # Ensure feed is stopped
        if hasattr(self, 'pipeline') and self.pipeline:
             self.pipeline.cleanup()
        if hasattr(self, 'cap') and self.cap:
             self.cap.release()
        print("CameraWidget resources released.")
        super().closeEvent(event)

def main():
    """Removed standalone window code as it's now integrated with PyQt6"""
    pass

if __name__ == "__main__":
    main()