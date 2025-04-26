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
        face_database_dir = './src/backend/face_database/'
        self.saved_embeddings = []
        self.saved_labels = []
        
        # FPS calculation variables
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
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
                            'name': self.saved_labels[best_idx] if best_score > self.recognition_threshold else "Unknown",
                            'similarity': best_score,
                            'last_seen': self.frame_count
                        })
                
                self.recog_queue.task_done()
            except Exception as e:
                print(f"Recognition error: {e}")

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
        self.running = False
        cv2.destroyAllWindows()

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
        self.stop_feed() # Ensure feed is stopped
        self.pipeline.cleanup()
        self.cap.release()
        super().closeEvent(event)
