import cv2
import torch
import numpy as np
import threading
import queue
import os
import time
import csv # Added for CSV logging
from collections import deque # Use deque for efficient fixed-size log
from torch.nn.functional import cosine_similarity
# Import helper functions (adjust path if needed)
import sys
project_root_inf = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_inf not in sys.path:
    sys.path.insert(0, project_root_inf)
from app import resource_path, get_writable_path # Import from app.py

from src.backend.utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedLayout
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor

# Use get_writable_path for logs directory
LOG_DIR = get_writable_path("logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, 'detection_log.csv') # Define log file path using resolved dir

class FaceRecognitionPipeline:
    def __init__(self):
        # Resolve model path first
        model_path = resource_path("src/backend/checkpoints/edgeface_s_gamma_05.pt")
        # Pass resolved path to loader
        self.model, self.device = load_face_recognition_model(model_path=model_path)
        # Use get_writable_path for the database directory
        self.face_database_dir = get_writable_path("face_database")
        self.saved_embeddings = []
        self.saved_labels = []
        self.lock = threading.Lock()
        self.detection_log = deque(maxlen=100)
        self.last_log_time_per_person = {}
        self.log_interval = 15
        self.show_fps = False  # Changed to False by default

        # Frame skipping variables
        self.frame_skip_interval = 1  # Default: process every frame
        self.last_boxes = []  # Store boxes from last detection

        # Ensure log directory exists (get_writable_path handles this)
        # os.makedirs(LOG_DIR, exist_ok=True) # No longer needed here
        # Load existing logs from file
        self._load_logs_from_file()

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

    def _load_logs_from_file(self):
        """Loads previous detection logs from the CSV file into the deque."""
        # LOG_FILE_PATH is now correctly resolved
        try:
            if os.path.exists(LOG_FILE_PATH):
                print(f"Loading logs from {LOG_FILE_PATH}...")
                loaded_count = 0
                with open(LOG_FILE_PATH, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader, None) # Skip header if exists
                    if header != ['Name', 'Timestamp', 'Similarity']:
                         print("Warning: Log file header mismatch or missing. Attempting to read anyway.")
                         # Reset reader if header was missing or incorrect to read from start
                         csvfile.seek(0)
                         reader = csv.reader(csvfile)

                    temp_logs = []
                    for row in reader:
                        try:
                            if len(row) == 3:
                                name, timestamp_str, similarity_str = row
                                timestamp = float(timestamp_str)
                                similarity = float(similarity_str)
                                temp_logs.append((name, timestamp, similarity))
                                loaded_count += 1
                            else:
                                print(f"Skipping malformed log row: {row}")
                        except ValueError as e:
                            print(f"Skipping row due to parsing error ({e}): {row}")
                        except Exception as e:
                             print(f"Unexpected error reading log row ({e}): {row}")

                    # Sort by timestamp and take the latest N (up to maxlen)
                    temp_logs.sort(key=lambda x: x[1])
                    # Use extend which efficiently adds items to the deque respecting maxlen
                    self.detection_log.extend(temp_logs)
                    print(f"Loaded {loaded_count} log entries. Deque size: {len(self.detection_log)}")

            else:
                print(f"Log file {LOG_FILE_PATH} not found. Starting fresh.")
                # Create the file with header if it doesn't exist
                # Ensure directory exists before writing (get_writable_path should have done this)
                os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
                with open(LOG_FILE_PATH, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Name', 'Timestamp', 'Similarity']) # Write header

        except Exception as e:
            print(f"Error loading log file {LOG_FILE_PATH}: {e}")

    def _append_log_to_file(self, person_label, timestamp, similarity):
        """Appends a single detection log entry to the CSV file."""
        # LOG_FILE_PATH is now correctly resolved
        try:
            # Ensure directory exists before writing (get_writable_path should have done this)
            os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
            # Open in append mode, create if doesn't exist (though __init__ should handle creation)
            with open(LOG_FILE_PATH, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([person_label, timestamp, similarity])
        except Exception as e:
            print(f"Error writing to log file {LOG_FILE_PATH}: {e}")

    def _load_all_embeddings(self):
        """Loads all embeddings from the database directory."""
        # self.face_database_dir is now correctly resolved
        print(f"Loading all known face embeddings from {self.face_database_dir}...")
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

    def remove_person(self, person_name):
        """Removes embeddings and labels for a specific person."""
        print(f"Attempting to remove {person_name} from live recognition...")
        with self.lock: # Ensure thread-safe update
            initial_count = len(self.saved_embeddings)
            # Create new lists excluding the person to be removed
            new_embeddings = []
            new_labels = []
            removed_count = 0
            for i in range(len(self.saved_labels)):
                if self.saved_labels[i] != person_name:
                    new_embeddings.append(self.saved_embeddings[i])
                    new_labels.append(self.saved_labels[i])
                else:
                    removed_count += 1

            # Update the shared lists
            self.saved_embeddings = new_embeddings
            self.saved_labels = new_labels
            final_count = len(self.saved_embeddings)
            print(f"Removed {removed_count} embeddings for {person_name}. "
                  f"Total embeddings now: {final_count} (was {initial_count}).")

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
                                 # --- Log the detection event (with time interval check) ---
                                 current_time = time.time()
                                 last_log_time = self.last_log_time_per_person.get(person_label, 0)
                                 # Use the instance variable for log interval check
                                 if current_time - last_log_time > self.log_interval:
                                     log_entry = (person_label, current_time, best_score)
                                     self.detection_log.append(log_entry)
                                     self.last_log_time_per_person[person_label] = current_time
                                     # Append to file immediately after adding to deque
                                     self._append_log_to_file(person_label, current_time, best_score)
                                     # print(f"Logged detection for {person_label} at {current_time}") # Optional debug print
                                 # ---------------------------------------------------------
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
        self.recognition_threshold = recognition_threshold
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        self.frame_count += 1
        
        # Calculate FPS
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.fps = 0.9 * self.fps + 0.1 * fps  # Smooth FPS using exponential moving average
        self.prev_frame_time = self.curr_frame_time

        # Frame skipping logic
        current_boxes = []
        if self.frame_count % self.frame_skip_interval == 0:
            # Only detect faces on interval frames
            boxes, _ = detect_faces(frame)
            if boxes is not None and len(boxes) > 0:
                self.last_boxes = boxes
                current_boxes = boxes
            else:
                self.last_boxes = []
                current_boxes = []
        else:
            # Use last known boxes for skipped frames
            current_boxes = self.last_boxes
        
        if len(current_boxes) == 0:
            if self.show_fps:
                cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
            
        for box in current_boxes:
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
        
        # Display FPS only if enabled (at the end of the method)
        if self.show_fps:
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

    def get_detection_log(self):
        """Returns a copy of the current detection log."""
        with self.lock: # Ensure thread-safe access
            return list(self.detection_log) # Return a copy as a list

    def set_log_interval(self, interval_seconds):
        """Sets the minimum interval between logging the same person."""
        if isinstance(interval_seconds, (int, float)) and interval_seconds >= 0:
            with self.lock: # Ensure thread-safe update if accessed concurrently
                self.log_interval = interval_seconds
            print(f"Log interval updated to {self.log_interval} seconds.")
        else:
            print(f"Invalid log interval value: {interval_seconds}. Must be a non-negative number.")

    def get_log_interval(self):
        """Returns the current log interval."""
        with self.lock: # Ensure thread-safe access
            return self.log_interval

    def set_frame_skip_interval(self, interval):
        """Sets the frame skip interval."""
        if isinstance(interval, int) and interval >= 1:
            with self.lock:
                self.frame_skip_interval = interval
            print(f"Frame skip interval set to: {interval}")
        else:
            print(f"Invalid frame skip interval: {interval}. Must be a positive integer.")

    def get_frame_skip_interval(self):
        """Returns the current frame skip interval."""
        with self.lock:
            return self.frame_skip_interval

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