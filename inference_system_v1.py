import cv2
import torch
import numpy as np
import threading
import queue
import os
from tkinter import Tk, StringVar, Canvas, ttk, Toplevel
from prepare_embeddings_system_v1 import FaceEmbeddingApp
import sv_ttk
from PIL import Image, ImageTk
from torch.nn.functional import cosine_similarity
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform

class FaceRecognitionPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        self.face_database_dir = './face_database/'
        self.saved_embeddings = []
        self.saved_labels = []
        self.load_embeddings()

        self.align_queue = queue.Queue(maxsize=5)
        self.recog_queue = queue.Queue(maxsize=5)
        self.results = {}
        self.next_id = 0
        self.frame_count = 0
        self.running = True
        self.lock = threading.Lock()

        # Configuration values
        self.min_accuracy = 0.5  # Default minimum accuracy for inference
        self.min_recognize = 0.8  # Default minimum accuracy for recognition

        # Start alignment and recognition worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def load_embeddings(self):
        """Load embeddings from the face database directory."""
        self.saved_embeddings = []
        self.saved_labels = []
        for filename in os.listdir(self.face_database_dir):
            if filename.endswith('.npz'):
                person_name = os.path.splitext(filename)[0]
                npz_path = os.path.join(self.face_database_dir, filename)
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
        
        # Configuration values
        self.min_accuracy = 0.5  # Default minimum accuracy for inference
        self.min_recognize = 0.8  # Default minimum accuracy for recognition
        
        # Start alignment and recognition worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def update_recognized_faces(self):
        """Update the recognized faces when embeddings are reloaded."""
        with self.lock:
            # Get the list of current labels in the embeddings
            current_labels = set(self.saved_labels)

            # Iterate through the results and mark unrecognized faces as "Unknown"
            for face_id, data in list(self.results.items()):
                if data['name'] not in current_labels:
                    self.results[face_id].update({
                        'name': "Unknown",
                        'similarity': 0.0,
                        'recognized': False
                    })

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
                        if best_score >= self.min_recognize:
                            self.results[face_id].update({
                                'name': self.saved_labels[best_idx],
                                'similarity': best_score,
                                'last_seen': self.frame_count,
                                'recognized': True  # Mark as recognized
                            })
                        elif best_score >= self.min_accuracy:
                            self.results[face_id].update({
                                'name': self.saved_labels[best_idx],
                                'similarity': best_score,
                                'last_seen': self.frame_count,
                                'recognized': False  # Not yet recognized
                            })
                        else:
                            self.results[face_id].update({
                                'name': "Unknown",
                                'similarity': best_score,
                                'last_seen': self.frame_count,
                                'recognized': False  # Unknown
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
                        'last_seen': self.frame_count,
                        'recognized': False  # Initially not recognized
                    }
                    self.next_id += 1
                
                self.results[face_id].update({
                    'box': box,
                    'last_seen': self.frame_count
                })
            
            if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass
            
            result = self.results[face_id]
            if result['recognized']:
                color = (255, 0, 0)  
                label = f"Recognized: {result['name']}"
            elif result['name'] != "Unknown":
                color = (0, 255, 0) 
                label = f"{result['name']}, score: {result['similarity']:.2f}"
            else:
                color = (0, 0, 255)
                label = "Unknown"
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if self.frame_count % 30 == 0:
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
        
        return frame

    def cleanup(self):
        """Stop the pipeline and release resources."""
        self.running = False
        cv2.destroyAllWindows()

class App:
    def __init__(self, window, window_title, pipeline):
        self.window = window
        window.resizable(False, False)
        self.window.title(window_title)
        self.pipeline = pipeline

        sv_ttk.set_theme("dark")

        self.side_panel = ttk.Frame(window, width=200)
        self.side_panel.pack(side="left", fill="y")

        self.title_label = ttk.Label(self.side_panel, text="Config", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)
        self.min_accuracy_label = ttk.Label(self.side_panel, text="Minimum Accuracy Threshold:")
        self.min_accuracy_label.pack(pady=5)
        self.min_accuracy_var = StringVar(value=str(self.pipeline.min_accuracy))
        self.min_accuracy_entry = ttk.Entry(self.side_panel, textvariable=self.min_accuracy_var)
        self.min_accuracy_entry.pack(pady=5)

        self.min_recognize_label = ttk.Label(self.side_panel, text="Minimum Recognize Threshold:")
        self.min_recognize_label.pack(pady=5)
        self.min_recognize_var = StringVar(value=str(self.pipeline.min_recognize))
        self.min_recognize_entry = ttk.Entry(self.side_panel, textvariable=self.min_recognize_var)
        self.min_recognize_entry.pack(pady=5)

        self.refresh_button = ttk.Button(self.side_panel, text="Refresh Config", command=self.refresh_config)
        self.refresh_button.pack(pady=10)

        # Add a new button to manage embeddings
        self.manage_embeddings_button = ttk.Button(self.side_panel, text="Manage Embeddings", command=self.open_embedding_manager)
        self.manage_embeddings_button.pack(pady=10)

        self.canvas = Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.cap = cv2.VideoCapture(0)
        self.update()

    def refresh_config(self):
        """Update the configuration values from the entry fields."""
        try:
            self.pipeline.min_accuracy = float(self.min_accuracy_var.get())
            self.pipeline.min_recognize = float(self.min_recognize_var.get())
            print(f"Config updated: Min Accuracy = {self.pipeline.min_accuracy}, Min Recognize = {self.pipeline.min_recognize}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    def update(self):
        """Update the OpenCV display in the tkinter window."""
        ret, frame = self.cap.read()
        if ret:
            frame = self.pipeline.process_frame(frame)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.window.after(10, self.update)

    def cleanup(self):
        """Release resources."""
        self.pipeline.cleanup()
        self.cap.release()

    def open_embedding_manager(self):
        """Open the FaceEmbeddingApp to manage embeddings."""
        embedding_window = Toplevel(self.window)
        FaceEmbeddingApp(embedding_window, self.pipeline)


def main():
    pipeline = FaceRecognitionPipeline()
    root = Tk()
    sv_ttk.use_dark_theme()  # Initialize Sun Valley theme
    app = App(root, "Facial Recognition System", pipeline)
    root.mainloop()
    app.cleanup()

if __name__ == "__main__":
    main()
