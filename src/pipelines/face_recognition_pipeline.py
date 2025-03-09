import cv2
import torch
import numpy as np
import queue
import threading
import os
from torch.nn.functional import cosine_similarity
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
from utils.path_patch_v1 import get_resource_path

class FaceRecognitionPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        self.face_database_dir = get_resource_path('face_database')
        self.records_dir = os.path.join(self.face_database_dir, 'records')
        os.makedirs(self.records_dir, exist_ok=True)
        self.saved_embeddings = []
        self.saved_labels = []
        self.saved_uids = []
        self.load_embeddings()
        self.frame_counters = {}

        self.align_queue = queue.Queue(maxsize=5)
        self.recog_queue = queue.Queue(maxsize=5)
        self.results = {}
        self.next_id = 0
        self.frame_count = self.get_highest_frame_number()  # Initialize frame_count
        self.running = True
        self.lock = threading.Lock()

        # Configuration values
        self.min_accuracy = 0.5  # Minimum accuracy for inference
        self.min_recognize = 0.8  # Minimum accuracy for recognition
        self.save_frame_interval = 60  # Save a frame every defined interval

        # Start worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def get_highest_frame_number(self):
        """Find the highest frame number in the records directory."""
        max_frame_number = 0
        for uid_folder in os.listdir(self.records_dir):
            uid_folder_path = os.path.join(self.records_dir, uid_folder)
            if os.path.isdir(uid_folder_path):
                for frame_file in os.listdir(uid_folder_path):
                    if frame_file.endswith('.jpg'):
                        frame_number = int(os.path.splitext(frame_file)[0])
                        if frame_number > max_frame_number:
                            max_frame_number = frame_number
        return max_frame_number

    def load_embeddings(self):
        """Load embeddings from the face database directory."""
        self.saved_embeddings = []
        self.saved_labels = []
        self.saved_uids = []
        for filename in os.listdir(self.face_database_dir):
            if filename.endswith('.npz'):
                npz_path = os.path.join(self.face_database_dir, filename)
                data = np.load(npz_path)
                embeddings = data['embeddings']
                person_name = data.get('person_name', os.path.splitext(filename)[0])
                uid = data.get('uid', os.path.splitext(filename)[0])  # Load UID
                for embedding in embeddings:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    self.saved_labels.append(person_name)
                    self.saved_uids.append(uid)

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

                if not self.saved_embeddings:
                    with self.lock:
                        if face_id in self.results:
                            self.results[face_id].update({'name': "Unknown", 'similarity': 0.0, 'recognized': False})
                    self.recog_queue.task_done()
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
                                'recognized': True,
                                'uid': self.saved_uids[best_idx]
                            })
                        elif best_score >= self.min_accuracy:
                            self.results[face_id].update({'name': self.saved_labels[best_idx], 'similarity': best_score, 'recognized': False})
                        else:
                            self.results[face_id].update({'name': "Unknown", 'similarity': best_score, 'recognized': False})
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
                    self.results[face_id] = {'name': "Unknown", 'similarity': 0.0, 'box': box, 'recognized': False}
                    self.next_id += 1

                self.results[face_id].update({'box': box})

            if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass

            result = self.results[face_id]
            if result['recognized']:
                color = (255, 0, 0)
                label = f"Recognized: {result['name']}"

                if self.frame_count % self.save_frame_interval == 0:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size != 0:
                        uid = str(result.get('uid', 'unknown'))
                        uid_folder = os.path.join(self.records_dir, uid)
                        os.makedirs(uid_folder, exist_ok=True)

                        # Ensure sequential numbering per UID
                        if uid not in self.frame_counters:
                            existing_frames = [int(f.split(".")[0]) for f in os.listdir(uid_folder) if f.endswith(".jpg") and f.split(".")[0].isdigit()]
                            self.frame_counters[uid] = max(existing_frames, default=0) + 1  

                        frame_filename = os.path.join(uid_folder, f"{self.frame_counters[uid]}.jpg")
                        cv2.imwrite(frame_filename, face_crop)

                        self.frame_counters[uid] += 1  

            elif result['name'] != "Unknown":
                color = (0, 255, 0)
                label = f"{result['name']}, score: {result['similarity']:.2f}"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def cleanup(self):
        """Stop the pipeline and release resources."""
        self.running = False
        cv2.destroyAllWindows()

# import cv2
# import torch
# import numpy as np
# import queue
# import threading
# import os
# from torch.nn.functional import cosine_similarity
# from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
# from utils.path_patch_v1 import get_resource_path

# class FaceRecognitionPipeline:
#     def __init__(self):
#         self.model, self.device = load_face_recognition_model()
#         self.face_database_dir = get_resource_path('face_database')
#         self.records_dir = os.path.join(self.face_database_dir, 'records')
#         os.makedirs(self.records_dir, exist_ok=True)
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         self.load_embeddings()
#         self.frame_counters = {}

#         self.align_queue = queue.Queue(maxsize=5)
#         self.recog_queue = queue.Queue(maxsize=5)
#         self.results = {}
#         self.next_id = 0
#         self.frame_count = self.get_highest_frame_number()  # Initialize frame_count
#         self.running = True
#         self.lock = threading.Lock()

#         # Configuration values
#         self.min_accuracy = 0.5  # Default minimum accuracy for inference
#         self.min_recognize = 0.8  # Default minimum accuracy for recognition
#         self.save_frame_interval = 60  # Save a frame every defined frames

#         # Start alignment and recognition worker threads
#         threading.Thread(target=self._alignment_worker, daemon=True).start()
#         threading.Thread(target=self._recognition_worker, daemon=True).start()

#     def get_highest_frame_number(self):
#         """Find the highest frame number in the records directory."""
#         max_frame_number = 0
#         for uid_folder in os.listdir(self.records_dir):
#             uid_folder_path = os.path.join(self.records_dir, uid_folder)
#             if os.path.isdir(uid_folder_path):
#                 for frame_file in os.listdir(uid_folder_path):
#                     if frame_file.endswith('.jpg'):
#                         frame_number = int(os.path.splitext(frame_file)[0])
#                         if frame_number > max_frame_number:
#                             max_frame_number = frame_number
#         return max_frame_number

#     def load_embeddings(self):
#         """Load embeddings from the face database directory."""
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         for filename in os.listdir(self.face_database_dir):
#             if filename.endswith('.npz'):
#                 npz_path = os.path.join(self.face_database_dir, filename)
#                 data = np.load(npz_path)
#                 embeddings = data['embeddings']
#                 person_name = data.get('person_name', os.path.splitext(filename)[0])
#                 uid = data.get('uid', os.path.splitext(filename)[0])  # Load the uid from the .npz file
#                 for embedding in embeddings:
#                     self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
#                     self.saved_labels.append(person_name)
#                     self.saved_uids.append(uid)

#     def update_recognized_faces(self):
#         """Update the recognized faces when embeddings are reloaded."""
#         with self.lock:
#             current_labels = set(self.saved_labels)  # Ensure labels are hashable (e.g., strings)
#             for face_id, data in list(self.results.items()):
#                 name = str(data['name']) if isinstance(data['name'], np.ndarray) else data['name']
#                 if name not in current_labels:
#                     self.results[face_id].update({
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'recognized': False
#                     })

#     def _alignment_worker(self):
#         """Thread worker for aligning faces before recognition."""
#         while self.running:
#             try:
#                 frame, box, face_id = self.align_queue.get()
#                 if frame is None:
#                     continue
#                 x1, y1, x2, y2 = map(int, box)
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue
#                 aligned_face = align_face(face_crop)
#                 if aligned_face is not None:
#                     self.recog_queue.put((aligned_face, face_id))
#                 self.align_queue.task_done()
#             except Exception as e:
#                 print(f"Alignment error: {e}")

#     def _recognition_worker(self):
#         """Thread worker for face recognition."""
#         while self.running:
#             try:
#                 aligned_face, face_id = self.recog_queue.get()
#                 if aligned_face is None:
#                     continue

#                 if not self.saved_embeddings:
#                     with self.lock:
#                         if face_id in self.results:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': 0.0,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                     self.recog_queue.task_done()
#                     continue

#                 face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
#                 embedding = self.model(face_tensor).detach()
#                 similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
#                 best_idx = np.argmax(similarities)
#                 best_score = similarities[best_idx]
#                 with self.lock:
#                     if face_id in self.results:
#                         if best_score >= self.min_recognize:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': True,
#                                 'uid': self.saved_uids[best_idx]  # Store the uid for the recognized face
#                             })
#                         elif best_score >= self.min_accuracy:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                         else:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                 self.recog_queue.task_done()
#             except Exception as e:
#                 print(f"Recognition error: {e}")

#     def process_frame(self, frame):
#         """Detect, align, and recognize faces in a video frame."""
#         if not isinstance(frame, np.ndarray) or frame.size == 0:
#             return frame
#         self.frame_count += 1  # This remains for global tracking, not for naming files.

#         boxes, _ = detect_faces(frame)
#         if len(boxes) == 0:
#             return frame

#         for box in boxes:
#             face_id = next(
#                 (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50),
#                 f"face_{self.next_id}"
#             )

#             with self.lock:
#                 if face_id not in self.results:
#                     self.results[face_id] = {
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'box': box,
#                         'last_seen': self.frame_count,
#                         'recognized': False
#                     }
#                     self.next_id += 1

#                 self.results[face_id].update({
#                     'box': box,
#                     'last_seen': self.frame_count
#                 })

#             # Ensure only unrecognized faces are sent for alignment
#             if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
#                 try:
#                     self.align_queue.put_nowait((frame.copy(), box, face_id))
#                 except queue.Full:
#                     pass

#             result = self.results[face_id]
#             if result['recognized']:
#                 color = (255, 0, 0)
#                 label = f"Recognized: {result['name']}"

#                 if self.frame_count % self.save_frame_interval == 0:
#                     x1, y1, x2, y2 = map(int, box)
#                     face_crop = frame[y1:y2, x1:x2]
#                     if face_crop.size != 0:
#                         uid = result.get('uid', 'unknown')
#                         if isinstance(uid, np.ndarray):  # Convert UID to string if it's a NumPy array
#                             uid = uid.item() if uid.size == 1 else str(uid)

#                         uid_folder = os.path.join(self.records_dir, uid)
#                         os.makedirs(uid_folder, exist_ok=True)

#                         # Ensure sequential numbering per UID
#                         if uid not in self.frame_counters:
#                             self.frame_counters[uid] = 1  # Start from 1 for each UID

#                         frame_filename = os.path.join(uid_folder, f"{self.frame_counters[uid]}.jpg")
#                         cv2.imwrite(frame_filename, face_crop)

#                         self.frame_counters[uid] += 1  # Increment count for the next frame

#             elif result['name'] != "Unknown":
#                 color = (0, 255, 0)
#                 label = f"{result['name']}, score: {result['similarity']:.2f}"
#             else:
#                 color = (0, 0, 255)
#                 label = "Unknown"

#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Remove stale face entries
#         if self.frame_count % 30 == 0:
#             with self.lock:
#                 self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}

#         return frame

#     def cleanup(self):
#         """Stop the pipeline and release resources."""
#         self.running = False
#         cv2.destroyAllWindows()


# import cv2
# import torch
# import numpy as np
# import queue
# import threading
# import os
# from torch.nn.functional import cosine_similarity
# from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
# from utils.path_patch_v1 import get_resource_path

# class FaceRecognitionPipeline:
#     def __init__(self):
#         self.model, self.device = load_face_recognition_model()
#         self.face_database_dir = get_resource_path('face_database')
#         self.records_dir = os.path.join(self.face_database_dir, 'records')
#         os.makedirs(self.records_dir, exist_ok=True)
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         self.load_embeddings()
#         self.frame_counters = {}

#         self.align_queue = queue.Queue(maxsize=5)
#         self.recog_queue = queue.Queue(maxsize=5)
#         self.results = {}
#         self.next_id = 0
#         self.frame_count = self.get_highest_frame_number()  # Initialize frame_count
#         self.running = True
#         self.lock = threading.Lock()

#         # Configuration values
#         self.min_accuracy = 0.5  # Default minimum accuracy for inference
#         self.min_recognize = 0.8  # Default minimum accuracy for recognition
#         self.save_frame_interval = 60  # Save a frame every defined frames

#         # Start alignment and recognition worker threads
#         threading.Thread(target=self._alignment_worker, daemon=True).start()
#         threading.Thread(target=self._recognition_worker, daemon=True).start()

#     def get_highest_frame_number(self):
#         """Find the highest frame number in the records directory."""
#         max_frame_number = 0
#         for uid_folder in os.listdir(self.records_dir):
#             uid_folder_path = os.path.join(self.records_dir, uid_folder)
#             if os.path.isdir(uid_folder_path):
#                 for frame_file in os.listdir(uid_folder_path):
#                     if frame_file.endswith('.jpg'):
#                         frame_number = int(os.path.splitext(frame_file)[0])
#                         if frame_number > max_frame_number:
#                             max_frame_number = frame_number
#         return max_frame_number

#     def load_embeddings(self):
#         """Load embeddings from the face database directory."""
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         for filename in os.listdir(self.face_database_dir):
#             if filename.endswith('.npz'):
#                 npz_path = os.path.join(self.face_database_dir, filename)
#                 data = np.load(npz_path)
#                 embeddings = data['embeddings']
#                 person_name = data.get('person_name', os.path.splitext(filename)[0])
#                 uid = data.get('uid', os.path.splitext(filename)[0])  # Load the uid from the .npz file
#                 for embedding in embeddings:
#                     self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
#                     self.saved_labels.append(person_name)
#                     self.saved_uids.append(uid)

#     def update_recognized_faces(self):
#         """Update the recognized faces when embeddings are reloaded."""
#         with self.lock:
#             current_labels = set(self.saved_labels)  # Ensure labels are hashable (e.g., strings)
#             for face_id, data in list(self.results.items()):
#                 name = str(data['name']) if isinstance(data['name'], np.ndarray) else data['name']
#                 if name not in current_labels:
#                     self.results[face_id].update({
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'recognized': False
#                     })

#     def _alignment_worker(self):
#         """Thread worker for aligning faces before recognition."""
#         while self.running:
#             try:
#                 frame, box, face_id = self.align_queue.get()
#                 if frame is None:
#                     continue
#                 x1, y1, x2, y2 = map(int, box)
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue
#                 aligned_face = align_face(face_crop)
#                 if aligned_face is not None:
#                     self.recog_queue.put((aligned_face, face_id))
#                 self.align_queue.task_done()
#             except Exception as e:
#                 print(f"Alignment error: {e}")

#     def _recognition_worker(self):
#         """Thread worker for face recognition."""
#         while self.running:
#             try:
#                 aligned_face, face_id = self.recog_queue.get()
#                 if aligned_face is None:
#                     continue

#                 if not self.saved_embeddings:
#                     with self.lock:
#                         if face_id in self.results:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': 0.0,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                     self.recog_queue.task_done()
#                     continue

#                 face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
#                 embedding = self.model(face_tensor).detach()
#                 similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
#                 best_idx = np.argmax(similarities)
#                 best_score = similarities[best_idx]
#                 with self.lock:
#                     if face_id in self.results:
#                         if best_score >= self.min_recognize:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': True,
#                                 'uid': self.saved_uids[best_idx]  # Store the uid for the recognized face
#                             })
#                         elif best_score >= self.min_accuracy:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                         else:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                 self.recog_queue.task_done()
#             except Exception as e:
#                 print(f"Recognition error: {e}")

#     def process_frame(self, frame):
#         """Detect, align, and recognize faces in a video frame."""
#         if not isinstance(frame, np.ndarray) or frame.size == 0:
#             return frame
#         self.frame_count += 1
#         boxes, _ = detect_faces(frame)
#         if len(boxes) == 0:
#             return frame
#         for box in boxes:
#             face_id = next(
#                 (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50),
#                 f"face_{self.next_id}"
#             )
#             with self.lock:
#                 if face_id not in self.results:
#                     self.results[face_id] = {
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'box': box,
#                         'last_seen': self.frame_count,
#                         'recognized': False
#                     }
#                     self.next_id += 1
#                 self.results[face_id].update({
#                     'box': box,
#                     'last_seen': self.frame_count
#                 })
#             if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
#                 try:
#                     self.align_queue.put_nowait((frame.copy(), box, face_id))
#                 except queue.Full:
#                     pass
#             result = self.results[face_id]
#             if result['recognized']:
#                 color = (255, 0, 0)
#                 label = f"Recognized: {result['name']}"
                
#                 # Save the frame inside the bounding box if the save_frame_interval condition is met
#                 if self.frame_count % self.save_frame_interval == 0:
#                     x1, y1, x2, y2 = map(int, box)
#                     face_crop = frame[y1:y2, x1:x2]
#                     if face_crop.size != 0:
#                         uid = result.get('uid', 'unknown')
#                         if isinstance(uid, np.ndarray):  # Convert uid to string if it's a NumPy array
#                             uid = uid.item() if uid.size == 1 else str(uid)
#                         # Use the UID folder created by process_video
#                         uid_folder = os.path.join(self.records_dir, uid)
#                         os.makedirs(uid_folder, exist_ok=True)
#                         frame_path = os.path.join(uid_folder, f"{self.frame_count}.jpg")
#                         cv2.imwrite(frame_path, face_crop)
#             elif result['name'] != "Unknown":
#                 color = (0, 255, 0)
#                 label = f"{result['name']}, score: {result['similarity']:.2f}"
#             else:
#                 color = (0, 0, 255)
#                 label = "Unknown"
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         if self.frame_count % 30 == 0:
#             with self.lock:
#                 self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
#         return frame

#     def cleanup(self):
#         """Stop the pipeline and release resources."""
#         self.running = False
#         cv2.destroyAllWindows()

# import cv2
# import torch
# import numpy as np
# import queue
# import threading
# import os
# from torch.nn.functional import cosine_similarity
# from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform
# from utils.path_patch_v1 import get_resource_path

# class FaceRecognitionPipeline:
#     def __init__(self):
#         self.model, self.device = load_face_recognition_model()
#         self.face_database_dir = get_resource_path('face_database')
#         self.records_dir = os.path.join(self.face_database_dir, 'records')
#         os.makedirs(self.records_dir, exist_ok=True)
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         self.load_embeddings()

#         self.align_queue = queue.Queue(maxsize=5)
#         self.recog_queue = queue.Queue(maxsize=5)
#         self.results = {}
#         self.next_id = 0
#         self.frame_count = 0
#         self.running = True
#         self.lock = threading.Lock()

#         # Configuration values
#         self.min_accuracy = 0.5  # Default minimum accuracy for inference
#         self.min_recognize = 0.8  # Default minimum accuracy for recognition
#         self.save_frame_interval = 30  # Save a frame every 30 frames

#         # Start alignment and recognition worker threads
#         threading.Thread(target=self._alignment_worker, daemon=True).start()
#         threading.Thread(target=self._recognition_worker, daemon=True).start()

#     def load_embeddings(self):
#         """Load embeddings from the face database directory."""
#         self.saved_embeddings = []
#         self.saved_labels = []
#         self.saved_uids = []
#         for filename in os.listdir(self.face_database_dir):
#             if filename.endswith('.npz'):
#                 npz_path = os.path.join(self.face_database_dir, filename)
#                 data = np.load(npz_path)
#                 embeddings = data['embeddings']
#                 person_name = data.get('person_name', os.path.splitext(filename)[0])
#                 uid = data.get('uid', os.path.splitext(filename)[0])  # Load the uid from the .npz file
#                 for embedding in embeddings:
#                     self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
#                     self.saved_labels.append(person_name)
#                     self.saved_uids.append(uid)

#     def update_recognized_faces(self):
#         """Update the recognized faces when embeddings are reloaded."""
#         with self.lock:
#             current_labels = set(self.saved_labels)  # Ensure labels are hashable (e.g., strings)
#             for face_id, data in list(self.results.items()):
#                 name = str(data['name']) if isinstance(data['name'], np.ndarray) else data['name']
#                 if name not in current_labels:
#                     self.results[face_id].update({
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'recognized': False
#                     })

#     def _alignment_worker(self):
#         """Thread worker for aligning faces before recognition."""
#         while self.running:
#             try:
#                 frame, box, face_id = self.align_queue.get()
#                 if frame is None:
#                     continue
#                 x1, y1, x2, y2 = map(int, box)
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue
#                 aligned_face = align_face(face_crop)
#                 if aligned_face is not None:
#                     self.recog_queue.put((aligned_face, face_id))
#                 self.align_queue.task_done()
#             except Exception as e:
#                 print(f"Alignment error: {e}")

#     def _recognition_worker(self):
#         """Thread worker for face recognition."""
#         while self.running:
#             try:
#                 aligned_face, face_id = self.recog_queue.get()
#                 if aligned_face is None:
#                     continue

#                 if not self.saved_embeddings:
#                     with self.lock:
#                         if face_id in self.results:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': 0.0,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                     self.recog_queue.task_done()
#                     continue

#                 face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
#                 embedding = self.model(face_tensor).detach()
#                 similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
#                 best_idx = np.argmax(similarities)
#                 best_score = similarities[best_idx]
#                 with self.lock:
#                     if face_id in self.results:
#                         if best_score >= self.min_recognize:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': True,
#                                 'uid': self.saved_uids[best_idx]  # Store the uid for the recognized face
#                             })
#                         elif best_score >= self.min_accuracy:
#                             self.results[face_id].update({
#                                 'name': self.saved_labels[best_idx],
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                         else:
#                             self.results[face_id].update({
#                                 'name': "Unknown",
#                                 'similarity': best_score,
#                                 'last_seen': self.frame_count,
#                                 'recognized': False
#                             })
#                 self.recog_queue.task_done()
#             except Exception as e:
#                 print(f"Recognition error: {e}")

#     def process_frame(self, frame):
#         """Detect, align, and recognize faces in a video frame."""
#         if not isinstance(frame, np.ndarray) or frame.size == 0:
#             return frame
#         self.frame_count += 1
#         boxes, _ = detect_faces(frame)
#         if len(boxes) == 0:
#             return frame
#         for box in boxes:
#             face_id = next(
#                 (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50),
#                 f"face_{self.next_id}"
#             )
#             with self.lock:
#                 if face_id not in self.results:
#                     self.results[face_id] = {
#                         'name': "Unknown",
#                         'similarity': 0.0,
#                         'box': box,
#                         'last_seen': self.frame_count,
#                         'recognized': False
#                     }
#                     self.next_id += 1
#                 self.results[face_id].update({
#                     'box': box,
#                     'last_seen': self.frame_count
#                 })
#             if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
#                 try:
#                     self.align_queue.put_nowait((frame.copy(), box, face_id))
#                 except queue.Full:
#                     pass
#             result = self.results[face_id]
#             if result['recognized']:
#                 color = (255, 0, 0)
#                 label = f"Recognized: {result['name']}"
                
#                 # Save the frame inside the bounding box if the save_frame_interval condition is met
#                 if self.frame_count % self.save_frame_interval == 0:
#                     x1, y1, x2, y2 = map(int, box)
#                     face_crop = frame[y1:y2, x1:x2]
#                     if face_crop.size != 0:
#                         uid = result.get('uid', 'unknown')
#                         if isinstance(uid, np.ndarray):  # Convert uid to string if it's a NumPy array
#                             uid = uid.item() if uid.size == 1 else str(uid)
#                         # Use the UID folder created by process_video
#                         uid_folder = os.path.join(self.records_dir, uid)
#                         os.makedirs(uid_folder, exist_ok=True)
#                         frame_path = os.path.join(uid_folder, f"{self.frame_count}.jpg")
#                         cv2.imwrite(frame_path, face_crop)
#             elif result['name'] != "Unknown":
#                 color = (0, 255, 0)
#                 label = f"{result['name']}, score: {result['similarity']:.2f}"
#             else:
#                 color = (0, 0, 255)
#                 label = "Unknown"
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         if self.frame_count % 30 == 0:
#             with self.lock:
#                 self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
#         return frame

#     def cleanup(self):
#         """Stop the pipeline and release resources."""
#         self.running = False
#         cv2.destroyAllWindows()

