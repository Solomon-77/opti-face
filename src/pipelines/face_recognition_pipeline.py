import cv2  # Detailed: Import OpenCV for image processing (reading, drawing, cropping, etc.).  
         # Simple: Lets us work with images.
import torch  # Detailed: Import PyTorch for tensor operations and deep learning model inference.  
           # Simple: Used for machine learning tasks.
import numpy as np  # Detailed: Import NumPy for numerical computations and array manipulations.  
                   # Simple: Helps with math and arrays.
import queue  # Detailed: Import the queue module to safely pass tasks between threads.  
             # Simple: Lets threads share work.
import threading  # Detailed: Import threading to run multiple tasks concurrently.  
                  # Simple: Allows the program to do several things at once.
import os  # Detailed: Import the os module to interact with the operating system (e.g., file paths, directory creation).  
         # Simple: Lets us work with files and folders.
from torch.nn.functional import cosine_similarity  # Detailed: Import cosine_similarity to measure similarity between face embeddings.  
                                                   # Simple: Compares how similar two face codes are.
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform  
# Detailed: Import custom functions for face detection, alignment, model loading, and image transformation.  
# Simple: Get functions to find faces, fix face images, load the face model, and prepare images.
from utils.path_patch_v1 import get_resource_path  
# Detailed: Import a helper function to obtain the correct file path, whether running as a script or an executable.  
# Simple: Helps find the right folder for files.

class FaceRecognitionPipeline:
    def __init__(self):
        # Detailed: Initialize the face recognition pipeline by loading the model, setting up directories, loading saved embeddings, 
        # and initializing threading queues and configuration parameters.
        # Simple: Set up the face recognition system, load saved data, and start background threads.
        self.model, self.device = load_face_recognition_model()  
        # Detailed: Load the pre-trained face recognition model and determine the device (GPU/CPU) to use.
        # Simple: Load the face model and choose GPU or CPU.
        self.face_database_dir = get_resource_path('face_database')  
        # Detailed: Get the absolute path to the 'face_database' folder, handling both development and executable modes.
        # Simple: Set the folder for saved face data.
        self.records_dir = os.path.join(self.face_database_dir, 'records')  
        # Detailed: Define a subdirectory 'records' inside the face database to save face frames.
        # Simple: Folder for saving face images.
        os.makedirs(self.records_dir, exist_ok=True)  
        # Detailed: Create the records directory if it does not exist.
        # Simple: Make sure the folder exists.
        
        self.saved_embeddings = []  # Detailed: List to store face embeddings loaded from files.
        # Simple: List for face codes.
        self.saved_labels = []  # Detailed: List to store corresponding labels (names) for each embedding.
        # Simple: List for names.
        self.saved_uids = []  # Detailed: List to store unique identifiers (UIDs) for each person.
        # Simple: List for unique IDs.
        self.load_embeddings()  # Detailed: Load saved embeddings, labels, and UIDs from the face database.
        # Simple: Read saved face codes from files.
        self.frame_counters = {}  # Detailed: Dictionary to track the number of saved frames per UID for sequential naming.
        # Simple: Count saved frames for each person.
        
        # Create queues for asynchronous processing
        self.align_queue = queue.Queue(maxsize=5)  # Detailed: Queue for tasks that involve aligning detected faces.
        # Simple: Queue for face alignment work.
        self.recog_queue = queue.Queue(maxsize=5)  # Detailed: Queue for tasks that involve recognizing aligned faces.
        # Simple: Queue for face recognition work.
        self.results = {}  # Detailed: Dictionary to store recognition results for each detected face, keyed by face_id.
        # Simple: Holds info for each face.
        self.next_id = 0  # Detailed: Counter to assign unique IDs to new detected faces.
        # Simple: ID counter for faces.
        self.frame_count = self.get_highest_frame_number()  
        # Detailed: Initialize the frame counter with the highest frame number already saved in records, ensuring sequential numbering.
        # Simple: Start counting frames from the highest saved number.
        self.running = True  # Detailed: Flag to control the execution of background worker threads.
        # Simple: Tells threads to keep running.
        self.lock = threading.Lock()  # Detailed: Lock to ensure thread-safe access to shared resources like results.
        # Simple: Prevents threads from interfering with each other.
        
        # Configuration values
        self.min_accuracy = 0.5  # Detailed: Minimum similarity score to consider a face match during inference.
        # Simple: Minimum score to start considering a match.
        self.min_recognize = 0.8  # Detailed: Minimum similarity score required to mark a face as confidently recognized.
        # Simple: Minimum score to label a face as recognized.
        self.save_frame_interval = 60  # Detailed: Interval (in frames) at which a frame of a recognized face is saved.
        # Simple: Save a face image every 60 frames.
        
        # Start worker threads for asynchronous face alignment and recognition
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        # Detailed: Launch a daemon thread that continuously processes face alignment tasks from the align_queue.
        # Simple: Start a background thread to align faces.
        threading.Thread(target=self._recognition_worker, daemon=True).start()
        # Detailed: Launch a daemon thread that continuously processes face recognition tasks from the recog_queue.
        # Simple: Start another background thread to recognize faces.

    def get_highest_frame_number(self):
        """Find the highest frame number in the records directory."""
        # Detailed: Iterate over all UID folders in the records directory, and for each, determine the highest numbered frame file.
        # Simple: Look for the highest frame number saved so far.
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
        # Detailed: Reset and load saved embeddings, labels, and UIDs from all .npz files in the face_database directory.
        # Simple: Read saved face codes, names, and unique IDs from files.
        self.saved_embeddings = []
        self.saved_labels = []
        self.saved_uids = []
        for filename in os.listdir(self.face_database_dir):
            if filename.endswith('.npz'):
                npz_path = os.path.join(self.face_database_dir, filename)
                data = np.load(npz_path)
                embeddings = data['embeddings']
                person_name = data.get('person_name', os.path.splitext(filename)[0])
                uid = data.get('uid', os.path.splitext(filename)[0])  # Load UID if available; default to filename without extension.
                for embedding in embeddings:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    self.saved_labels.append(person_name)
                    self.saved_uids.append(uid)

    def _alignment_worker(self):
        """Thread worker for aligning faces before recognition."""
        # Detailed: Continuously retrieve tasks from the align_queue, crop the face region, align it using landmarks, and then put the aligned face into the recog_queue.
        # Simple: Get a face from the queue, fix its orientation, and send it for recognition.
        while self.running:
            try:
                frame, box, face_id = self.align_queue.get()
                if frame is None:
                    continue
                x1, y1, x2, y2 = map(int, box)  # Detailed: Convert the bounding box coordinates to integers.
                                               # Simple: Get the box corners as whole numbers.
                face_crop = frame[y1:y2, x1:x2]  # Detailed: Crop the face region from the frame using the bounding box coordinates.
                                                # Simple: Cut out the face from the frame.
                if face_crop.size == 0:
                    continue
                aligned_face = align_face(face_crop)  # Detailed: Align the cropped face image to a canonical pose using facial landmarks.
                                                      # Simple: Adjust the face image so its features line up correctly.
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))  # Detailed: Put the aligned face and its identifier into the recognition queue.
                                                                   # Simple: Send the fixed face for recognition.
                self.align_queue.task_done()  # Detailed: Mark the alignment task as complete in the queue.
                                              # Simple: Tell the queue the job is done.
            except Exception as e:
                print(f"Alignment error: {e}")  # Detailed: Print any errors encountered during the alignment process.
                                                # Simple: Show an error if something goes wrong in alignment.

    def _recognition_worker(self):
        """Thread worker for face recognition."""
        # Detailed: Continuously retrieve aligned faces from the recog_queue, compute their embeddings, compare them to saved embeddings using cosine similarity,
        # and update the recognition results accordingly.
        # Simple: Get a fixed face from the queue, compute its code, compare with saved codes, and update results.
        while self.running:
            try:
                aligned_face, face_id = self.recog_queue.get()
                if aligned_face is None:
                    continue

                if not self.saved_embeddings:
                    # Detailed: If no embeddings are loaded, mark the face as unknown.
                    # Simple: If there are no saved face codes, label as unknown.
                    with self.lock:
                        if face_id in self.results:
                            self.results[face_id].update({'name': "Unknown", 'similarity': 0.0, 'recognized': False})
                    self.recog_queue.task_done()
                    continue

                face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)  # Detailed: Transform the aligned face into a tensor, add a batch dimension, and send it to the device.
                                                                                    # Simple: Convert the face image to numbers for the model.
                embedding = self.model(face_tensor).detach()  # Detailed: Compute the face embedding by passing the tensor through the model and detaching it from the computation graph.
                                                             # Simple: Get the face code from the model.
                similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
                # Detailed: Compute cosine similarity between the computed embedding and each saved embedding, resulting in a list of similarity scores.
                # Simple: Compare the new face code with all saved ones.
                best_idx = np.argmax(similarities)  # Detailed: Find the index of the highest similarity score.
                                                    # Simple: Identify the best match.
                best_score = similarities[best_idx]  # Detailed: Retrieve the highest similarity score.
                                                     # Simple: Get the best matching score.
                with self.lock:
                    if face_id in self.results:
                        if best_score >= self.min_recognize:
                            # Detailed: If the similarity score exceeds the recognition threshold, update the result as recognized with the corresponding name, score, and UID.
                            # Simple: If score is high, mark as recognized and record the name.
                            self.results[face_id].update({
                                'name': self.saved_labels[best_idx],
                                'similarity': best_score,
                                'recognized': True,
                                'uid': self.saved_uids[best_idx]
                            })
                        elif best_score >= self.min_accuracy:
                            # Detailed: If the similarity score is above the minimum accuracy threshold but below the recognition threshold, update the result with the name and score, but mark as not confidently recognized.
                            # Simple: If score is moderate, record the name but do not mark as recognized.
                            self.results[face_id].update({'name': self.saved_labels[best_idx], 'similarity': best_score, 'recognized': False})
                        else:
                            # Detailed: If the similarity score is below the minimum threshold, mark the face as unknown.
                            # Simple: If score is low, label the face as unknown.
                            self.results[face_id].update({'name': "Unknown", 'similarity': best_score, 'recognized': False})
                self.recog_queue.task_done()  # Detailed: Mark the recognition task as complete in the queue.
                                               # Simple: Tell the queue the job is done.
            except Exception as e:
                print(f"Recognition error: {e}")  # Detailed: Print any errors encountered during the recognition process.
                                                  # Simple: Show an error if recognition fails.

    def process_frame(self, frame):
        """Detect, align, and recognize faces in a video frame."""
        # Detailed: Process a video frame by detecting faces, assigning or updating unique face IDs, queuing alignment tasks periodically for unrecognized faces,
        # and drawing bounding boxes and labels on the frame; also saves face images for recognized faces at defined intervals.
        # Simple: Look for faces in the frame, update their info, draw boxes, and save pictures if needed.
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame  # Detailed: If the frame is invalid or empty, return it unmodified.
                         # Simple: If the frame isn’t good, just return it.
        self.frame_count += 1  # Detailed: Increment the frame counter for each processed frame.
                              # Simple: Add one to the frame count.
        
        boxes, _ = detect_faces(frame)  # Detailed: Detect faces in the frame using the detect_faces function.
                                       # Simple: Find faces in the image.
        if len(boxes) == 0:
            return frame  # Detailed: If no faces are detected, return the original frame.
                         # Simple: If there are no faces, do nothing.
        
        for box in boxes:
            # Detailed: Determine a unique face ID by checking if an existing face's bounding box is within 50 pixels of the current box; otherwise, assign a new ID.
            # Simple: Use an existing face ID if the box is similar, or create a new one.
            face_id = next(
                (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50),
                f"face_{self.next_id}"
            )
            
            with self.lock:
                if face_id not in self.results:
                    # Detailed: If this face ID is new, initialize its record with default values.
                    # Simple: If the face is new, add it to the results.
                    self.results[face_id] = {'name': "Unknown", 'similarity': 0.0, 'box': box, 'recognized': False}
                    self.next_id += 1
                
                self.results[face_id].update({'box': box})
                # Detailed: Update the face record with the latest bounding box.
                # Simple: Refresh the face's box coordinates.
            
            # Detailed: Every 5 frames, if the face is not yet recognized, add its data to the alignment queue.
            # Simple: Every 5 frames, send unrecognized faces for alignment.
            if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                except queue.Full:
                    pass  # Detailed: If the queue is full, skip adding this task.
                           # Simple: If too many tasks, do nothing.
            
            result = self.results[face_id]
            # Detailed: Retrieve the current recognition result for this face.
            # Simple: Get the face info.
            if result['recognized']:
                # Detailed: If the face is confidently recognized, set the annotation color to blue and label accordingly.
                # Simple: If recognized, use blue and show the name.
                color = (255, 0, 0)
                label = f"Recognized: {result['name']}"
                # Detailed: Additionally, save a cropped image of the recognized face at intervals defined by save_frame_interval.
                # Simple: Save a face picture every few frames.
                if self.frame_count % self.save_frame_interval == 0:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size != 0:
                        uid = str(result.get('uid', 'unknown'))
                        uid_folder = os.path.join(self.records_dir, uid)
                        os.makedirs(uid_folder, exist_ok=True)
                        # Detailed: Ensure sequential numbering of saved frames per UID.
                        # Simple: Number the saved images in order.
                        if uid not in self.frame_counters:
                            existing_frames = [int(f.split(".")[0]) for f in os.listdir(uid_folder) if f.endswith(".jpg") and f.split(".")[0].isdigit()]
                            self.frame_counters[uid] = max(existing_frames, default=0) + 1
                        frame_filename = os.path.join(uid_folder, f"{self.frame_counters[uid]}.jpg")
                        cv2.imwrite(frame_filename, face_crop)
                        self.frame_counters[uid] += 1
            elif result['name'] != "Unknown":
                # Detailed: If the face has a name assigned but is not confidently recognized, use a green rectangle and display the similarity score.
                # Simple: If the face is somewhat known, use green and show the score.
                color = (0, 255, 0)
                label = f"{result['name']}, score: {result['similarity']:.2f}"
            else:
                # Detailed: For unknown faces, use a red rectangle and label as "Unknown".
                # Simple: If unknown, use red and label as unknown.
                color = (0, 0, 255)
                label = "Unknown"
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Detailed: Draw a rectangle around the face using the selected color.
                                                                  # Simple: Draw a box around the face.
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  
            # Detailed: Overlay the label (name or score) above the bounding box using the chosen color and font.
            # Simple: Write the face's name or score near the box.
        
        return frame  # Detailed: Return the annotated frame with detection, recognition, and saved face images if applicable.
                      # Simple: Give back the updated frame.

    def cleanup(self):
        """Stop the pipeline and release resources."""
        # Detailed: Set the running flag to False to stop all worker threads and destroy all OpenCV windows.
        # Simple: Tell threads to stop and close all video windows.
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

