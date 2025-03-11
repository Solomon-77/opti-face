import cv2  # Detailed: Import OpenCV for image and video processing operations.
           # Simple: Lets us work with videos and pictures.
import torch  # Detailed: Import PyTorch for tensor computations and deep learning operations.
            # Simple: Used for machine learning tasks.
import numpy as np  # Detailed: Import NumPy for numerical operations and array handling.
                   # Simple: Helps with math and arrays.
import threading  # Detailed: Import threading to allow running multiple tasks concurrently.
                  # Simple: Lets the program do several things at once.
import queue  # Detailed: Import the queue module to safely share tasks between threads.
             # Simple: Helps share work between different threads.
import os  # Detailed: Import the OS module to interact with the file system (files and directories).
         # Simple: Lets us work with files and folders.
from tkinter import Tk, StringVar, Canvas, ttk, Toplevel  
# Detailed: Import Tkinter components for building the graphical user interface (GUI).
# Simple: These help create windows, text boxes, buttons, etc.
from prepare_embeddings_system_v1 import FaceEmbeddingApp  
# Detailed: Import a custom FaceEmbeddingApp class used for managing face embeddings.
# Simple: Lets us manage saved face codes using another module.
import sv_ttk  # Detailed: Import sv_ttk to set a modern dark theme for the Tkinter GUI.
              # Simple: Gives the GUI a cool dark look.
from PIL import Image, ImageTk  # Detailed: Import PIL (Pillow) to convert OpenCV images for display in Tkinter.
                               # Simple: Helps change pictures so they work in the GUI.
from torch.nn.functional import cosine_similarity  
# Detailed: Import the cosine_similarity function from PyTorch to compare face embeddings.
# Simple: Compares how similar two face codes are.
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform  
# Detailed: Import custom utility functions for detecting, aligning, and transforming face images, and for loading a pre-trained face recognition model.
# Simple: Gets functions to find faces, fix their orientation, prepare images, and load the face model.

# NOTE: This version uses asynchronous processing with multithreading to align and recognize faces concurrently.

class FaceRecognitionPipeline:
    def __init__(self):
        # Detailed: Initialize the pipeline by loading the face recognition model, setting up the face database, and starting worker threads.
        # Simple: Set up the face recognition system by loading the model, saved face codes, and starting background tasks.
        self.model, self.device = load_face_recognition_model()
        # Detailed: Load a pre-trained face recognition model and determine the device (CPU or GPU) for computation.
        # Simple: Load the face model and choose the correct computer part.
        self.face_database_dir = './face_database/'
        # Detailed: Define the directory that contains the saved face embeddings.
        # Simple: Folder where saved face codes are kept.
        self.saved_embeddings = []  # Detailed: Initialize a list to hold face embedding tensors loaded from storage.
        # Simple: List for storing face codes.
        self.saved_labels = []  # Detailed: Initialize a list to hold labels (person names) corresponding to each embedding.
                             # Simple: List for storing names.
        self.load_embeddings()  # Detailed: Load embeddings and labels from the face database files.
                               # Simple: Read saved face codes from the folder.

        # Create queues for asynchronous processing
        self.align_queue = queue.Queue(maxsize=5)
        # Detailed: Create a thread-safe queue for face alignment tasks with a maximum of 5 items.
        # Simple: Set up a queue for face alignment work.
        self.recog_queue = queue.Queue(maxsize=5)
        # Detailed: Create a thread-safe queue for face recognition tasks with a maximum of 5 items.
        # Simple: Set up a queue for face recognition work.
        self.results = {}  # Detailed: Initialize a dictionary to store recognition results for each detected face.
                         # Simple: Holds info about each face found.
        self.next_id = 0  # Detailed: Initialize a counter for generating unique IDs for new faces.
                        # Simple: Start counting new faces.
        self.frame_count = 0  # Detailed: Counter for the number of processed video frames.
                           # Simple: Count how many frames we’ve seen.
        self.running = True  # Detailed: Flag to control the execution of worker threads.
                           # Simple: Tells threads to keep running.
        self.lock = threading.Lock()  # Detailed: Create a lock to synchronize access to shared resources among threads.
                                    # Simple: Prevents threads from interfering with each other.

        # Configuration values for recognition thresholds
        self.min_accuracy = 0.5  # Detailed: Set the minimum accuracy threshold for initial inference.
                               # Simple: Minimum score to consider a face somewhat similar.
        self.min_recognize = 0.8  # Detailed: Set the minimum accuracy threshold required to mark a face as recognized.
                                # Simple: Minimum score to call a face recognized.

        # Start asynchronous worker threads for alignment and recognition
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        # Detailed: Start a daemon thread that continuously runs the _alignment_worker to process face alignment tasks.
        # Simple: Begin a background thread to fix face images.
        threading.Thread(target=self._recognition_worker, daemon=True).start()
        # Detailed: Start a daemon thread that continuously runs the _recognition_worker to process face recognition tasks.
        # Simple: Begin another background thread to compare face codes.

    def load_embeddings(self):
        """Load embeddings from the face database directory."""
        # Detailed: Load saved face embeddings and corresponding labels from .npz files in the face database directory.
        # Simple: Read saved face codes and names from the folder.
        self.saved_embeddings = []
        self.saved_labels = []
        for filename in os.listdir(self.face_database_dir):
            # Detailed: Iterate over each file in the face database directory.
            # Simple: Look at every file in the folder.
            if filename.endswith('.npz'):
                # Detailed: Process only files with the .npz extension.
                # Simple: Only use files ending in .npz.
                person_name = os.path.splitext(filename)[0]
                # Detailed: Extract the person’s name from the filename by removing the file extension.
                # Simple: Get the name from the filename.
                npz_path = os.path.join(self.face_database_dir, filename)
                # Detailed: Create the full file path to the .npz file.
                # Simple: Build the complete path to the file.
                data = np.load(npz_path)
                # Detailed: Load the .npz file's data.
                # Simple: Open the file to get its contents.
                embeddings = data['embeddings']
                # Detailed: Retrieve the array of embeddings stored in the file.
                # Simple: Get the face codes from the file.
                for embedding in embeddings:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    # Detailed: Convert each embedding into a PyTorch tensor, move it to the correct device, and save it.
                    # Simple: Turn the face code into a tensor and save it.
                    self.saved_labels.append(person_name)
                    # Detailed: Save the corresponding person’s name.
                    # Simple: Store the name.

        # Reinitialize queues and related variables (useful when embeddings are reloaded)
        self.align_queue = queue.Queue(maxsize=5)
        self.recog_queue = queue.Queue(maxsize=5)
        self.results = {}
        self.next_id = 0
        self.frame_count = 0
        self.running = True
        self.lock = threading.Lock()
        # Reapply configuration values
        self.min_accuracy = 0.5
        self.min_recognize = 0.8
        # Restart worker threads for asynchronous processing
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def update_recognized_faces(self):
        """Update the recognized faces when embeddings are reloaded."""
        # Detailed: Check current results against the loaded embeddings and mark faces as "Unknown" if their names are no longer in the saved labels.
        # Simple: Update face info to "Unknown" if the saved names have changed.
        with self.lock:
            current_labels = set(self.saved_labels)
            for face_id, data in list(self.results.items()):
                if data['name'] not in current_labels:
                    self.results[face_id].update({
                        'name': "Unknown",
                        'similarity': 0.0,
                        'recognized': False
                    })

    def _alignment_worker(self):
        """Thread worker for aligning faces before recognition."""
        # Detailed: Continuously process items from the alignment queue to crop and align face images before sending them for recognition.
        # Simple: This thread takes face images, fixes them, and sends them for comparison.
        while self.running:
            try:
                frame, box, face_id = self.align_queue.get()
                # Detailed: Retrieve a task containing the frame, face bounding box, and a unique face identifier.
                # Simple: Get a job from the alignment queue.
                if frame is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                # Detailed: Convert the bounding box coordinates to integers.
                # Simple: Get the face box coordinates as whole numbers.
                face_crop = frame[y1:y2, x1:x2]
                # Detailed: Crop the face region from the frame using the bounding box.
                # Simple: Cut out the face from the frame.
                if face_crop.size == 0:
                    continue
                aligned_face = align_face(face_crop)
                # Detailed: Align the cropped face using a dedicated alignment function to standardize its orientation and size.
                # Simple: Fix the face image so it can be compared properly.
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))
                    # Detailed: If alignment succeeds, add the aligned face along with its face ID to the recognition queue.
                    # Simple: If the face is good, send it for recognition.
                self.align_queue.task_done()
                # Detailed: Mark the current alignment task as completed.
                # Simple: Tell the queue that the job is done.
            except Exception as e:
                print(f"Alignment error: {e}")
                # Detailed: Print any error encountered during face alignment for debugging.
                # Simple: Show an error if something goes wrong in alignment.

    def _recognition_worker(self):
        """Thread worker for face recognition."""
        # Detailed: Continuously process items from the recognition queue by computing face embeddings and comparing them to saved embeddings.
        # Simple: This thread takes fixed face images, gets their face codes, and compares them to saved ones.
        while self.running:
            try:
                aligned_face, face_id = self.recog_queue.get()
                # Detailed: Retrieve an aligned face and its corresponding face ID from the recognition queue.
                # Simple: Get a job from the recognition queue.
                if aligned_face is None:
                    continue
                face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
                # Detailed: Preprocess the aligned face image to create a tensor suitable for the model (add batch dimension and move to device).
                # Simple: Prepare the face image for the model.
                embedding = self.model(face_tensor).detach()
                # Detailed: Compute the face embedding by passing the tensor through the model and detaching the output.
                # Simple: Get the face code from the model.
                similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
                # Detailed: Calculate cosine similarity between the computed embedding and each saved embedding.
                # Simple: Compare the new face code with all saved ones.
                best_idx = np.argmax(similarities)
                # Detailed: Identify the index with the highest similarity score.
                # Simple: Find the closest match.
                best_score = similarities[best_idx]
                # Detailed: Retrieve the best similarity score.
                # Simple: Get the best matching score.
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
                                'recognized': False  # Not yet confidently recognized
                            })
                        else:
                            self.results[face_id].update({
                                'name': "Unknown",
                                'similarity': best_score,
                                'last_seen': self.frame_count,
                                'recognized': False  # Unknown face
                            })
                self.recog_queue.task_done()
                # Detailed: Mark the current recognition task as complete.
                # Simple: Let the queue know the job is done.
            except Exception as e:
                print(f"Recognition error: {e}")
                # Detailed: Print any errors encountered during face recognition.
                # Simple: Show an error if recognition fails.

    def process_frame(self, frame):
        """Detect, align, and recognize faces in a video frame."""
        # Detailed: Process each video frame by detecting faces, queuing them for alignment/recognition, and annotating the frame with results.
        # Simple: Look at a frame, find faces, update their info, and draw boxes with names.
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
        self.frame_count += 1  # Detailed: Increment the frame counter to track processing.
                              # Simple: Count the frame.
        boxes, _ = detect_faces(frame)
        # Detailed: Use the detect_faces function to locate faces in the frame, returning bounding boxes.
        # Simple: Find face locations in the frame.
        if len(boxes) == 0:
            return frame
        for box in boxes:
            # Detailed: For each detected face bounding box, determine a unique face ID.
            # Simple: For every face found:
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
            # If it's time to align and the face hasn't been confidently recognized, queue it for processing
            if self.frame_count % 5 == 0 and not self.results[face_id]['recognized']:
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                    # Detailed: Attempt to add the current face's data to the alignment queue without waiting.
                    # Simple: Try to send the face for alignment.
                except queue.Full:
                    pass
            result = self.results[face_id]
            # Determine annotation color and label based on recognition status
            if result['recognized']:
                color = (255, 0, 0)  # Detailed: Use blue color for confidently recognized faces.
                                   # Simple: Blue means recognized.
                label = f"Recognized: {result['name']}"
            elif result['name'] != "Unknown":
                color = (0, 255, 0)  # Detailed: Use green color for faces with a moderate match.
                                   # Simple: Green if a name is assigned.
                label = f"{result['name']}, score: {result['similarity']:.2f}"
            else:
                color = (0, 0, 255)  # Detailed: Use red color for unknown faces.
                                   # Simple: Red means unknown.
                label = "Unknown"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Detailed: Draw a rectangle around the face.
            # Simple: Draw a box around the face.
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Detailed: Overlay the label text above the bounding box.
            # Simple: Write the name and score near the face.
        # Periodically remove old face records that haven't been updated recently
        if self.frame_count % 30 == 0:
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
        return frame

    def cleanup(self):
        """Stop the pipeline and release resources."""
        # Detailed: Set the running flag to False to stop all worker threads and close OpenCV windows.
        # Simple: Tell threads to stop and close all video windows.
        self.running = False
        cv2.destroyAllWindows()

class App:
    def __init__(self, window, window_title, pipeline):
        # Detailed: Initialize the Tkinter GUI application, configure side panels, and set up the video display canvas.
        # Simple: Set up the window, buttons, and video area.
        self.window = window
        window.resizable(False, False)
        self.window.title(window_title)
        self.pipeline = pipeline

        sv_ttk.set_theme("dark")  # Detailed: Set the Sun Valley dark theme for the GUI.
                                  # Simple: Make the GUI look dark.
        # Create a side panel for configuration controls
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
        # Button to open the embedding manager window
        self.manage_embeddings_button = ttk.Button(self.side_panel, text="Manage Embeddings", command=self.open_embedding_manager)
        self.manage_embeddings_button.pack(pady=10)

        # Create a canvas for video display
        self.canvas = Canvas(window, width=640, height=480)
        self.canvas.pack()
        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update()  # Begin updating the GUI with video frames

    def refresh_config(self):
        """Update the configuration values from the entry fields."""
        # Detailed: Update the pipeline's configuration thresholds based on the user's input from the GUI.
        # Simple: Read new settings from the boxes and update the system.
        try:
            self.pipeline.min_accuracy = float(self.min_accuracy_var.get())
            self.pipeline.min_recognize = float(self.min_recognize_var.get())
            print(f"Config updated: Min Accuracy = {self.pipeline.min_accuracy}, Min Recognize = {self.pipeline.min_recognize}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    def update(self):
        """Update the OpenCV display in the tkinter window."""
        # Detailed: Read a frame from the video capture, process it through the pipeline, and update the Tkinter canvas with the processed image.
        # Simple: Get a frame from the webcam, process it, and show it in the window.
        ret, frame = self.cap.read()
        if ret:
            frame = self.pipeline.process_frame(frame)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.window.after(10, self.update)  # Schedule the next update in 10 milliseconds

    def cleanup(self):
        """Release resources."""
        # Detailed: Clean up by stopping the pipeline and releasing the video capture.
        # Simple: Stop the face recognition system and turn off the webcam.
        self.pipeline.cleanup()
        self.cap.release()

    def open_embedding_manager(self):
        """Open the FaceEmbeddingApp to manage embeddings."""
        # Detailed: Open a new top-level window and launch the FaceEmbeddingApp for managing face embeddings.
        # Simple: Open a new window to manage saved face codes.
        embedding_window = Toplevel(self.window)
        FaceEmbeddingApp(embedding_window, self.pipeline)

def main():
    pipeline = FaceRecognitionPipeline()
    # Detailed: Create an instance of the FaceRecognitionPipeline, initializing the model, queues, and worker threads.
    # Simple: Start the face recognition system.
    root = Tk()
    sv_ttk.use_dark_theme()  # Detailed: Initialize the Sun Valley dark theme for the GUI.
                              # Simple: Use the dark look for the window.
    app = App(root, "Facial Recognition System", pipeline)
    root.mainloop()  # Detailed: Enter the Tkinter main loop to run the GUI.
                    # Simple: Start the window.
    app.cleanup()  # Detailed: After the GUI is closed, clean up resources.
                   # Simple: Stop the system.

if __name__ == "__main__":
    main()  # Detailed: If this script is executed as the main program, run the main function.
           # Simple: Start the program.
