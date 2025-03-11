import cv2  # Detailed: Import OpenCV for image and video processing operations.
           # Simple: Lets us work with videos and pictures.
import torch  # Detailed: Import PyTorch for tensor computations and deep learning operations.
            # Simple: Used for machine learning tasks.
import numpy as np  # Detailed: Import NumPy for numerical calculations and array operations.
                   # Simple: Helps with math and arrays.
import threading  # Detailed: Import the threading module to run multiple operations concurrently.
                  # Simple: Lets the program do several things at once.
import queue  # Detailed: Import the queue module to manage tasks safely between threads.
             # Simple: Helps share work between threads.
import os  # Detailed: Import the OS module for handling file and directory operations.
         # Simple: Lets us work with files and folders.
from torch.nn.functional import cosine_similarity  
# Detailed: Import the cosine_similarity function from PyTorch to compare two tensors by computing their cosine similarity.
# Simple: Helps compare two face codes to see how similar they are.
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform  
# Detailed: Import custom utility functions for face operations: detect_faces for locating faces, align_face for standardizing face images, load_face_recognition_model for loading the pre-trained model, and transform for preprocessing images.
# Simple: Gets functions to find, fix, load the model, and prepare face images.

class FaceRecognitionPipeline:
    def __init__(self):
        # Detailed: Initialize the face recognition pipeline by loading the model, retrieving saved embeddings, and starting worker threads.
        # Simple: Set up the system: load model, get saved face codes, and start background tasks.
        self.model, self.device = load_face_recognition_model()
        # Detailed: Load a pre-trained face recognition model and the device (CPU/GPU) on which to run it.
        # Simple: Load the face model and choose which computer part (CPU/GPU) to use.
        face_database_dir = './face_database/'
        # Detailed: Specify the directory where face embeddings (.npz files) are stored.
        # Simple: Folder where saved face codes are kept.
        self.saved_embeddings = []
        self.saved_labels = []
        # Detailed: Initialize empty lists to store the loaded face embeddings and their corresponding labels.
        # Simple: Create empty lists to hold face codes and names.
        for filename in os.listdir(face_database_dir):
            # Detailed: Iterate over all files in the face database directory.
            # Simple: Look at each file in the face folder.
            if filename.endswith('.npz'):
                # Detailed: Check if the file is a .npz file, which contains saved face embeddings.
                # Simple: Only use files with the .npz extension.
                person_name = os.path.splitext(filename)[0]
                # Detailed: Extract the person’s name from the filename by removing the .npz extension.
                # Simple: Get the name by removing ".npz" from the filename.
                npz_path = os.path.join(face_database_dir, filename)
                # Detailed: Build the complete file path to the .npz file.
                # Simple: Create the full path for the file.
                data = np.load(npz_path)
                # Detailed: Load the .npz file into a variable.
                # Simple: Open the file and read its data.
                embeddings = data['embeddings']
                # Detailed: Extract the 'embeddings' array from the loaded data.
                # Simple: Get the face codes from the file.
                for embedding in embeddings:
                    # Detailed: Loop over each embedding stored in the array.
                    # Simple: For each face code in the file:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    # Detailed: Convert the embedding to a PyTorch tensor, move it to the designated device, and add it to the list of saved embeddings.
                    # Simple: Turn the face code into a tensor and save it.
                    self.saved_labels.append(person_name)
                    # Detailed: Append the corresponding person’s name to the list of saved labels.
                    # Simple: Save the person's name.
        
        self.align_queue = queue.Queue(maxsize=5)
        # Detailed: Create a thread-safe queue to hold tasks for face alignment with a maximum of 5 items.
        # Simple: Set up a queue for face alignment tasks, limited to 5 items.
        self.recog_queue = queue.Queue(maxsize=5)
        # Detailed: Create another thread-safe queue for face recognition tasks with a maximum size of 5.
        # Simple: Set up a queue for face recognition tasks.
        self.results = {}
        # Detailed: Initialize a dictionary to store recognition results for each detected face, keyed by a unique face ID.
        # Simple: Make a dictionary to hold results for each face.
        self.next_id = 0
        # Detailed: Initialize a counter to assign unique IDs to new faces.
        # Simple: Start counting new faces from zero.
        self.frame_count = 0
        # Detailed: Initialize a counter to keep track of the number of frames processed.
        # Simple: Count how many frames we’ve seen.
        self.running = True
        # Detailed: Set a flag to indicate that the pipeline is running; used to control the worker threads.
        # Simple: A flag to tell the threads to keep working.
        self.lock = threading.Lock()
        # Detailed: Create a lock to ensure that shared data (like the results dictionary) is accessed safely by multiple threads.
        # Simple: A lock to stop threads from interfering with each other.
        
        # Start alignment and recognition worker threads
        threading.Thread(target=self._alignment_worker, daemon=True).start()
        # Detailed: Launch a background (daemon) thread to run the _alignment_worker method continuously.
        # Simple: Start a thread to align faces.
        threading.Thread(target=self._recognition_worker, daemon=True).start()
        # Detailed: Launch another daemon thread to run the _recognition_worker method continuously.
        # Simple: Start a thread to recognize faces.

    def _alignment_worker(self):
        """Thread worker for aligning faces before recognition."""
        # Detailed: This worker continuously retrieves tasks from the alignment queue, crops and aligns detected face regions, and then passes them to the recognition queue.
        # Simple: This thread takes face parts from a queue, fixes their orientation/size, and sends them for recognition.
        while self.running:
            try:
                frame, box, face_id = self.align_queue.get()
                # Detailed: Retrieve a tuple (frame, bounding box, face ID) from the alignment queue.
                # Simple: Get a task from the alignment queue.
                if frame is None:
                    continue
                    # Detailed: If the frame is None, skip to the next task.
                    # Simple: If there's no frame, skip it.
                x1, y1, x2, y2 = map(int, box)
                # Detailed: Convert bounding box coordinates to integers.
                # Simple: Get the box coordinates as whole numbers.
                face_crop = frame[y1:y2, x1:x2]
                # Detailed: Crop the region of the frame defined by the bounding box.
                # Simple: Cut out the face area from the frame.
                if face_crop.size == 0:
                    continue
                    # Detailed: If the cropped image is empty, skip processing this task.
                    # Simple: If the face crop is empty, move on.
                aligned_face = align_face(face_crop)
                # Detailed: Align the cropped face using the align_face function to standardize its orientation and size.
                # Simple: Fix the face image so it's ready for the model.
                if aligned_face is not None:
                    self.recog_queue.put((aligned_face, face_id))
                    # Detailed: If the alignment is successful, put the aligned face and its face ID into the recognition queue.
                    # Simple: If the face looks good, send it to be recognized.
                self.align_queue.task_done()
                # Detailed: Mark the current task as done in the alignment queue.
                # Simple: Tell the queue this task is finished.
            except Exception as e:
                print(f"Alignment error: {e}")
                # Detailed: If an error occurs, print the error message for debugging.
                # Simple: Show an error message if something goes wrong.

    def _recognition_worker(self):
        """Thread worker for face recognition."""
        # Detailed: This worker continuously processes tasks from the recognition queue by transforming the aligned face, computing its embedding, comparing it with saved embeddings, and updating the results.
        # Simple: This thread takes fixed face images, gets their codes, compares them to saved ones, and updates the results.
        while self.running:
            try:
                aligned_face, face_id = self.recog_queue.get()
                # Detailed: Retrieve an aligned face and its corresponding face ID from the recognition queue.
                # Simple: Get a task from the recognition queue.
                if aligned_face is None:
                    continue
                    # Detailed: If the aligned face is None, skip this task.
                    # Simple: If the face image is missing, skip it.
                face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
                # Detailed: Apply preprocessing to the aligned face image using the transform function, add a batch dimension with unsqueeze, and transfer the tensor to the correct device.
                # Simple: Prepare the face image for the model by converting it to a tensor and moving it to the right device.
                embedding = self.model(face_tensor).detach()
                # Detailed: Pass the processed face tensor through the model to obtain its embedding, and detach it from the computation graph.
                # Simple: Get the face code from the model.
                similarities = [cosine_similarity(saved_emb, embedding).item() for saved_emb in self.saved_embeddings]
                # Detailed: Compute cosine similarities between the computed embedding and each saved embedding, resulting in a list of similarity scores.
                # Simple: Compare the new face code with all saved ones to see how similar they are.
                best_idx = np.argmax(similarities)
                # Detailed: Determine the index of the highest similarity score.
                # Simple: Find the closest match.
                best_score = similarities[best_idx]
                # Detailed: Retrieve the best (highest) similarity score.
                # Simple: Get the best match's score.
                with self.lock:
                    if face_id in self.results:
                        self.results[face_id].update({
                            'name': self.saved_labels[best_idx] if best_score > 0.5 else "Unknown",
                            'similarity': best_score,
                            'last_seen': self.frame_count
                        })
                        # Detailed: Update the result for the given face_id: assign the recognized name if the similarity exceeds 0.5; otherwise, label as "Unknown". Also, update the similarity score and record the last seen frame.
                        # Simple: Update the face's record with a name (if similar enough) and the score, plus when it was last seen.
                self.recog_queue.task_done()
                # Detailed: Mark the current recognition task as completed.
                # Simple: Tell the queue this task is finished.
            except Exception as e:
                print(f"Recognition error: {e}")
                # Detailed: If an error occurs during recognition, print the error message.
                # Simple: Show an error if something goes wrong.

    def process_frame(self, frame):
        """Detect, align, and recognize faces in a video frame."""
        # Detailed: Process a single video frame by detecting faces, updating results, queuing tasks for alignment and recognition, and drawing annotations on the frame.
        # Simple: Look at a video frame, find faces, update their info, and draw boxes and labels.
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            # Detailed: If the frame is not a valid NumPy array or is empty, return it unmodified.
            # Simple: If the frame isn’t good, just return it.
        self.frame_count += 1
        # Detailed: Increment the frame counter for every frame processed.
        # Simple: Add one to the frame count.
        boxes, _ = detect_faces(frame)
        # Detailed: Detect faces in the frame using the detect_faces function, which returns bounding boxes and additional information.
        # Simple: Find faces in the frame.
        if len(boxes) == 0:
            return frame
            # Detailed: If no faces are detected, return the original frame.
            # Simple: If no faces are found, do nothing and return the frame.
        for box in boxes:
            # Detailed: Iterate over each detected face bounding box.
            # Simple: For every detected face:
            face_id = next(
                (fid for fid, data in self.results.items() if np.linalg.norm(np.array(data['box']) - box) < 50), 
                f"face_{self.next_id}"
            )
            # Detailed: Determine a unique face ID by checking if an existing face in the results has a similar bounding box (within a threshold of 50); otherwise, assign a new face ID.
            # Simple: Use an existing face ID if the box is similar, or create a new one.
            with self.lock:
                if face_id not in self.results:
                    self.results[face_id] = {
                        'name': "Unknown",
                        'similarity': 0.0,
                        'box': box,
                        'last_seen': self.frame_count
                    }
                    # Detailed: If this face is new, initialize its record in the results dictionary with default values.
                    # Simple: If this face is new, add it to the results with default info.
                    self.next_id += 1
                    # Detailed: Increment the counter for future new faces.
                    # Simple: Increase the face counter.
                self.results[face_id].update({
                    'box': box,
                    'last_seen': self.frame_count
                })
                # Detailed: Update the record for the face with the current bounding box and frame count.
                # Simple: Refresh the face's position and last seen time.
            if self.frame_count % 5 == 0:
                # Detailed: Every 5 frames, queue the current face for alignment processing.
                # Simple: Every 5 frames, send the face for alignment.
                try:
                    self.align_queue.put_nowait((frame.copy(), box, face_id))
                    # Detailed: Attempt to add a tuple (copy of the frame, bounding box, face ID) to the alignment queue without waiting.
                    # Simple: Try to add the face data to the alignment queue immediately.
                except queue.Full:
                    pass
                    # Detailed: If the alignment queue is full, skip adding this task.
                    # Simple: If the queue is full, do nothing.
            result = self.results[face_id]
            # Detailed: Retrieve the current result record for the face.
            # Simple: Get the saved info for this face.
            color = (0, 255, 0) if result['name'] != "Unknown" else (0, 0, 255)
            # Detailed: Set the annotation color: green if the face is recognized, red if unknown.
            # Simple: Use green if known, red if unknown.
            x1, y1, x2, y2 = map(int, box)
            # Detailed: Convert the bounding box coordinates to integers.
            # Simple: Get the box coordinates as whole numbers.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Detailed: Draw a rectangle around the face on the frame using the specified color and a thickness of 2.
            # Simple: Draw a box around the face.
            cv2.putText(frame, f"{result['name']}, score: {result['similarity']:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Detailed: Overlay text above the bounding box displaying the face's name and similarity score.
            # Simple: Write the name and score near the face box.
        if self.frame_count % 30 == 0:
            # Detailed: Every 30 frames, remove face records that haven't been updated in the last 15 frames.
            # Simple: Every 30 frames, delete faces not seen recently.
            with self.lock:
                self.results = {fid: data for fid, data in self.results.items() if self.frame_count - data['last_seen'] <= 15}
                # Detailed: Filter the results dictionary to keep only those entries where the face was seen in the last 15 frames.
                # Simple: Only keep faces that were seen recently.
        return frame
        # Detailed: Return the annotated frame.
        # Simple: Give back the updated frame.

    def cleanup(self):
        """Stop the pipeline and release resources."""
        # Detailed: Stop the running threads and close all OpenCV windows to release resources.
        # Simple: Stop the system and close any open windows.
        self.running = False
        # Detailed: Set the running flag to False to signal all worker threads to exit.
        # Simple: Tell the threads to stop.
        cv2.destroyAllWindows()
        # Detailed: Close any windows created by OpenCV.
        # Simple: Close all video windows.

def main():
    pipeline = FaceRecognitionPipeline()
    # Detailed: Create an instance of the FaceRecognitionPipeline, initializing the model, queues, and worker threads.
    # Simple: Start the face recognition system.
    cap = cv2.VideoCapture(0)
    # Detailed: Open the default camera (device index 0) to capture video frames.
    # Simple: Turn on the webcam.

    try:
        while cap.isOpened():
            # Detailed: Continuously read frames from the video capture while it remains open.
            # Simple: Keep getting frames from the camera.
            ret, frame = cap.read()
            # Detailed: Read a frame from the camera; ret indicates if the frame was read successfully.
            if not ret:
                break
                # Detailed: If a frame could not be read, exit the loop.
                # Simple: If no frame is returned, stop the loop.
            processed = pipeline.process_frame(frame)
            # Detailed: Process the captured frame through the pipeline to detect, align, and recognize faces.
            # Simple: Run the frame through the face recognition system.
            if processed is not None:
                cv2.imshow('Face Recognition', processed)
                # Detailed: Display the processed frame in a window titled 'Face Recognition'.
                # Simple: Show the updated frame on the screen.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # Detailed: Wait for 1 ms for a key press; if the 'q' key is pressed, exit the loop.
                # Simple: If you press 'q', stop the program.
    finally:
        pipeline.cleanup()
        # Detailed: Ensure that the pipeline is cleaned up by stopping worker threads and closing windows.
        # Simple: Clean up the face recognition system.
        cap.release()
        # Detailed: Release the video capture resource.
        # Simple: Turn off the webcam.

if __name__ == "__main__":
    main()
    # Detailed: If the script is executed as the main program, run the main function to start the pipeline.
    # Simple: When you run this file, start the program.
