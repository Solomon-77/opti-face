import cv2  # Detailed: Import OpenCV for image and video processing operations.
           # Simple: Lets us work with videos and pictures.
import torch  # Detailed: Import PyTorch for tensor computations and deep learning operations.
            # Simple: Used for machine learning tasks.
import numpy as np  # Detailed: Import NumPy for numerical operations and array handling.
                   # Simple: Helps with math and arrays.
import os  # Detailed: Import the OS module to interact with the file system (files and folders).
         # Simple: Lets us work with files and directories.
import time  # Detailed: Import the time module to perform time-related functions (e.g., FPS calculation).
           # Simple: Helps measure time.
from torch.nn.functional import cosine_similarity  
# Detailed: Import the cosine_similarity function from PyTorch to measure the similarity between two face embeddings.
# Simple: Compares how similar two face codes are.
from utils.face_utils import detect_faces, align_face, load_face_recognition_model, transform  
# Detailed: Import custom utility functions: detect_faces to locate faces in an image, align_face to standardize face images, load_face_recognition_model to load the pre-trained face model, and transform to preprocess images.
# Simple: Gets functions to find faces, fix face images, load the model, and prepare images.

# NOTE: This version is entirely synchronous and does not use asynchronous or multithreading.
#       It processes each frame one at a time.

class FaceRecognitionPipeline:
    def __init__(self):
        # Detailed: Initialize the face recognition pipeline by loading the model, device, and saved face embeddings from the database.
        # Simple: Set up the system by loading the face model and saved face codes.
        self.model, self.device = load_face_recognition_model()
        # Detailed: Load the pre-trained face recognition model and determine the computation device (CPU/GPU).
        # Simple: Load the face model and decide whether to use CPU or GPU.
        face_database_dir = './face_database/'
        # Detailed: Define the directory where face embeddings (.npz files) are stored.
        # Simple: Folder containing saved face codes.
        self.saved_embeddings = []  # Detailed: Initialize a list to store face embedding tensors loaded from files.
                                  # Simple: List for storing face codes.
        self.saved_labels = []  # Detailed: Initialize a list to store the corresponding labels (names) for each embedding.
                              # Simple: List for storing names.

        # FPS calculation variables
        self.prev_frame_time = 0  # Detailed: Variable to store the timestamp of the previous frame for FPS calculation.
                                  # Simple: Time of the last frame.
        self.curr_frame_time = 0  # Detailed: Variable to store the current frame timestamp.
                                  # Simple: Time of the current frame.
        self.fps = 0  # Detailed: Variable to store the smoothed frames per second (FPS) value.
                    # Simple: Stores the frame rate.

        # Load face database
        for filename in os.listdir(face_database_dir):
            # Detailed: Iterate over each file in the face database directory.
            # Simple: Go through every file in the face folder.
            if filename.endswith('.npz'):
                # Detailed: Check if the file is a .npz file (which contains saved face embeddings).
                # Simple: Only use files ending with .npz.
                person_name = os.path.splitext(filename)[0]
                # Detailed: Extract the person's name from the filename by removing the file extension.
                # Simple: Get the name by removing ".npz" from the filename.
                npz_path = os.path.join(face_database_dir, filename)
                # Detailed: Build the full path to the .npz file.
                # Simple: Create the complete file path.
                data = np.load(npz_path)
                # Detailed: Load the data from the .npz file.
                # Simple: Open the file to get its data.
                embeddings = data['embeddings']
                # Detailed: Extract the 'embeddings' array from the loaded data.
                # Simple: Get the face codes from the file.
                for embedding in embeddings:
                    # Detailed: Iterate over each embedding in the array.
                    # Simple: For every face code in the file:
                    self.saved_embeddings.append(torch.tensor(embedding).to(self.device))
                    # Detailed: Convert the embedding to a PyTorch tensor, move it to the designated device, and store it.
                    # Simple: Turn the face code into a tensor and save it.
                    self.saved_labels.append(person_name)
                    # Detailed: Save the corresponding person's name to the labels list.
                    # Simple: Save the name.

    def recognize_face(self, aligned_face):
        """Recognize a face from an aligned image."""
        # Detailed: Transform the aligned face image to a tensor, compute its embedding using the model, and compare it with saved embeddings.
        # Simple: Convert the fixed face image into a code and compare it with saved codes.
        face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
        # Detailed: Apply preprocessing to the aligned face image, add a batch dimension, and move the tensor to the computation device.
        # Simple: Prepare the face image so the model can process it.
        embedding = self.model(face_tensor).detach()
        # Detailed: Pass the processed tensor through the model to get its embedding, and detach it from the computation graph.
        # Simple: Get the face code from the model.

        similarities = [cosine_similarity(saved_emb, embedding).item() 
                       for saved_emb in self.saved_embeddings]
        # Detailed: Calculate the cosine similarity between the computed embedding and each saved embedding, yielding a list of similarity scores.
        # Simple: Compare the new face code with all saved ones to see how similar they are.
        best_idx = np.argmax(similarities)
        # Detailed: Identify the index of the highest similarity score.
        # Simple: Find which saved code is closest.
        best_score = similarities[best_idx]
        # Detailed: Retrieve the highest similarity score from the list.
        # Simple: Get the best matching score.
        name = self.saved_labels[best_idx] if best_score > 0.5 else "Unknown"
        # Detailed: Determine the recognized name based on a similarity threshold (0.5); if below threshold, label as "Unknown".
        # Simple: If the score is high enough, use the saved name; otherwise, mark as unknown.
        return name, best_score

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        # Detailed: Process a video frame by detecting faces, aligning them, recognizing each face, and annotating the frame.
        # Simple: Look at a video frame, find faces, get their names, and draw boxes and labels.
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            # Detailed: Check if the frame is a valid NumPy array and not empty; if invalid, return the original frame.
            # Simple: If the frame is bad or empty, just return it.
            return frame

        # Calculate FPS
        self.curr_frame_time = time.time()
        # Detailed: Record the current time to compute the frame rate.
        # Simple: Get the current time.
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        # Detailed: Calculate the instantaneous FPS by taking the inverse of the time difference between the current and previous frames.
        # Simple: Compute how many frames per second by using the time difference.
        self.fps = 0.9 * self.fps + 0.1 * fps  # Smooth FPS using exponential moving average
        # Detailed: Update the FPS value using an exponential moving average to smooth out sudden changes.
        # Simple: Smooth out the FPS measurement.
        self.prev_frame_time = self.curr_frame_time
        # Detailed: Update the previous frame time to the current time for the next iteration.
        # Simple: Save the current time for the next FPS calculation.

        # Detect faces
        boxes, _ = detect_faces(frame)
        # Detailed: Use the detect_faces function to locate faces in the frame, returning bounding boxes and other data.
        # Simple: Find faces in the frame.
        if len(boxes) == 0:
            # Detailed: If no faces are detected, overlay the FPS on the frame and return it.
            # Simple: If there are no faces, just show the FPS and return the frame.
            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame

        # Process each detected face
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Detailed: Convert the bounding box coordinates to integers.
            # Simple: Get the face box coordinates as whole numbers.
            face_crop = frame[y1:y2, x1:x2]
            # Detailed: Crop the face region from the frame based on the bounding box.
            # Simple: Cut out the face from the frame.
            if face_crop.size == 0:
                # Detailed: If the face crop is empty, skip processing this box.
                # Simple: If no face image was captured, skip it.
                continue

            # Align and recognize face
            aligned_face = align_face(face_crop)
            # Detailed: Align the cropped face image to standardize its orientation and size.
            # Simple: Fix the face image so it's easier to compare.
            if aligned_face is not None:
                name, similarity = self.recognize_face(aligned_face)
                # Detailed: Recognize the face by computing its embedding and comparing it to saved embeddings.
                # Simple: Get the name and score of the face.

                # Draw results
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                # Detailed: Choose green color if the face is recognized, otherwise red for unknown.
                # Simple: Use green for known faces and red for unknown.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Detailed: Draw a rectangle around the detected face with the chosen color.
                # Simple: Draw a box around the face.
                cv2.putText(frame, f"{name}, score: {similarity:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Detailed: Overlay the recognized name and similarity score above the bounding box.
                # Simple: Write the face's name and score near the box.

        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Detailed: Draw the current FPS value on the frame for performance monitoring.
        # Simple: Show the frame rate on the screen.

        return frame
        # Detailed: Return the processed and annotated frame.
        # Simple: Give back the updated frame.

    def cleanup(self):
        """Release resources."""
        # Detailed: Release any resources held by OpenCV by destroying all created windows.
        # Simple: Close all video windows.
        cv2.destroyAllWindows()

def main():
    pipeline = FaceRecognitionPipeline()
    # Detailed: Create an instance of the FaceRecognitionPipeline, which loads the face model and database.
    # Simple: Start the face recognition system.
    cap = cv2.VideoCapture(0)
    # Detailed: Open a video capture stream using the default camera (index 0).
    # Simple: Turn on the webcam.

    try:
        while cap.isOpened():
            # Detailed: Continuously capture frames from the webcam as long as it is open.
            # Simple: Keep getting frames from the camera.
            ret, frame = cap.read()
            # Detailed: Read a frame from the video capture; ret is True if successful.
            if not ret:
                # Detailed: If the frame was not successfully captured, break the loop.
                # Simple: Stop if no frame is received.
                break

            processed = pipeline.process_frame(frame)
            # Detailed: Process the frame through the pipeline to detect and recognize faces, and annotate the frame.
            # Simple: Run the frame through the face recognition system.
            if processed is not None:
                cv2.imshow('Face Recognition', processed)
                # Detailed: Display the processed frame in a window titled 'Face Recognition'.
                # Simple: Show the updated frame on the screen.

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Detailed: Wait for 1 millisecond for a key press, and if 'q' is pressed, exit the loop.
                # Simple: If you press 'q', stop the program.
                break
    finally:
        pipeline.cleanup()
        # Detailed: Clean up resources by closing OpenCV windows.
        # Simple: Close the video windows.
        cap.release()
        # Detailed: Release the video capture device.
        # Simple: Turn off the webcam.

if __name__ == "__main__":
    main()
    # Detailed: Execute the main function if this script is run as the main program.
    # Simple: Start the program when this file is run.
