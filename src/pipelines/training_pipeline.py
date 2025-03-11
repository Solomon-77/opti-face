import os  # Detailed: Import the os module to interact with the operating system (e.g., file and folder operations).  
         # Simple: Lets us work with files and folders.
import numpy as np  # Detailed: Import NumPy for numerical operations and array handling.  
                   # Simple: Helps with math and arrays.
import cv2  # Detailed: Import OpenCV for image and video processing tasks such as reading and writing images.  
         # Simple: Lets us work with pictures and videos.
import uuid  # Detailed: Import uuid to generate unique identifiers.  
           # Simple: Helps make unique IDs.
from datetime import datetime  # Detailed: Import datetime to work with date and time, such as timestamps.  
                               # Simple: Lets us use the current time.
from utils.face_utils import preprocess_image, load_face_recognition_model  
# Detailed: Import custom functions for face preprocessing and loading a pre-trained face recognition model.  
# Simple: Get functions to prepare images and load the face model.
from utils.path_patch_v1 import get_resource_path  
# Detailed: Import a helper function to get the correct resource file path in both development and executable modes.  
# Simple: Helps find the right folder for files.

class TrainingPipeline:
    def __init__(self):
        # Detailed: Initialize the training pipeline by loading the face recognition model, setting up directories,
        # and ensuring that saved face records are stored properly.
        # Simple: Set up the system, load the model, and create folders for saving face data.
        self.model, self.device = load_face_recognition_model()  
        # Detailed: Load the pre-trained face recognition model and determine whether to use GPU or CPU.
        # Simple: Load the face model and choose GPU or CPU.
        self.face_database_dir = get_resource_path('face_database')  
        # Detailed: Get the absolute path to the 'face_database' folder using the helper function.
        # Simple: Set the folder where face data is saved.
        self.records_dir = os.path.join(os.getcwd(), 'face_database', 'records')  
        # Detailed: Define the records directory path by joining the current working directory with 'face_database/records'.
        # Simple: Set the folder for saving face images (records).
        os.makedirs(self.records_dir, exist_ok=True)  
        # Detailed: Create the records directory if it does not already exist.
        # Simple: Make sure the folder exists.

    def generate_uid(self):
        """Generate a unique identifier using timestamp and a random UUID."""
        # Detailed: This function generates a unique identifier by combining the current timestamp (formatted as YYYYMMDDHHMMSS)
        # with the first 8 characters of a randomly generated UUID.
        # Simple: Create a unique ID using the current time and a random string.
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  
        # Detailed: Get the current time and format it as a string "YYYYMMDDHHMMSS".
        # Simple: Get the current time in a simple number format.
        unique_id = str(uuid.uuid4().hex)[:8]  
        # Detailed: Generate a random UUID, convert it to a hexadecimal string, and take the first 8 characters.
        # Simple: Make a random 8-character string.
        return f"{timestamp}_{unique_id}"  
        # Detailed: Return the unique identifier by concatenating the timestamp and unique ID with an underscore.
        # Simple: Return a unique ID like "20230315123045_ab12cd34".

    def process_video(self, person_name, video_path, progress_callback=None):
        """Process the video to extract frames, create embeddings, and save with UID."""
        # Detailed: Process the provided video by extracting every frame, saving them, computing face embeddings,
        # and saving the resulting data with a unique identifier (UID) and metadata.
        # Simple: Read the video, save its frames, compute face codes, and store everything with a unique ID.
        person_folder = os.path.join(self.face_database_dir, person_name)
        # Detailed: Create a folder for the person by joining the face database directory with the person's name.
        # Simple: Make a folder for this person.
        os.makedirs(person_folder, exist_ok=True)
        # Detailed: Ensure that the person's folder exists by creating it if necessary.
        # Simple: Make sure the person's folder is there.
        
        uid = self.generate_uid()  
        # Detailed: Generate a unique identifier (UID) for the person using the generate_uid method.
        # Simple: Create a unique ID for this person.
        
        uid_folder = os.path.join(self.records_dir, uid)
        # Detailed: Create a folder path within the records directory using the UID.
        # Simple: Set the folder for this unique ID.
        os.makedirs(uid_folder, exist_ok=True)
        # Detailed: Ensure that the UID folder exists.
        # Simple: Make sure the UID folder is there.

        cap = cv2.VideoCapture(video_path)
        # Detailed: Open the video file located at video_path using OpenCV.
        # Simple: Start reading the video.
        frame_count = 0
        # Detailed: Initialize a frame counter to keep track of how many frames have been processed.
        # Simple: Start counting frames from zero.
        while True:
            ret, frame = cap.read()
            # Detailed: Read a frame from the video; ret indicates whether the read was successful.
            # Simple: Get a frame from the video.
            if not ret:
                break  # Detailed: If no frame is returned, exit the loop.
                       # Simple: Stop if there are no more frames.
            frame_path = os.path.join(person_folder, f"frame_{frame_count}.jpg")
            # Detailed: Construct a file path for the current frame image, naming it using the frame counter.
            # Simple: Create a file name for this frame.
            cv2.imwrite(frame_path, frame)
            # Detailed: Save the current frame as an image file to the person's folder.
            # Simple: Write the frame to disk.
            frame_count += 1
            # Detailed: Increment the frame counter.
            # Simple: Add one to the frame count.
            if progress_callback:
                progress_callback.update_frame_progress(frame_count)
                # Detailed: If a progress callback is provided, update the frame extraction progress.
                # Simple: Update the progress display with the new frame count.
        cap.release()
        # Detailed: Release the video capture object.
        # Simple: Close the video file.

        person_embeddings = []
        # Detailed: Initialize an empty list to store embeddings computed from the face images.
        # Simple: Create a list for face codes.
        embedding_count = 0
        # Detailed: Initialize a counter for the number of embeddings created.
        # Simple: Start counting face codes.
        for image_name in os.listdir(person_folder):
            # Detailed: Loop over all saved frame images in the person's folder.
            # Simple: For each frame file in the folder:
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            # Detailed: Preprocess the image (detect and align the face) to get a tensor suitable for the model.
            # Simple: Prepare the face image for the model.
            if face_tensor is not None:
                embedding = self.model(face_tensor.to(self.device)).detach().cpu().numpy()
                # Detailed: Pass the face tensor through the model to compute its embedding, detach it from the graph, move it to CPU, and convert it to a NumPy array.
                # Simple: Get the face code (a list of numbers) from the model.
                person_embeddings.append(embedding)
                # Detailed: Append the computed embedding to the list of embeddings.
                # Simple: Save the face code.
                embedding_count += 1
                # Detailed: Increment the embedding counter.
                # Simple: Count this face code.
                if progress_callback:
                    progress_callback.update_embedding_progress(embedding_count)
                    # Detailed: If a progress callback is provided, update the embedding extraction progress.
                    # Simple: Update the progress display with the new face code count.
            os.remove(os.path.join(person_folder, image_name))
            # Detailed: Remove the frame image after processing it to free up space.
            # Simple: Delete the frame file.
        os.rmdir(person_folder)
        # Detailed: Remove the person's folder once all frames have been processed and deleted.
        # Simple: Delete the now-empty folder.

        if person_embeddings:
            # Detailed: If any embeddings were successfully computed, save them along with metadata.
            # Simple: If face codes exist, save them.
            npz_path = os.path.join(self.face_database_dir, f"{person_name}_{uid}.npz")
            # Detailed: Construct the file path for the .npz file by combining the person's name and UID.
            # Simple: Create a file name using the person's name and unique ID.
            np.savez(npz_path, 
                     embeddings=np.array(person_embeddings),  # Detailed: Save the embeddings as a NumPy array.
                     person_name=person_name,  # Detailed: Save the person's name.
                     uid=uid,  # Detailed: Save the unique identifier.
                     creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Detailed: Save the creation date and time.
                     records_folder_path=uid_folder)  # Detailed: Save the path to the UID folder containing records.
            # Simple: Store the face codes, name, unique ID, current date, and the folder path in a .npz file.
            return npz_path
        return None  # Detailed: If no embeddings were computed, return None.
                      # Simple: If nothing was saved, return nothing.
