import os
import numpy as np
import cv2
import uuid
from datetime import datetime
from utils.face_utils import preprocess_image, load_face_recognition_model
from utils.path_patch_v1 import get_resource_path

class TrainingPipeline:
    def __init__(self):
        self.model, self.device = load_face_recognition_model()
        self.face_database_dir = get_resource_path('face_database')
        self.records_dir = os.path.join(os.getcwd(), 'face_database', 'records')  # Path to src > face_database > records
        os.makedirs(self.records_dir, exist_ok=True)  # Ensure the records directory exists

    def generate_uid(self):
        """Generate a unique identifier using timestamp and a random UUID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]  # Take first 8 characters of UUID
        return f"{timestamp}_{unique_id}"

    def process_video(self, person_name, video_path, progress_callback=None):
        """Process the video to extract frames, create embeddings, and save with UID."""
        person_folder = os.path.join(self.face_database_dir, person_name)
        os.makedirs(person_folder, exist_ok=True)
        
        # Generate a unique identifier for the person
        uid = self.generate_uid()
        
        # Create a new folder in src > face_database > records named after the UID
        uid_folder = os.path.join(self.records_dir, uid)
        os.makedirs(uid_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(person_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            if progress_callback:
                progress_callback.update_frame_progress(frame_count)
        cap.release()

        person_embeddings = []
        embedding_count = 0
        for image_name in os.listdir(person_folder):
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            if face_tensor is not None:
                embedding = self.model(face_tensor.to(self.device)).detach().cpu().numpy()
                person_embeddings.append(embedding)
                embedding_count += 1
                if progress_callback:
                    progress_callback.update_embedding_progress(embedding_count)
            os.remove(os.path.join(person_folder, image_name))
        os.rmdir(person_folder)

        if person_embeddings:
            # Save embeddings along with metadata
            npz_path = os.path.join(self.face_database_dir, f"{person_name}_{uid}.npz")
            np.savez(npz_path, 
                     embeddings=np.array(person_embeddings),
                     person_name=person_name,
                     uid=uid,
                     creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     records_folder_path=uid_folder)  # Save the path to the UID folder
            return npz_path
        return None
