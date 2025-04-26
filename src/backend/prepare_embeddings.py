import os
import cv2
import numpy as np
import shutil
from src.backend.utils.face_utils import preprocess_image, load_face_recognition_model

def create_face_embeddings(face_database_dir='./face_database/'):#original function
    model, device = load_face_recognition_model()
    
    # Process each person's directory
    for person_name in os.listdir(face_database_dir):
        person_folder = os.path.join(face_database_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
            
        person_embeddings = []
        print(f"Processing {person_name}...")
        
        for image_name in os.listdir(person_folder):
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            if face_tensor is not None:
                embedding = model(face_tensor.to(device)).detach().cpu().numpy()
                person_embeddings.append(embedding)
        
        if person_embeddings:
            # Save individual .npz file for each person
            npz_path = os.path.join(face_database_dir, f"{person_name}.npz")
            np.savez(npz_path, embeddings=np.array(person_embeddings))
            print(f"Saved embeddings for {person_name} to {npz_path}")

def create_face_embeddings_targeted(frames_folder, person_name):
    model, device = load_face_recognition_model()
    
    # Process frames from the provided folder
    person_embeddings = []
    print(f"Processing frames from {frames_folder}...")

    for image_name in os.listdir(frames_folder):
        image_path = os.path.join(frames_folder, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Ignore non-image files

        face_tensor = preprocess_image(image_path)
        if face_tensor is not None:
            embedding = model(face_tensor.to(device)).detach().cpu().numpy()
            person_embeddings.append(embedding)
    
    if person_embeddings:
        # Save the embeddings as a .npz file using the provided name
        npz_path = os.path.join("./src/backend/face_database", f"{person_name}.npz")
        np.savez(npz_path, embeddings=np.array(person_embeddings))
        print(f"Saved embeddings to {npz_path}")

        # Delete the folder with extracted frames after saving
        print(f"Deleting temporary frames folder {frames_folder}...")
        shutil.rmtree(frames_folder)
    else:
        print(f"No valid faces found in frames from {frames_folder}, skipping deletion.")


def video_to_frames(video_path, output_folder, fps=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)

    if frame_interval == 0:
        frame_interval = 1

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")
    return output_folder

def video_to_embeddings(video_path, output_folder, fps=1, person_name="Unknown"):
    frames_folder = video_to_frames(video_path, output_folder, fps)
    create_face_embeddings_targeted(frames_folder, person_name)