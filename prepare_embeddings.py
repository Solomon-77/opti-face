import os
import numpy as np
from utils.face_utils import preprocess_image, load_face_recognition_model

def create_face_embeddings(face_database_dir='./face_database/'):
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

create_face_embeddings()