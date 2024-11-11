import os
import numpy as np
from utils.face_utils import preprocess_image, load_face_recognition_model

def create_face_embeddings(face_database_dir='./face_database/'):
    model, device = load_face_recognition_model()
    embeddings, labels = [], []
    
    for person_name in os.listdir(face_database_dir):
        person_folder = os.path.join(face_database_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
            
        print(f"Processing {person_name}...")
        for image_name in os.listdir(person_folder):
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            if face_tensor is not None:
                embedding = model(face_tensor.to(device)).detach().cpu().numpy()
                embeddings.append(embedding)
                labels.append(person_name)
    
    np.save('face_embeddings.npy', {'embeddings': np.array(embeddings), 'labels': labels})
    print("Face embeddings and labels saved to face_embeddings.npy")
    
create_face_embeddings()