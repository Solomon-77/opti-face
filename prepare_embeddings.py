import os
import numpy as np
from utils.face_utils import preprocess_image
from utils.model_utils import load_face_recognition_model

# Load the face recognition model
model = load_face_recognition_model()

# Directory containing face dataset
face_database_dir = './face_database/'

# Function to create face embeddings from face_database
def create_face_embeddings():
    embeddings = []
    labels = []
    
    # Traverse the face_database folder
    for person_name in os.listdir(face_database_dir):
        person_folder = os.path.join(face_database_dir, person_name)
        
        if os.path.isdir(person_folder):
            print(f"Processing images for {person_name}...")
            
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                
                # Preprocess the image to get the face tensor
                face_tensor = preprocess_image(image_path)
                
                if face_tensor is not None:
                    # Create the face embedding
                    embedding = model(face_tensor).detach().cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(person_name)
                else:
                    print(f"Face not detected in {image_name}")

    np.save('face_embeddings.npy', {'embeddings': np.array(embeddings), 'labels': labels})
    print("Face embeddings and labels saved to face_embeddings.npy")

create_face_embeddings()