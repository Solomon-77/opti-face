import os
import torch
import numpy as np
from torchvision import transforms
from face_alignment import align
from utils.face_recognition import get_model
import torch.nn.functional as F

# Load model
model = get_model("edgeface_xs_gamma_06")
model.load_state_dict(torch.load('./models/face_recognition/edgeface_xs_gamma_06.pt', map_location='cpu'))
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Preprocess and align the image
def preprocess_image(image_path):
    aligned_face, _ = align.get_aligned_face(image_path)
    if aligned_face:
        return transform(aligned_face).unsqueeze(0)
    return None

# Directory containing face dataset
face_dataset_dir = './face_database/'

# Create embeddings and labels
def create_face_embeddings():
    embeddings, labels = [], []
    
    for person_name in os.listdir(face_dataset_dir):
        person_folder = os.path.join(face_dataset_dir, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                face_tensor = preprocess_image(image_path)
                
                if face_tensor is not None:
                    embedding = model(face_tensor).detach().cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(person_name)

    np.save('face_embeddings.npy', {'embeddings': np.array(embeddings), 'labels': labels})

# Call to create face embeddings
create_face_embeddings()

# Load saved embeddings
saved_data = np.load('face_embeddings.npy', allow_pickle=True).item()
saved_embeddings, saved_labels = saved_data['embeddings'], saved_data['labels']

# Recognize face by comparing embeddings
def recognize_face(image_path):
    face_tensor = preprocess_image(image_path)
    if face_tensor is None:
        print("No face detected in the image.")
        return "Unknown", None
    
    compare_embedding = model(face_tensor).detach().cpu().numpy()
    
    highest_similarity, recognized_person = -1, "Unknown"
    
    for saved_embedding, label in zip(saved_embeddings, saved_labels):
        similarity = F.cosine_similarity(torch.tensor(saved_embedding), torch.tensor(compare_embedding)).item()
        if similarity > highest_similarity:
            highest_similarity, recognized_person = similarity, label

    similarity_threshold = 0.40
    if highest_similarity > similarity_threshold:
        print(f'Match found: {recognized_person}, Similarity: {highest_similarity:.2f}')
        return recognized_person, highest_similarity
    else:
        print('No match. Label: Unknown')
        return "Unknown", None
