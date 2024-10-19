import os
import torch
import numpy as np
from torchvision import transforms
from face_alignment import align
from utils.face_recognition import get_model
from utils.face_detection import FaceDetector  # Import the same detector
import cv2
from PIL import Image

# Load the face detection model
face_detector = FaceDetector('./models/face_detection/RFB-320.mnn')

# Load the face recognition model
model = get_model("edgeface_xs_gamma_06")
model.load_state_dict(torch.load('./models/face_recognition/edgeface_xs_gamma_06.pt', map_location='cpu'))
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Preprocess and align the image (use face detection from inference.py)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    boxes, _, _ = face_detector.detect(image)
    faces = []
    
    for x1, y1, x2, y2 in boxes.astype(int):
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2)
        face_img = image[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Use the same alignment method (if needed) or directly use the face region
        aligned_face, _ = align.get_aligned_face(rgb_pil_image=pil_img)
        
        if aligned_face:
            return transform(aligned_face).unsqueeze(0)
        else:
            return None

# Directory containing face dataset
face_database_dir = './face_database/'

# Function to create face embeddings from face_database
def create_face_embeddings():
    embeddings = []
    labels = []
    
    # Traverse the face_database folder
    for person_name in os.listdir(face_database_dir):
        person_folder = os.path.join(face_database_dir, person_name)
        
        # Check if it's a directory (representing a person)
        if os.path.isdir(person_folder):
            print(f"Processing images for {person_name}...")
            
            # Loop over each image file in the person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                
                # Preprocess the image to get the face tensor
                face_tensor = preprocess_image(image_path)
                
                if face_tensor is not None:
                    # Create the face embedding using the model
                    embedding = model(face_tensor).detach().cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(person_name)
                else:
                    print(f"Face not detected in {image_name}")

    np.save('face_embeddings.npy', {'embeddings': np.array(embeddings), 'labels': labels})
    print("Face embeddings and labels saved to face_embeddings.npy")

create_face_embeddings()