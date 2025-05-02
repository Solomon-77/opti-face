import os
import numpy as np
import torch
# Import helper functions (adjust path if needed)
import sys
project_root_prep = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_prep not in sys.path:
    sys.path.insert(0, project_root_prep)
from app import resource_path, get_writable_path # Import from app.py

from src.backend.utils.face_utils import preprocess_image, load_face_recognition_model

# Default paths using helper functions
DEFAULT_FACE_DATABASE_DIR = get_writable_path("face_database")
DEFAULT_MODEL_PATH = resource_path("src/backend/checkpoints/edgeface_s_gamma_05.pt")

def generate_and_save_embeddings(person_name, person_folder, output_dir, model, device):
    """
    Generates embeddings for all images in a person's folder and saves them to a .npz file.

    Args:
        person_name (str): The name of the person.
        person_folder (str): The path to the directory containing the person's images.
        output_dir (str): The directory where the .npz file should be saved.
        model: The loaded face recognition model.
        device: The device (CPU or CUDA) to run the model on.
    """
    if not os.path.isdir(person_folder):
        print(f"Error: Folder not found for {person_name} at {person_folder}")
        return

    person_embeddings = []
    print(f"Processing images for {person_name} in {person_folder}...")

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        # Basic check for image file extensions
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            face_tensor = preprocess_image(image_path)
            if face_tensor is not None:
                try:
                    with torch.no_grad(): # Ensure no gradients are calculated
                        embedding = model(face_tensor.to(device)).detach().cpu().numpy()
                    person_embeddings.append(embedding.squeeze()) # Squeeze to remove batch dim if present
                except Exception as e:
                    print(f"Error processing image {image_name} for {person_name}: {e}")
            else:
                print(f"Could not preprocess or detect face in {image_name} for {person_name}.")
        else:
            print(f"Skipping non-image file: {image_name}")


    if person_embeddings:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save individual .npz file for the person in the output directory
        npz_path = os.path.join(output_dir, f"{person_name}.npz")
        try:
            np.savez(npz_path, embeddings=np.array(person_embeddings))
            print(f"Saved embeddings for {person_name} to {npz_path}")
            return npz_path # Return the path to the saved file
        except Exception as e:
            print(f"Error saving embeddings for {person_name}: {e}")
    else:
        print(f"No embeddings generated for {person_name}.")
    return None


def process_all_persons(face_database_dir=None, model_path=None):
    """Processes all person folders in the face database directory."""
    # Use defaults if None, otherwise use provided paths
    db_dir = face_database_dir if face_database_dir is not None else DEFAULT_FACE_DATABASE_DIR
    m_path = model_path if model_path is not None else DEFAULT_MODEL_PATH

    print(f"Processing database: {db_dir}")
    print(f"Using model: {m_path}")

    model, device = load_face_recognition_model(model_path=m_path) # Pass resolved model path

    # Process each person's directory found directly under face_database_dir
    if not os.path.isdir(db_dir):
        print(f"Error: Database directory not found: {db_dir}")
        return

    for item_name in os.listdir(db_dir):
        item_path = os.path.join(db_dir, item_name)
        # Check if it's a directory AND not a .npz file (to avoid processing existing embeddings)
        if os.path.isdir(item_path):
            # Pass the resolved db_dir as the output directory
            generate_and_save_embeddings(item_name, item_path, db_dir, model, device)

# Allow running the script directly to process all persons
if __name__ == "__main__":
    print("Starting batch processing of all persons in the database...")
    # Uses resolved default paths defined above
    process_all_persons()
    print("Batch processing finished.")