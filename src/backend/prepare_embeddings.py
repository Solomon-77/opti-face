import os
import numpy as np
import torch # Added for type hinting if needed, ensure installed
from src.backend.utils.face_utils import preprocess_image, load_face_recognition_model # Corrected import path assuming prepare_embeddings is run from root or src

# Default paths (can be overridden)
DEFAULT_FACE_DATABASE_DIR = './src/backend/face_database/'
DEFAULT_MODEL_PATH = "src/backend/checkpoints/edgeface_s_gamma_05.pt"

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


def process_all_persons(face_database_dir=DEFAULT_FACE_DATABASE_DIR, model_path=DEFAULT_MODEL_PATH):
    """Processes all person folders in the face database directory."""
    model, device = load_face_recognition_model(model_path=model_path)

    # Process each person's directory found directly under face_database_dir
    for person_name in os.listdir(face_database_dir):
        person_folder = os.path.join(face_database_dir, person_name)
        # Check if it's a directory AND not a .npz file (to avoid processing existing embeddings)
        if os.path.isdir(person_folder):
            generate_and_save_embeddings(person_name, person_folder, face_database_dir, model, device)

# Allow running the script directly to process all persons
if __name__ == "__main__":
    print("Starting batch processing of all persons in the database...")
    # Use default directory, assumes script is run from a location where the path is valid
    process_all_persons()
    print("Batch processing finished.")

# Remove the standalone call create_face_embeddings() if it exists outside __main__
# create_face_embeddings() # Remove this line if present