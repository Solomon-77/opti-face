import os  # Detailed: Import the built-in OS module for interacting with the operating system, such as handling file paths, directories, and files.
         # Simple: Lets the program work with files and folders.

import numpy as np  # Detailed: Import NumPy and alias it as 'np' for efficient numerical operations and array handling.
                   # Simple: NumPy helps with math and storing numbers in arrays.

from utils.face_utils import preprocess_image, load_face_recognition_model  
# Detailed: Import the 'preprocess_image' and 'load_face_recognition_model' functions from a module in the 'utils' directory, which are used to prepare images and load a face recognition model.
# Simple: Get the functions that help read images and load the face recognition model.

def create_face_embeddings(face_database_dir='./face_database/'):
    # Detailed: Define a function 'create_face_embeddings' with an optional parameter 'face_database_dir', which specifies the directory containing face images organized by person.
    # Simple: This function makes face codes (embeddings) for images in folders. The folder path is './face_database/' by default.
    
    """
    Detailed:
    This function computes and saves face embeddings for each person stored in a face database directory.
    Each subdirectory in 'face_database_dir' should correspond to a person and contain that person's face images.
    
    Simple:
    It reads face pictures from folders, turns them into number codes (embeddings) using a model, and saves them in a file.
    """
    
    model, device = load_face_recognition_model()
    # Detailed: Load a pre-trained face recognition model along with the computation device (e.g., CPU or GPU) to process the images.
    # Simple: Load the face recognition model and determine whether to use the computer's processor or graphics card.

    for person_name in os.listdir(face_database_dir):
        # Detailed: Loop through every item in the 'face_database_dir'; each item is expected to be a directory corresponding to a person.
        # Simple: For each folder (person) in the database folder, do the following.
        
        person_folder = os.path.join(face_database_dir, person_name)
        # Detailed: Construct the full path to the person's folder by joining the base directory with the person's name.
        # Simple: Create the path to the person's folder.
        
        if not os.path.isdir(person_folder):
            # Detailed: Check if the path corresponds to a directory. If not (e.g., if it's a file), skip this iteration.
            # Simple: If it's not a folder, skip it.
            continue
            
        person_embeddings = []
        # Detailed: Initialize an empty list to hold the embeddings for the current person.
        # Simple: Start an empty list to save the face codes for this person.
        
        print(f"Processing {person_name}...")
        # Detailed: Print a message indicating that the images for the current person are being processed.
        # Simple: Show a message that this person's images are now being processed.
        
        for image_name in os.listdir(person_folder):
            # Detailed: Loop through every file in the current person's folder; each file is expected to be an image.
            # Simple: For each image file in the person's folder, do the following.
            
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            # Detailed: Create the full path to the image file and pass it to 'preprocess_image', which reads the image, detects the face, and converts it to a tensor suitable for model input.
            # Simple: Read and prepare the image so it can be processed by the model.
            
            if face_tensor is not None:
                # Detailed: Check if the preprocessing was successful (i.e., a face was detected and converted into a tensor). If it failed, skip this image.
                # Simple: Only continue if the image was good and a face was found.
                
                embedding = model(face_tensor.to(device)).detach().cpu().numpy()
                # Detailed: Move the preprocessed face tensor to the specified device (CPU or GPU), pass it through the model to compute the embedding,
                # then detach the resulting tensor from the computation graph, transfer it to the CPU, and convert it into a NumPy array.
                # Simple: Run the prepared image through the model to get a number code (embedding), and then convert it into a normal array.
                
                person_embeddings.append(embedding)
                # Detailed: Append the computed embedding to the list for this person.
                # Simple: Save this face code to the person's list.
        
        if person_embeddings:
            # Detailed: Check if any embeddings were successfully computed for the current person.
            # Simple: If there are any face codes for this person, then continue.
            
            npz_path = os.path.join(face_database_dir, f"{person_name}.npz")
            # Detailed: Create the file path for saving the embeddings by joining the base directory with a filename based on the person's name and a .npz extension.
            # Simple: Create a file path to save the face codes, naming the file after the person.
            
            np.savez(npz_path, embeddings=np.array(person_embeddings))
            # Detailed: Convert the list of embeddings into a NumPy array and save it as a compressed .npz file at the specified path using np.savez.
            # Simple: Save the list of face codes as a file that can be loaded later.
            
            print(f"Saved embeddings for {person_name} to {npz_path}")
            # Detailed: Print a confirmation message indicating that the embeddings for the current person have been successfully saved to the specified path.
            # Simple: Show a message confirming that the file was saved for this person.

create_face_embeddings()
# Detailed: Invoke the 'create_face_embeddings' function to begin processing the face database and generating embeddings for each person.
# Simple: Run the function to start making face codes for all the people.
