import cv2  # Detailed: Import OpenCV for image processing operations such as reading, resizing, and warping images.
         # Simple: Lets us work with images.
import numpy as np  # Detailed: Import NumPy for numerical operations and handling arrays.
                   # Simple: Helps with math and arrays.
import mediapipe as mp  # Detailed: Import MediaPipe, which is used here for detecting and extracting facial landmarks.
                       # Simple: Helps find face features.
from PIL import Image  # Detailed: Import the Python Imaging Library (PIL) to handle image operations like format conversion.
                       # Simple: Lets us work with images in a different way.
import torch  # Detailed: Import PyTorch for deep learning functionalities, tensor operations, and model handling.
            # Simple: Used for machine learning.
import os  # Detailed: Import the os module to interact with the file system.
         # Simple: Lets us work with files and directories.
import warnings  # Detailed: Import warnings to control warning messages during execution.
warnings.filterwarnings("ignore")  # Detailed: Suppress warning messages for a cleaner output.
                                   # Simple: Hide warnings.
from torchvision import transforms  # Detailed: Import transforms from torchvision to perform image transformations (e.g., normalization).
                                    # Simple: Helps change images before processing.
from utils.scrfd import FaceDetector  # Detailed: Import the FaceDetector class from the SCRFD module for face detection.
                                       # Simple: Get the face detector.
from utils.edgeface import get_model  # Detailed: Import the get_model function from edgeface module to load the face recognition model.
                                       # Simple: Get the function to load the face model.

#############################################
# Initialize Face Detector and MediaPipe FaceMesh
#############################################

face_detector = FaceDetector(onnx_file='checkpoints/scrfd_500m.onnx')
# Detailed: Create an instance of FaceDetector using an ONNX model file for SCRFD with 500 million parameters.
# Simple: Set up the face detector with a pre-trained model.

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,  # Detailed: Process images as static images rather than as a video stream.
                             # Simple: Treat each picture individually.
    max_num_faces=1,         # Detailed: Limit detection to at most one face per image.
                             # Simple: Only look for one face.
    min_detection_confidence=0.5,  # Detailed: Set the minimum confidence threshold for face detection.
                                  # Simple: Only detect if at least 50% sure.
    min_tracking_confidence=0.5,   # Detailed: Set the minimum confidence for tracking face landmarks (if applicable).
                                  # Simple: Only track if at least 50% sure.
    refine_landmarks=True          # Detailed: Enable refinement of landmarks for higher accuracy.
                                  # Simple: Get more precise face points.
)

#############################################
# Define Transformation Pipeline for Face Preprocessing
#############################################

transform = transforms.Compose([
    transforms.ToTensor(),  # Detailed: Convert the PIL image to a PyTorch tensor.
                            # Simple: Turn the image into numbers.
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # Detailed: Normalize the tensor image with mean 0.5 and standard deviation 0.5 for each channel.
                                               # Simple: Scale the numbers for the model.
])

#############################################
# Template Landmarks for Face Alignment (Normalized Coordinates)
#############################################

TEMPLATE_LANDMARKS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.6963],  # Detailed: Coordinates for the centers of the eyes.
                                             # Simple: Points for the eyes.
    [56.0252, 71.7366],                      # Detailed: Coordinate for the tip of the nose.
                                             # Simple: Point for the nose.
    [41.5493, 92.3655], [70.7299, 92.3655]     # Detailed: Coordinates for the corners of the mouth.
                                             # Simple: Points for the mouth.
]) / 112.0  # Detailed: Normalize the coordinates by dividing by 112.0.
            # Simple: Scale the points to a [0,1] range.

#############################################
# Landmark Groups for Different Facial Features
#############################################

LANDMARK_GROUPS = [
    [([33, 133, 246], [130, 243, 112, 156, 157, 158, 153, 154, 145, 144, 163])],
    [([362, 263, 466], [359, 466, 341, 384, 385, 386, 380, 381, 374, 373, 390])],
    [([1, 4], [2, 98, 327, 168, 197, 195, 5, 4, 45, 51])],
    [([61, 91, 84], [62, 87, 146, 177, 178, 88, 95, 78, 191, 80, 81])],
    [([291, 321, 314], [292, 317, 375, 407, 408, 318, 325, 308, 409, 310, 311])]
]
# Detailed: Define groups of landmark indices for different parts of the face (e.g., eyes, nose, mouth).
# Simple: Group face points for eyes, nose, and mouth.

#############################################
# Face Recognition Model Loader
#############################################

def load_face_recognition_model(model_path="checkpoints/edgeface_s_gamma_05.pt"):
    # Detailed: Load the face recognition model from a checkpoint file. The model name is inferred from the file name.
    # Simple: Load the face model from a file.
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Detailed: Choose the computation device (GPU if available, otherwise CPU).
    # Simple: Use GPU if available, else CPU.
    
    model = get_model(model_name)
    # Detailed: Retrieve the model architecture using the model name.
    # Simple: Build the face model.
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Detailed: Load the model weights from the checkpoint file and map them to the selected device.
    # Simple: Load the saved weights into the model.
    model.to(device)
    # Detailed: Transfer the model to the chosen computation device.
    # Simple: Put the model on the GPU or CPU.
    model.eval()
    # Detailed: Set the model to evaluation mode to disable training-specific operations.
    # Simple: Make the model ready for testing.
    
    return model, device
    # Detailed: Return the loaded model and device.
    # Simple: Give back the model and device.

#############################################
# Face Landmark Extraction using MediaPipe FaceMesh
#############################################

def get_landmarks(image):
    """Extract face landmarks from an image using MediaPipe FaceMesh."""
    # Detailed: Convert the image from BGR to RGB and process it with MediaPipe to extract facial landmarks.
    # Simple: Find face points in the image.
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        # Detailed: If no landmarks are detected, return None.
        # Simple: If no face points are found, return nothing.
        return None

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    points = []

    # Loop over each landmark group defined in LANDMARK_GROUPS
    for main_idx, surrounding_idx in [group[0] for group in LANDMARK_GROUPS]:
        # Detailed: Compute the mean position for the main landmark indices.
        # Simple: Average the main face points.
        main_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                               for idx in (main_idx if isinstance(main_idx, (tuple, list)) else [main_idx])], axis=0)
        
        if surrounding_idx:
            # Detailed: Compute the mean position for the surrounding landmarks and blend them with the main points.
            # Simple: Average the nearby face points and mix with the main ones.
            surrounding_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                                          for idx in surrounding_idx], axis=0)
            main_points = 0.8 * main_points + 0.2 * surrounding_points
        
        points.append(main_points)
    # Detailed: Return the landmarks as a NumPy array with type float32.
    # Simple: Return the face points.
    return np.array(points, dtype=np.float32)

#############################################
# Transformation Matrix Computation for Face Alignment
#############################################

def get_transform(src, dst):
    """Calculate the transformation matrix to align `src` landmarks to `dst` landmarks."""
    # Detailed: Compute the transformation matrix that aligns source landmarks to destination landmarks using centroids, scaling, and rotation.
    # Simple: Calculate how to move and scale the face points to match a standard template.
    if src.shape != dst.shape:
        return None

    # Calculate centroids for both source and destination landmarks.
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center the landmarks by subtracting the centroids.
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Calculate the scale factor based on the average distance of the landmarks from the centroid.
    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
    # Compute the rotation matrix using a least-squares approach.
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)

    # Perform Singular Value Decomposition (SVD) on H.
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    
    # Ensure a proper rotation (determinant should be positive).
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # Construct the full 3x3 transformation matrix.
    T = np.eye(3)
    T[:2, :2] = scale * R  # Apply scaling and rotation.
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)  # Compute translation.
    return T

#############################################
# Face Detection using SCRFD
#############################################

def detect_faces(image):
    """Detect faces in an image using SCRFD."""
    # Detailed: Use the face_detector instance to detect faces in the input image, using a fixed input size.
    # Simple: Find faces in the image.
    det, _ = face_detector.detect(image, input_size=(640, 640))
    if det is None or len(det) == 0:
        return np.array([]), np.array([])

    # Extract bounding boxes and confidence scores from detections.
    boxes = det[:, :4]
    scores = det[:, 4]
    return boxes, scores

#############################################
# Face Alignment using Landmarks
#############################################

def align_face(face_img, size=(112, 112)):
    """Align a face image to a canonical pose using landmarks."""
    # Detailed: Obtain facial landmarks from the face image and compute the transformation matrix to align the face.
    # Simple: Adjust the face image so that the features are in the right place.
    landmarks = get_landmarks(face_img)
    if landmarks is None:
        return None

    # Calculate the transformation matrix to align the detected landmarks to the template landmarks.
    tform = get_transform(landmarks, TEMPLATE_LANDMARKS * size)
    if tform is None:
        return None

    # Warp the face image using the computed transformation matrix.
    warped = cv2.warpPerspective(face_img, tform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Convert the aligned face from BGR to RGB and return as a PIL Image.
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

#############################################
# Preprocess an Image for Face Recognition
#############################################

def preprocess_image(image_path):
    """Preprocess an image for face recognition by detecting and aligning the faces."""
    # Detailed: Read an image from the provided path, detect faces in it, align each face, and transform it into a tensor.
    # Simple: Open the image, find the face, fix its orientation, and prepare it for the model.
    image = cv2.imread(image_path)
    if image is None:
        return None

    boxes, _ = detect_faces(image)
    if not len(boxes):
        return None

    # Process each detected face and return the first one that can be aligned.
    for box in boxes.astype(int):
        face_img = image[box[1]:box[3], box[0]:box[2]]
        aligned_face = align_face(face_img)
        if aligned_face:
            # Apply the transformation pipeline to convert the image into a tensor and add a batch dimension.
            return transform(aligned_face).unsqueeze(0)
    
    return None

#############################################
# Cleanup Function for MediaPipe Resources
#############################################

def cleanup():
    """Release resources used by MediaPipe FaceMesh."""
    # Detailed: Close the MediaPipe FaceMesh instance to free up resources.
    # Simple: Shut down the face landmark detector.
    mp_face_mesh.close()
