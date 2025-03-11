import cv2  # Detailed: Import OpenCV for various image processing functions like reading, resizing, and warping images.
         # Simple: Lets us work with pictures.
import numpy as np  # Detailed: Import NumPy for numerical computations and array handling.
                   # Simple: Helps with math and working with lists of numbers.
import mediapipe as mp  # Detailed: Import MediaPipe, a framework for building perception pipelines, used here for extracting facial landmarks.
                       # Simple: Helps find face features like eyes and nose.
from PIL import Image  # Detailed: Import the Python Imaging Library (PIL) to handle image operations such as conversion between formats.
                       # Simple: Lets us work with images in different formats.
import torch  # Detailed: Import PyTorch for tensor operations and handling deep learning models.
            # Simple: Used for machine learning tasks.
import os  # Detailed: Import the os module to interact with the operating system (e.g., file paths and directories).
         # Simple: Lets us work with files and folders.
import warnings  # Detailed: Import the warnings module to manage warning messages.
warnings.filterwarnings("ignore")  # Detailed: Suppress warning messages to keep the output clean.
                                   # Simple: Hide warnings.
from torchvision import transforms  # Detailed: Import image transformation utilities from torchvision for preprocessing images.
                                    # Simple: Helps prepare images for the model.
from utils.scrfd import FaceDetector  # Detailed: Import the FaceDetector class from the SCRFD module, which is used for face detection.
                                       # Simple: Get the face detection tool.
from utils.edgeface import get_model  # Detailed: Import the get_model function from the edgeface module to load the face recognition model.
                                       # Simple: Get the function to load the face model.
from utils.path_patch_v1 import get_resource_path  # Detailed: Import a helper function to get the correct file paths for resources in both development and executable mode.
                                                   # Simple: Helps find the right folder for files.

#############################################
# Initialize Face Detector and MediaPipe FaceMesh
#############################################

face_detector = FaceDetector(onnx_file=get_resource_path('checkpoints/scrfd_500m.onnx'))
# Detailed: Create an instance of FaceDetector using the ONNX model file located at the resource path 'checkpoints/scrfd_500m.onnx'.
# Simple: Set up the face detector with a pre-trained model file.
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,  # Detailed: Configure FaceMesh to treat input images as static (not video), which improves accuracy for individual images.
                             # Simple: Treat each picture on its own.
    max_num_faces=1,         # Detailed: Limit the detection to at most one face per image.
                             # Simple: Only detect one face.
    min_detection_confidence=0.5,  # Detailed: Set the minimum confidence threshold for face detection to 50%.
                                  # Simple: Only detect faces if 50% sure.
    min_tracking_confidence=0.5,   # Detailed: Set the minimum confidence for tracking face landmarks to 50%.
                                  # Simple: Only track if 50% sure.
    refine_landmarks=True          # Detailed: Enable refinement of detected landmarks for improved accuracy.
                                  # Simple: Make the face points more precise.
)

#############################################
# Define Transformation Pipeline for Face Preprocessing
#############################################

transform = transforms.Compose([
    transforms.ToTensor(),  # Detailed: Convert a PIL image to a PyTorch tensor.
                            # Simple: Turn the image into numbers.
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # Detailed: Normalize the tensor image using a mean and standard deviation of 0.5 for each of the three channels.
                                               # Simple: Scale the numbers to help the model.
])

#############################################
# Template Landmarks for Face Alignment (Normalized Coordinates)
#############################################

TEMPLATE_LANDMARKS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.6963],  # Detailed: Predefined coordinates for the centers of the eyes in a normalized face image.
                                             # Simple: Points for the eyes.
    [56.0252, 71.7366],                      # Detailed: Predefined coordinate for the tip of the nose.
                                             # Simple: Point for the nose.
    [41.5493, 92.3655], [70.7299, 92.3655]     # Detailed: Predefined coordinates for the corners of the mouth.
                                             # Simple: Points for the mouth corners.
]) / 112.0  # Detailed: Normalize the landmark coordinates by dividing by 112 to fit within a [0,1] range.
            # Simple: Scale down the points.

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
# Detailed: Define groups of landmark indices that correspond to different facial features (e.g., eyes, nose, mouth) for use in alignment.
# Simple: Group face points for features like eyes, nose, and mouth.

#############################################
# Load Face Recognition Model
#############################################

def load_face_recognition_model(model_path=get_resource_path('checkpoints/edgeface_s_gamma_05.pt')):
    # Detailed: Load the face recognition model from a checkpoint file. The model name is inferred automatically from the file name.
    # Simple: Load the face model from a file.
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # Detailed: Extract the model name by removing the file extension from the checkpoint file name.
    # Simple: Get the model name from the file name.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Detailed: Determine whether to use a GPU ("cuda") or CPU based on availability.
    # Simple: Use GPU if available, otherwise use CPU.
    
    model = get_model(model_name)
    # Detailed: Instantiate the model architecture using the get_model function from the edgeface module.
    # Simple: Build the face model.
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Detailed: Load the pre-trained model weights from the checkpoint, mapping them to the selected device.
    # Simple: Load the saved weights into the model.
    model.to(device)
    # Detailed: Transfer the model to the chosen computation device.
    # Simple: Put the model on GPU or CPU.
    model.eval()
    # Detailed: Set the model to evaluation mode to disable training-specific operations.
    # Simple: Make the model ready for testing.
    
    return model, device
    # Detailed: Return the loaded model and the device it is on.
    # Simple: Give back the model and the device.

#############################################
# Extract Face Landmarks using MediaPipe FaceMesh
#############################################

def get_landmarks(image):
    """Extract face landmarks from an image using MediaPipe FaceMesh."""
    # Detailed: Convert the image from BGR to RGB and process it using MediaPipe's FaceMesh to detect facial landmarks.
    # Simple: Find face points in the image.
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        # Detailed: If no face landmarks are detected, return None.
        # Simple: If no face points found, return nothing.
        return None

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    points = []

    # Iterate over each landmark group defined in LANDMARK_GROUPS
    for main_idx, surrounding_idx in [group[0] for group in LANDMARK_GROUPS]:
        # Detailed: For each group, calculate the mean coordinates for the main landmarks.
        # Simple: Average the main face points.
        main_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                               for idx in (main_idx if isinstance(main_idx, (tuple, list)) else [main_idx])], axis=0)
        
        if surrounding_idx:
            # Detailed: If there are surrounding landmark indices, calculate their mean and blend it with the main landmarks.
            # Simple: Average nearby face points and mix with the main ones.
            surrounding_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                                          for idx in surrounding_idx], axis=0)
            main_points = 0.8 * main_points + 0.2 * surrounding_points
        
        points.append(main_points)
    # Detailed: Return the array of computed landmark points as a NumPy array of type float32.
    # Simple: Return the face points.
    return np.array(points, dtype=np.float32)

#############################################
# Compute Transformation Matrix for Face Alignment
#############################################

def get_transform(src, dst):
    """Calculate the transformation matrix to align `src` landmarks to `dst` landmarks."""
    # Detailed: If the source and destination landmarks do not have the same shape, return None.
    # Simple: If the face points don't match the template points, stop.
    if src.shape != dst.shape:
        return None

    # Calculate centroids for source and destination landmarks.
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center the landmarks by subtracting their centroids.
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Calculate the scaling factor: ratio of the average distance of destination landmarks to source landmarks.
    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
    # Compute the rotation matrix using a least-squares solution.
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)

    # Perform Singular Value Decomposition (SVD) to compute the rotation.
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    
    # Ensure a proper rotation (determinant should be positive).
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # Construct the full transformation matrix T as a 3x3 matrix.
    T = np.eye(3)
    T[:2, :2] = scale * R  # Detailed: Combine scaling and rotation.
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)  # Detailed: Compute the translation vector.
    return T
    # Detailed: Return the computed transformation matrix.
    # Simple: Give back the matrix to align the face.

#############################################
# Face Detection using SCRFD
#############################################

def detect_faces(image):
    """Detect faces in an image using SCRFD."""
    # Detailed: Use the face_detector instance to detect faces with a fixed input size of 640x640.
    # Simple: Find faces in the image.
    det, _ = face_detector.detect(image, input_size=(640, 640))
    if det is None or len(det) == 0:
        # Detailed: If no detections are found, return empty arrays.
        # Simple: If no faces, return empty lists.
        return np.array([]), np.array([])

    # Detailed: Extract the bounding box coordinates and the associated confidence scores from the detections.
    # Simple: Get the boxes and scores for each detected face.
    boxes = det[:, :4]
    scores = det[:, 4]
    return boxes, scores

#############################################
# Face Alignment using Landmarks
#############################################

def align_face(face_img, size=(112, 112)):
    """Align a face image to a canonical pose using landmarks."""
    # Detailed: Extract facial landmarks from the input face image.
    # Simple: Find face points in the picture.
    landmarks = get_landmarks(face_img)
    if landmarks is None:
        return None

    # Detailed: Compute the transformation matrix that maps the detected landmarks to the template landmarks scaled by the desired size.
    # Simple: Calculate how to adjust the face points to match a standard face.
    tform = get_transform(landmarks, TEMPLATE_LANDMARKS * size)
    if tform is None:
        return None

    # Detailed: Apply a perspective warp to the face image using the transformation matrix, aligning it to the canonical pose.
    # Simple: Adjust the face image so that its features line up with the standard positions.
    warped = cv2.warpPerspective(face_img, tform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Detailed: Convert the warped image from BGR to RGB format and return it as a PIL Image.
    # Simple: Change the image color format and return it.
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

#############################################
# Preprocess an Image for Face Recognition
#############################################

def preprocess_image(image_path):
    """Preprocess an image for face recognition by detecting and aligning the faces."""
    # Detailed: Read an image from the specified path using OpenCV.
    # Simple: Open the image file.
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Detailed: Detect faces in the image; if none are found, return None.
    # Simple: Find faces, and if there are none, stop.
    boxes, _ = detect_faces(image)
    if not len(boxes):
        return None

    # Detailed: Iterate over each detected face, crop it, align it, and transform it into a tensor suitable for the face recognition model.
    # Simple: For each face found, cut it out, adjust it, and prepare it for the model.
    for box in boxes.astype(int):
        face_img = image[box[1]:box[3], box[0]:box[2]]
        aligned_face = align_face(face_img)
        if aligned_face:
            # Detailed: Apply the defined transformation pipeline to convert the aligned face image into a tensor, adding a batch dimension.
            # Simple: Convert the fixed face image into numbers for the model.
            return transform(aligned_face).unsqueeze(0)
    
    return None

#############################################
# Cleanup Function to Release MediaPipe Resources
#############################################

def cleanup():
    """Release resources used by MediaPipe FaceMesh."""
    # Detailed: Close the MediaPipe FaceMesh instance to free up system resources.
    # Simple: Shut down the face detector.
    mp_face_mesh.close()
