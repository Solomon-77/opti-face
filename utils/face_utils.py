import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from utils.face_detection import FaceDetector

# Initialize face detector and MediaPipe FaceMesh
face_detector = FaceDetector(onnx_file='checkpoints/scrfd.onnx')
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=20,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_landmarks=True
)

# Transformation pipeline for face preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Template landmarks for face alignment (normalized coordinates)
TEMPLATE_LANDMARKS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.6963],  # Eyes
    [56.0252, 71.7366],                      # Nose
    [41.5493, 92.3655], [70.7299, 92.3655]   # Mouth corners
]) / 112.0

# Landmark groups for different facial features
LANDMARK_GROUPS = [
    [([33, 133], [130, 243, 112, 156, 157, 158])],     # Left eye
    [([362, 263], [359, 466, 341, 384, 385, 386])],    # Right eye
    [([1], [2, 98, 327, 168])],                        # Nose
    [([61, 91], [62, 87, 146, 177, 178])],             # Left mouth
    [([291, 321], [292, 317, 375, 407, 408])]          # Right mouth
]

def get_landmarks(image):
    """Extract face landmarks from an image using MediaPipe FaceMesh."""
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    points = []

    for main_idx, surrounding_idx in [group[0] for group in LANDMARK_GROUPS]:
        # Calculate the mean position of the main indices
        main_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                               for idx in (main_idx if isinstance(main_idx, (tuple, list)) else [main_idx])], axis=0)
        
        if surrounding_idx:
            # Calculate the mean position of surrounding indices, if available
            surrounding_points = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                                          for idx in surrounding_idx], axis=0)
            main_points = 0.8 * main_points + 0.2 * surrounding_points
        
        points.append(main_points)

    return np.array(points, dtype=np.float32)

def get_transform(src, dst):
    """Calculate the transformation matrix to align `src` landmarks to `dst` landmarks."""
    if src.shape != dst.shape:
        return None

    # Calculate centroids
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # Center the landmarks
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Calculate the scale and rotation matrix
    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)

    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    
    # Ensure correct orientation
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # Construct the transformation matrix
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)
    return T

def detect_faces(image):
    """Detect faces in an image using SCRFD."""
    det, _ = face_detector.detect(image, thresh=0.5, input_size=(640, 640))
    if det is None or len(det) == 0:
        return np.array([]), np.array([])

    # Extract bounding boxes and confidence scores
    boxes = det[:, :4]
    scores = det[:, 4]
    return boxes, scores

def align_face(face_img, size=(112, 112)):
    """Align a face image to a canonical pose using landmarks."""
    landmarks = get_landmarks(face_img)
    if landmarks is None:
        return None

    # Calculate the transformation matrix for alignment
    tform = get_transform(landmarks, TEMPLATE_LANDMARKS * size)
    if tform is None:
        return None

    # Warp the face image to align it
    warped = cv2.warpPerspective(face_img, tform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def preprocess_image(image_path):
    """Preprocess an image for face recognition by detecting and aligning the faces."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    boxes, _ = detect_faces(image)
    if not len(boxes):
        return None

    # Process each detected face
    for box in boxes.astype(int):
        face_img = image[box[1]:box[3], box[0]:box[2]]
        aligned_face = align_face(face_img)
        if aligned_face:
            # Return the transformed face tensor
            return transform(aligned_face).unsqueeze(0)
    
    return None

def cleanup():
    """Release resources used by MediaPipe FaceMesh."""
    mp_face_mesh.close()