import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from ultralight import UltraLightDetector

# Initialize face detector and face mesh
face_detector = UltraLightDetector()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=20,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_landmarks=True
)

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Template landmarks for face alignment (normalized to [0, 1] range for an image of size 112x112)
TEMPLATE_LANDMARKS = np.float32([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.6963],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.3655]   # Right mouth corner
]) / 112.0

def detect_faces(image):
    """Detect faces in an image using the UltraLightDetector."""
    return face_detector.detect_one(image)

def get_face_landmarks(image):
    """Extract face landmarks from an image using Mediapipe's FaceMesh."""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]  # Assuming we are using the first detected face

    # Define specific landmark groups for eyes, nose, mouth
    landmark_indices = [
        [([33, 133], [130, 243, 112, 156, 157, 158])],  # Left eye and surroundings
        [([362, 263], [359, 466, 341, 384, 385, 386])],  # Right eye and surroundings
        [([1], [2, 98, 327])],                          # Nose and surroundings
        [([61, 91], [62, 87, 146, 177, 178])],          # Left mouth corner and surroundings
        [([291, 321], [292, 317, 375, 407, 408])]       # Right mouth corner and surroundings
    ]

    points = []
    for group in landmark_indices:
        main_idx, surrounding_idx = group[0]
        
        # Compute the main point from the group of key landmarks
        point = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                         for idx in (main_idx if isinstance(main_idx, (list, tuple)) else [main_idx])], axis=0)
        
        # Adjust point by averaging with surrounding landmarks
        if surrounding_idx:
            surrounding = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                                   for idx in surrounding_idx], axis=0)
            point = 0.8 * point + 0.2 * surrounding
        
        points.append(point)
    
    return np.array(points, dtype=np.float32)

def get_similarity_transform(src_points, dst_points):
    """Get the similarity transform (rotation, scaling, translation) between two sets of points."""
    if src_points.shape[0] != dst_points.shape[0]:
        return None

    # Compute centroids
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)

    # Center the points around the origin
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # Compute scaling factor
    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))

    # Compute rotation matrix using Singular Value Decomposition (SVD)
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)

    # Correct for possible reflection in the rotation matrix
    if np.linalg.det(R) < 0:
        R = np.dot(U, np.dot(np.diag([1, -1]), Vt))

    # Create the transformation matrix
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)

    return T

def align_face(face_img, target_size=(112, 112)):
    """Align a face image to a canonical pose using predefined template landmarks."""
    landmarks = get_face_landmarks(face_img)
    if landmarks is None:
        return None

    # Scale template landmarks to the target size
    target_landmarks = TEMPLATE_LANDMARKS.copy() * target_size

    # Get the transformation matrix
    tform = get_similarity_transform(landmarks, target_landmarks)
    if tform is None:
        return None

    # Apply the perspective warp to align the face
    warped = cv2.warpPerspective(
        face_img, tform, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Convert the aligned image to PIL format
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def preprocess_image(image_path):
    """Preprocess an image: detect faces, align them, and apply transformations."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Detect faces in the image
    boxes, _ = detect_faces(image)
    if len(boxes) == 0:
        return None

    # Align each detected face
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        face = align_face(image[y1:y2, x1:x2])
        if face:
            return transform(face).unsqueeze(0)

    return None

def cleanup():
    """Release resources used by Mediapipe FaceMesh."""
    face_mesh.close()