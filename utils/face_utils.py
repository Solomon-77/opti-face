import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from ultralight import UltraLightDetector

# Load the UltraLight face detection model
face_detector = UltraLightDetector()

# Initialize MediaPipe Face Mesh with higher confidence threshold
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=20,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_landmarks=True  # Enable refined landmarks for better accuracy
)

# Preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Refined template landmarks positions (calibrated to match MTCNN more closely)
TEMPLATE_LANDMARKS = np.float32([
    [0.3150, 0.4600],  # Left eye
    [0.6850, 0.4600],  # Right eye
    [0.5000, 0.6500],  # Nose tip
    [0.3500, 0.8250],  # Left mouth corner
    [0.6500, 0.8250]   # Right mouth corner
])

def detect_faces(image):
    boxes, scores = face_detector.detect_one(image)
    return boxes, scores

def get_refined_landmark_points(landmarks, idx_list, h, w):
    """Get refined landmark points by averaging nearby points for stability."""
    points = []
    for landmark_group in idx_list:
        main_idx, surrounding_idx = landmark_group[0]  # Unpack from the first tuple in the group
        
        if isinstance(main_idx, (list, tuple)):
            # Average multiple points for main landmark
            point = np.mean([[landmarks.landmark[idx].x * w,
                            landmarks.landmark[idx].y * h] for idx in main_idx], axis=0)
        else:
            point = np.array([landmarks.landmark[main_idx].x * w,
                            landmarks.landmark[main_idx].y * h])
            
        # Add influence from surrounding points if provided
        if surrounding_idx:
            surrounding_points = np.array([[landmarks.landmark[idx].x * w,
                                          landmarks.landmark[idx].y * h] 
                                         for idx in surrounding_idx])
            point = 0.8 * point + 0.2 * np.mean(surrounding_points, axis=0)
            
        points.append(point)
    return np.array(points, dtype=np.float32)

def get_face_landmarks(image):
    """Get facial landmarks using MediaPipe Face Mesh with refined key points."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    
    # Refined landmark indices with surrounding points for better stability
    # Format: [(main_point_idx/indices, [surrounding_point_indices])]
    refined_landmarks_idx = [
        [([33, 133], [130, 243, 112, 156, 157, 158])],  # Left eye
        [([362, 263], [359, 466, 341, 384, 385, 386])],  # Right eye
        [([1], [2, 98, 327])],  # Nose
        [([61, 91], [62, 87, 146, 177, 178])],  # Left mouth
        [([291, 321], [292, 317, 375, 407, 408])]  # Right mouth
    ]
    
    points = get_refined_landmark_points(landmarks, refined_landmarks_idx, h, w)
    return points

def get_similarity_transform(src_points, dst_points):
    """Calculate similarity transform matrix with improved stability."""
    if src_points.shape[0] != dst_points.shape[0]:
        return None
        
    # Calculate centroid and normalize points
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    
    # Calculate rotation and scale
    src_norm = np.linalg.norm(src_centered, axis=1)
    dst_norm = np.linalg.norm(dst_centered, axis=1)
    scale = np.mean(dst_norm) / np.mean(src_norm)
    
    H = np.dot(dst_centered.T, src_centered) / np.sum(src_norm ** 2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    
    if np.linalg.det(R) < 0:
        R = np.dot(U, np.dot(np.diag([1, -1]), Vt))
    
    # Combine scale, rotation and translation
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)
    
    return T

def align_face(face_img, target_size=(112, 112)):
    """Align face with improved transformation stability."""
    landmarks = get_face_landmarks(face_img)
    
    if landmarks is None:
        return None
        
    h, w = face_img.shape[:2]
    
    # Calculate target landmarks
    target_landmarks = TEMPLATE_LANDMARKS.copy()
    target_landmarks[:, 0] *= target_size[0]
    target_landmarks[:, 1] *= target_size[1]
    
    # Get transformation matrix
    tform = get_similarity_transform(landmarks, target_landmarks)
    if tform is None:
        return None
    
    # Apply transformation with improved interpolation
    warped = cv2.warpPerspective(
        face_img, 
        tform, 
        target_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    boxes, _ = detect_faces(image)
    
    if len(boxes) == 0:
        return None
        
    for x1, y1, x2, y2 in boxes.astype(int):
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        face_img = image[y1:y2, x1:x2]
        
        aligned_face = align_face(face_img)
        
        if aligned_face:
            return transform(aligned_face).unsqueeze(0)
    
    return None

def preprocess_frame(frame):
    boxes, _ = detect_faces(frame)
    faces = []
    
    for x1, y1, x2, y2 in boxes.astype(int):
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        face_img = frame[y1:y2, x1:x2]
        
        aligned_face = align_face(face_img)
        
        if aligned_face:
            face_tensor = transform(aligned_face).unsqueeze(0)
            faces.append((face_tensor, (x1, y1, x2, y2)))
    
    return faces

def cleanup():
    """Clean up MediaPipe resources"""
    face_mesh.close()