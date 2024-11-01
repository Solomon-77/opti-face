import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from ultralight import UltraLightDetector

# Initialize global components
face_detector = UltraLightDetector()
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=20,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    refine_landmarks=True
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Constants
TEMPLATE_LANDMARKS = np.float32([
    [38.2946, 51.6963], [73.5318, 51.6963],  # Eyes
    [56.0252, 71.7366],                       # Nose
    [41.5493, 92.3655], [70.7299, 92.3655]   # Mouth corners
]) / 112.0

LANDMARK_GROUPS = [
    [([33, 133], [130, 243, 112, 156, 157, 158])],     # Left eye
    [([362, 263], [359, 466, 341, 384, 385, 386])],    # Right eye
    [([1], [2, 98, 327])],                             # Nose
    [([61, 91], [62, 87, 146, 177, 178])],             # Left mouth
    [([291, 321], [292, 317, 375, 407, 408])]          # Right mouth
]

def get_landmarks(image):
    """Extract face landmarks from an image."""
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    points = []

    for main_idx, surrounding_idx in [group[0] for group in LANDMARK_GROUPS]:
        point = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                       for idx in (main_idx if isinstance(main_idx, (tuple, list)) else [main_idx])], axis=0)
        
        if surrounding_idx:
            surrounding = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                                 for idx in surrounding_idx], axis=0)
            point = 0.8 * point + 0.2 * surrounding
        
        points.append(point)

    return np.array(points, dtype=np.float32)

def get_transform(src, dst):
    """Calculate transformation matrix."""
    if src.shape != dst.shape:
        return None

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt if np.linalg.det(np.dot(U, Vt)) > 0 else np.dot(np.diag([1, -1]), Vt))

    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)
    return T

def detect_faces(image):
    """Detect faces in an image."""
    return face_detector.detect_one(image)

def align_face(face_img, size=(112, 112)):
    """Align a face image to a canonical pose."""
    landmarks = get_landmarks(face_img)
    if landmarks is None:
        return None

    tform = get_transform(landmarks, TEMPLATE_LANDMARKS * size)
    if tform is None:
        return None

    warped = cv2.warpPerspective(face_img, tform, size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def preprocess_image(image_path):
    """Preprocess an image for face recognition."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    boxes, _ = detect_faces(image)
    if not len(boxes):
        return None

    for box in boxes.astype(int):
        face = align_face(image[box[1]:box[3], box[0]:box[2]])
        if face:
            return transform(face).unsqueeze(0)
    return None

def cleanup():
    """Release resources."""
    mp_face_mesh.close()