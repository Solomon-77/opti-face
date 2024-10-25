import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from ultralight import UltraLightDetector

face_detector = UltraLightDetector()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=20,
    min_detection_confidence=0.7, min_tracking_confidence=0.7,
    refine_landmarks=True
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

TEMPLATE_LANDMARKS = np.float32([
    [0.3150, 0.4600], [0.6850, 0.4600],  # Eyes
    [0.5000, 0.6500], [0.3500, 0.8250],  # Nose, Mouth
    [0.6500, 0.8250]
])

def detect_faces(image):
    return face_detector.detect_one(image)

def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
        
    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0]
    
    landmark_indices = [
        [([33, 133], [130, 243, 112, 156, 157, 158])],
        [([362, 263], [359, 466, 341, 384, 385, 386])],
        [([1], [2, 98, 327])],
        [([61, 91], [62, 87, 146, 177, 178])],
        [([291, 321], [292, 317, 375, 407, 408])]
    ]
    
    points = []
    for group in landmark_indices:
        main_idx, surrounding_idx = group[0]
        point = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                        for idx in (main_idx if isinstance(main_idx, (list, tuple)) else [main_idx])], axis=0)
        
        if surrounding_idx:
            surrounding = np.mean([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                                 for idx in surrounding_idx], axis=0)
            point = 0.8 * point + 0.2 * surrounding
        points.append(point)
    
    return np.array(points, dtype=np.float32)

def get_similarity_transform(src_points, dst_points):
    if src_points.shape[0] != dst_points.shape[0]:
        return None
        
    src_mean, dst_mean = np.mean(src_points, axis=0), np.mean(dst_points, axis=0)
    src_centered, dst_centered = src_points - src_mean, dst_points - dst_mean
    
    scale = np.mean(np.linalg.norm(dst_centered, axis=1)) / np.mean(np.linalg.norm(src_centered, axis=1))
    H = np.dot(dst_centered.T, src_centered) / np.sum(np.linalg.norm(src_centered, axis=1) ** 2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    
    if np.linalg.det(R) < 0:
        R = np.dot(U, np.dot(np.diag([1, -1]), Vt))
    
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = dst_mean - np.dot(scale * R, src_mean)
    
    return T

def align_face(face_img, target_size=(112, 112)):
    landmarks = get_face_landmarks(face_img)
    if landmarks is None:
        return None
        
    target_landmarks = TEMPLATE_LANDMARKS.copy() * target_size
    tform = get_similarity_transform(landmarks, target_landmarks)
    if tform is None:
        return None
    
    warped = cv2.warpPerspective(face_img, tform, target_size,
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    boxes, _ = detect_faces(image)
    if len(boxes) == 0:
        return None
        
    for box in boxes.astype(int):
        x1, y1, x2, y2 = [max(0, v) for v in (box[:2] - 10)] + [min(v, s) for v, s in zip(box[2:] + 10, image.shape[:2][::-1])]
        face = align_face(image[y1:y2, x1:x2])
        if face:
            return transform(face).unsqueeze(0)
    return None

def cleanup():
    face_mesh.close()