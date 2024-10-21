import cv2
from PIL import Image
from torchvision import transforms
from ultralight import UltraLightDetector
from face_alignment import align

# Load the UltraLight face detection model
face_detector = UltraLightDetector()

# Preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def detect_faces(image):
    boxes, scores = face_detector.detect_one(image)
    return boxes, scores

def align_face(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    aligned_face, _ = align.get_aligned_face(rgb_pil_image=pil_img)
    return aligned_face

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    boxes, _ = detect_faces(image)
    
    for x1, y1, x2, y2 in boxes.astype(int):
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2)
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
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2)
        face_img = frame[y1:y2, x1:x2]
        
        aligned_face = align_face(face_img)
        
        if aligned_face:
            face_tensor = transform(aligned_face).unsqueeze(0)
            faces.append((face_tensor, (x1, y1, x2, y2)))
    
    return faces