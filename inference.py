import cv2
from utils.face_detection import FaceDetector, draw_boxes

detector_path = "./models/version-RFB/RFB-320.mnn"

def inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = FaceDetector(detector_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        boxes, labels, probs = detector.detect(frame)
        frame_with_boxes = draw_boxes(frame, boxes)

        cv2.imshow("UltraFace MNN Real-Time", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()