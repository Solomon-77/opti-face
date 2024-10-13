import cv2
import numpy as np
import MNN
import utils.box_utils_numpy as box_utils

# Constants
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
model_path = "./models/version-RFB/RFB-320.mnn"
input_size = [320, 240]
threshold = 0.5

# Pre-compute priors
priors = box_utils.define_img_size(input_size)

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size), interpolation=cv2.INTER_LINEAR)
    image = (image - image_mean) / image_std
    return image.transpose((2, 0, 1)).astype(np.float32)

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, :4] *= np.array([width, height, width, height])
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    # Pre-allocate output tensors
    scores_tensor = interpreter.getSessionOutput(session, "scores")
    boxes_tensor = interpreter.getSessionOutput(session, "boxes")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        image_ori = frame
        image = process_image(image_ori)
        
        tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)

        interpreter.runSession(session)

        scores = scores_tensor.getData()
        boxes = boxes_tensor.getData()

        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

        boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1 = max(0, box[0] - 10)
            y1 = max(0, box[1] - 10)
            x2 = min(image_ori.shape[1], box[2] + 10)
            y2 = min(image_ori.shape[0], box[3])
            cv2.rectangle(image_ori, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.imshow("UltraFace MNN Real-Time", image_ori)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()