import cv2
import numpy as np
import MNN
import utils.box_utils_numpy as box_utils

# Constants
image_mean = np.array([127, 127, 127], dtype=np.float32)
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
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
    confidences = confidences.squeeze(0)
    boxes = boxes.squeeze(0)
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size == 0:
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

class FaceDetector:
    def __init__(self, model_path):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.scores_tensor = self.interpreter.getSessionOutput(self.session, "scores")
        self.boxes_tensor = self.interpreter.getSessionOutput(self.session, "boxes")

    def detect(self, frame):
        image_ori = frame
        image = process_image(image_ori)

        tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(tmp_input)

        self.interpreter.runSession(self.session)

        scores = self.scores_tensor.getData()
        boxes = self.boxes_tensor.getData()

        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

        boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, threshold)

        return boxes, labels, probs

def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(image.shape[1], x2 + 10)
        y2 = min(image.shape[0], y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return image