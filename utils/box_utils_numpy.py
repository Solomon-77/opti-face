import numpy as np
from math import ceil

def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    priors = np.expand_dims(priors, 0) if len(priors.shape) + 1 == len(locations.shape) else priors
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)

def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    center_form_priors = np.expand_dims(center_form_priors, 0) if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape) else center_form_priors
    return np.concatenate([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=len(center_form_boxes.shape) - 1)

def area_of(left_top, right_bottom):
    hw = np.maximum(right_bottom - left_top, 0.0)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], axis=-1)

def corner_form_to_center_form(boxes):
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], axis=-1)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    
    # Adjust candidate_size to not exceed the length of scores
    candidate_size = min(candidate_size, len(scores))

    # Get indices of top candidate_size scores
    top_scores_indices = np.argpartition(-scores, candidate_size - 1)[:candidate_size]
    top_scores_indices = top_scores_indices[np.argsort(-scores[top_scores_indices])]

    picked = []
    while top_scores_indices.size > 0:
        current = top_scores_indices[0]
        picked.append(current)
        if 0 < top_k == len(picked):
            break
        current_box = boxes[current]
        iou = iou_of(boxes[top_scores_indices[1:]], current_box[np.newaxis])
        top_scores_indices = top_scores_indices[1:][iou <= iou_threshold]

    return box_scores[picked]

def define_img_size(image_size):
    shrinkage_list = [8, 16, 32, 64]
    feature_map_w_h_list = [[ceil(image_size[0] / size), ceil(image_size[1] / size)] for size in shrinkage_list]
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(len(feature_map_list)):
        scale_w = image_size[0] / shrinkage_list[index]
        scale_h = image_size[1] / shrinkage_list[index]
        for j in range(feature_map_list[index][1]):
            for i in range(feature_map_list[index][0]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h
                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    return np.array(priors)