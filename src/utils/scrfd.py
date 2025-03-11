import os  # Detailed: Import the os module to interact with the operating system (e.g., for file paths and directory operations).
         # Simple: Lets us work with files and folders.
import cv2  # Detailed: Import OpenCV for image processing tasks such as reading, resizing, and drawing on images.
         # Simple: Helps us work with images.
import numpy as np  # Detailed: Import NumPy for numerical operations and handling arrays.
                   # Simple: Used for math and arrays.

#############################################
# Helper Functions: Converting Distances to Bounding Boxes and Keypoints
#############################################

def distance2bbox(points, distance, max_shape=None):
    # Detailed: Compute bounding box coordinates by subtracting and adding distances to the center points.
    # Simple: Calculate box corners using center points and distances.
    x1 = points[:, 0] - distance[:, 0]  # Detailed: x1 is computed by subtracting the left distance from the x-coordinate.
                                       # Simple: Get the left x-coordinate.
    y1 = points[:, 1] - distance[:, 1]  # Detailed: y1 is computed by subtracting the top distance from the y-coordinate.
                                       # Simple: Get the top y-coordinate.
    x2 = points[:, 0] + distance[:, 2]  # Detailed: x2 is computed by adding the right distance to the x-coordinate.
                                       # Simple: Get the right x-coordinate.
    y2 = points[:, 1] + distance[:, 3]  # Detailed: y2 is computed by adding the bottom distance to the y-coordinate.
                                       # Simple: Get the bottom y-coordinate.
    if max_shape is not None:
        # Detailed: If a maximum shape is provided, clamp the bounding box coordinates within the image dimensions.
        # Simple: Keep the box inside the image.
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    # Detailed: Stack the four coordinates along the last axis to form complete bounding boxes.
    # Simple: Combine the corners into a box.
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    # Detailed: Compute keypoint coordinates from distances relative to center points.
    # Simple: Calculate face keypoints from center points and distances.
    preds = []
    # Detailed: Iterate through the distance array with a step of 2, because keypoints are represented by (x, y) pairs.
    # Simple: Process pairs of numbers for each keypoint.
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]  # Detailed: Compute x-coordinate of the keypoint by adding the corresponding distance.
                                                 # Simple: Get keypoint x-coordinate.
        py = points[:, i % 2 + 1] + distance[:, i + 1]  # Detailed: Compute y-coordinate of the keypoint similarly.
                                                       # Simple: Get keypoint y-coordinate.
        if max_shape is not None:
            # Detailed: Clamp the keypoint coordinates to ensure they lie within the image boundaries.
            # Simple: Keep keypoints inside the image.
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)  # Detailed: Append the computed x-coordinate.
                          # Simple: Save keypoint x.
        preds.append(py)  # Detailed: Append the computed y-coordinate.
                          # Simple: Save keypoint y.
    # Detailed: Stack all keypoint coordinates along the last axis to produce a structured output.
    # Simple: Combine the keypoint coordinates.
    return np.stack(preds, axis=-1)

#############################################
# FaceDetector Class: Using SCRFD ONNX Model for Face Detection
#############################################

class FaceDetector:
    def __init__(self, onnx_file=None, session=None):
        # Detailed: Initialize the face detector. Optionally use an existing ONNX runtime session, or create one using the provided ONNX file.
        # Simple: Set up the face detector using a model file or given session.
        from onnxruntime import InferenceSession  # Detailed: Import InferenceSession from onnxruntime to run the ONNX model.
                                                  # Simple: Lets us run the ONNX model.
        self.session = session  # Detailed: If a session is passed, use it.
                               # Simple: Use the provided session if available.

        self.batched = False  # Detailed: Initialize a flag to indicate if the model outputs are batched.
                              # Simple: Start with no batching.
        if self.session is None:
            # Detailed: If no session is provided, check that an ONNX file is provided and exists, then create a session using CUDA.
            # Simple: If no session, load the model from the file.
            assert onnx_file is not None
            assert os.path.exists(onnx_file)
            self.session = InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])

        self.nms_thresh = 0.4  # Detailed: Set the threshold for Non-Maximum Suppression (NMS) to filter overlapping detections.
                              # Simple: Define a threshold to remove duplicate boxes.
        self.center_cache = {}  # Detailed: Initialize a cache to store computed anchor centers for different feature map shapes.
                               # Simple: Cache computed center points.
        self._init_vars()  # Detailed: Initialize variables based on the model's input and output configurations.
                          # Simple: Set up model-specific parameters.

    def _init_vars(self):
        # Detailed: Internal method to initialize model-specific variables like input size, input name, output names, and flags for keypoint usage.
        # Simple: Set up important parameters from the model.
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape  # Detailed: Get the shape of the input tensor.
                                      # Simple: Read the input dimensions.
        if isinstance(input_shape[2], str):
            self.input_size = None  # Detailed: If the input dimensions are dynamic (given as strings), set input_size to None.
                                   # Simple: If the size is not fixed, use None.
        else:
            self.input_size = tuple(input_shape[2:4][::-1])  # Detailed: Otherwise, define input_size as (width, height) by reversing dimensions.
                                                           # Simple: Set the fixed input size.
        input_name = input_cfg.name  # Detailed: Retrieve the name of the input tensor.
                                    # Simple: Save the input tensor's name.
        outputs = self.session.get_outputs()  # Detailed: Retrieve the model's outputs.
                                             # Simple: Get the outputs.
        if len(outputs[0].shape) == 3:
            self.batched = True  # Detailed: Determine if the model output is batched based on the number of dimensions.
                                # Simple: Set batched mode if the output has 3 dimensions.
        output_names = []
        for o in outputs:
            output_names.append(o.name)  # Detailed: Collect the names of each output tensor.
                                        # Simple: Save each output's name.
        self.input_name = input_name  # Detailed: Store the input tensor name.
                                     # Simple: Remember the input name.
        self.output_names = output_names  # Detailed: Store all output tensor names.
                                         # Simple: Remember the output names.
        self.use_kps = False  # Detailed: Initialize a flag for using keypoints (landmark detection) to False.
                            # Simple: Start with no keypoints.
        self._num_anchors = 1  # Detailed: Initialize the number of anchors per spatial location to 1.
                             # Simple: Default to one anchor.
        if len(outputs) == 6:
            # Detailed: For models with 6 outputs, configure parameters for feature map count, stride values, and anchor count.
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            # Detailed: For models with 9 outputs, configure similarly but enable keypoints.
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            # Detailed: For models with 10 outputs, set parameters for a deeper feature pyramid.
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            # Detailed: For models with 15 outputs, configure for keypoint usage and a larger feature pyramid.
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, image, thresh):
        # Detailed: Run a forward pass on the preprocessed image to compute detection scores, bounding boxes, and keypoints.
        # Simple: Process the image through the model to get face scores and boxes.
        scores_list = []  # Detailed: Initialize a list to collect detection scores from different feature map levels.
                         # Simple: List for detection scores.
        bboxes_list = []  # Detailed: Initialize a list to collect bounding boxes.
                          # Simple: List for face boxes.
        kps_list = []  # Detailed: Initialize a list to collect keypoints (if available).
                     # Simple: List for face keypoints.
        input_size = tuple(image.shape[0:2][::-1])  # Detailed: Determine the input size (width, height) from the image.
                                                   # Simple: Get the image size.
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128, input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        # Detailed: Convert the image into a blob (a preprocessed format) with scaling and mean subtraction.
        # Simple: Prepare the image for the model.
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # Detailed: Run the model with the blob as input to get outputs corresponding to scores, bounding boxes, and keypoints.
        # Simple: Get the model's output.

        fmc = self.fmc  # Detailed: Use the configured feature map count.
        input_width = blob.shape[3]  # Detailed: Get the blob width.
        input_height = blob.shape[2]  # Detailed: Get the blob height.

        # Detailed: Loop over each stride in the feature pyramid and process outputs.
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                # Detailed: If the model output is batched, take the first element of each output.
                scores = net_outs[idx][0]
                boxes = net_outs[idx + fmc][0]
                boxes = boxes * stride  # Detailed: Scale the bounding box predictions by the stride.
                if self.use_kps:
                    points = net_outs[idx + fmc * 2][0] * stride  # Detailed: Scale keypoint predictions if available.
            else:
                # Detailed: For non-batched outputs, use them directly.
                scores = net_outs[idx]
                boxes = net_outs[idx + fmc]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2] * stride

            # Detailed: Calculate the height and width of the feature map for this stride.
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)  # Detailed: Create a cache key based on feature map dimensions and stride.
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]  # Detailed: Retrieve cached anchor centers if available.
            else:
                # Detailed: Generate a grid of anchor centers using np.mgrid and scale them by stride.
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    # Detailed: If multiple anchors per location are used, duplicate the centers accordingly.
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers  # Detailed: Cache the computed anchor centers.
            pos_index = np.where(scores >= thresh)[0]  # Detailed: Identify indices where the detection score meets or exceeds the threshold.
            bboxes = distance2bbox(anchor_centers, boxes)  # Detailed: Convert distance predictions to bounding boxes using anchor centers.
            pos_scores = scores[pos_index]  # Detailed: Select scores that exceed the threshold.
            pos_bboxes = bboxes[pos_index]  # Detailed: Select corresponding bounding boxes.
            scores_list.append(pos_scores)  # Detailed: Append these scores to the list.
            bboxes_list.append(pos_bboxes)  # Detailed: Append these boxes to the list.
            if self.use_kps:
                kpss = distance2kps(anchor_centers, points)  # Detailed: Compute keypoints from distances and anchor centers.
                kpss = kpss.reshape((kpss.shape[0], -1, 2))  # Detailed: Reshape the keypoints to have (N, num_keypoints, 2) dimensions.
                kps_list.append(kpss[pos_index])  # Detailed: Append keypoints corresponding to high scoring detections.
        return scores_list, bboxes_list, kps_list  # Detailed: Return lists containing scores, bounding boxes, and keypoints from all feature map levels.

    def detect(self, image, thresh=0.5, input_size=None, max_num=0, metric='default'):
        # Detailed: Detect faces in the image by resizing, running forward pass, applying NMS, and optionally selecting top detections.
        # Simple: Find faces in the image and filter out overlaps.
        assert input_size is not None or self.input_size is not None  # Detailed: Ensure an input size is provided either as a parameter or from the model.
        input_size = self.input_size if input_size is None else input_size

        # Detailed: Compute aspect ratios for the input image and the model to determine resizing dimensions while maintaining aspect ratio.
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]  # Detailed: Compute the scaling factor used during resizing.
        resized_img = cv2.resize(image, (new_width, new_height))  # Detailed: Resize the image to the computed dimensions.
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)  # Detailed: Create a blank image of the model's input size.
        det_img[:new_height, :new_width, :] = resized_img  # Detailed: Place the resized image onto the blank image.

        # Detailed: Run the forward pass to get detection outputs.
        scores_list, bboxes_list, kps_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)  # Detailed: Concatenate scores from all feature levels.
        scores_ravel = scores.ravel()  # Detailed: Flatten the scores array.
        order = scores_ravel.argsort()[::-1]  # Detailed: Sort indices of detections based on descending score order.
        bboxes = np.vstack(bboxes_list) / det_scale  # Detailed: Concatenate bounding boxes and scale them back to the original image size.
        if self.use_kps:
            kpss = np.vstack(kps_list) / det_scale  # Detailed: Do the same for keypoints if they are used.
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)  # Detailed: Combine bounding boxes and scores into a single array.
        pre_det = pre_det[order, :]  # Detailed: Reorder detections based on the sorted scores.
        keep = self.nms(pre_det)  # Detailed: Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
        det = pre_det[keep, :]  # Detailed: Select the detections that survive NMS.
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if 0 < max_num < det.shape[0]:
            # Detailed: If a maximum number of detections is specified, limit the results by computing a metric that considers area and centering.
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # Detailed: Penalize boxes far from the center.
            bindex = np.argsort(values)[::-1]  # Detailed: Sort the boxes based on the computed metric.
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss  # Detailed: Return the final detections and keypoints.
                             # Simple: Output the face boxes and keypoints.

    def nms(self, dets):
        # Detailed: Perform Non-Maximum Suppression (NMS) to remove overlapping detections based on a threshold.
        # Simple: Filter out duplicate overlapping boxes.
        thresh = self.nms_thresh  # Detailed: Use the configured NMS threshold.
        x1 = dets[:, 0]  # Detailed: Extract x1 coordinates (left side) from detections.
        y1 = dets[:, 1]  # Detailed: Extract y1 coordinates (top side).
        x2 = dets[:, 2]  # Detailed: Extract x2 coordinates (right side).
        y2 = dets[:, 3]  # Detailed: Extract y2 coordinates (bottom side).
        scores = dets[:, 4]  # Detailed: Extract detection confidence scores.

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # Detailed: Compute the area for each detection box.
        order = scores.argsort()[::-1]  # Detailed: Sort the detections in descending order based on scores.

        keep = []  # Detailed: Initialize a list to hold indices of boxes to keep.
        while order.size > 0:
            i = order[0]  # Detailed: Select the detection with the highest score.
            keep.append(i)  # Detailed: Keep this detection.
            # Detailed: Compute the coordinates for the intersection between the current box and the rest.
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)  # Detailed: Compute the width of the intersection.
            h = np.maximum(0.0, yy2 - yy1 + 1)  # Detailed: Compute the height of the intersection.
            inter = w * h  # Detailed: Calculate the area of intersection.
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  # Detailed: Compute the Intersection over Union (IoU) for overlap.
            index = np.where(ovr <= thresh)[0]  # Detailed: Identify indices where IoU is below the threshold.
            order = order[index + 1]  # Detailed: Update the order list to exclude boxes with high overlap.
        return keep  # Detailed: Return the list of indices for boxes that passed NMS.
                   # Simple: Output the indices of good boxes.
