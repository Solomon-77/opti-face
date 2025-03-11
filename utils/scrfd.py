import os  # Detailed: Import the os module to interact with the operating system (e.g., file paths, directory operations).
           # Simple: Lets us work with files and folders.
import cv2  # Detailed: Import OpenCV for image and video processing tasks.
           # Simple: Helps with handling images and videos.
import numpy as np  # Detailed: Import NumPy for numerical operations and array manipulations.
                   # Simple: Used for math and working with arrays.

#############################################
# Helper Functions for Bounding Boxes and Keypoints
#############################################

def distance2bbox(points, distance, max_shape=None):
    """
    Detailed: Calculate bounding box coordinates from center points and distances.
              The function subtracts and adds the distance to the center point to obtain the box coordinates.
    Simple: Compute box corners using a center point and distances.
    """
    # Calculate top-left coordinates by subtracting the left and top distances
    x1 = points[:, 0] - distance[:, 0]  # Detailed: Compute x-coordinate of top-left corner.
                                        # Simple: Get left x-coordinate.
    y1 = points[:, 1] - distance[:, 1]  # Detailed: Compute y-coordinate of top-left corner.
                                        # Simple: Get top y-coordinate.
    # Calculate bottom-right coordinates by adding the right and bottom distances
    x2 = points[:, 0] + distance[:, 2]  # Detailed: Compute x-coordinate of bottom-right corner.
                                        # Simple: Get right x-coordinate.
    y2 = points[:, 1] + distance[:, 3]  # Detailed: Compute y-coordinate of bottom-right corner.
                                        # Simple: Get bottom y-coordinate.
    if max_shape is not None:
        # Clamp the coordinates to be within the maximum shape dimensions
        x1 = x1.clamp(min=0, max=max_shape[1])  # Detailed: Ensure x1 does not go below 0 or above the image width.
                                               # Simple: Keep x1 inside the image.
        y1 = y1.clamp(min=0, max=max_shape[0])  # Detailed: Ensure y1 does not go below 0 or above the image height.
                                               # Simple: Keep y1 inside the image.
        x2 = x2.clamp(min=0, max=max_shape[1])  # Detailed: Ensure x2 is within valid width boundaries.
                                               # Simple: Keep x2 inside the image.
        y2 = y2.clamp(min=0, max=max_shape[0])  # Detailed: Ensure y2 is within valid height boundaries.
                                               # Simple: Keep y2 inside the image.
    # Stack the computed coordinates along the last axis to form a complete bounding box array
    return np.stack([x1, y1, x2, y2], axis=-1)  # Detailed: Combine x1, y1, x2, y2 into a single array for each bounding box.
                                                # Simple: Put the corners together into a box.

def distance2kps(points, distance, max_shape=None):
    """
    Detailed: Compute keypoints positions from center points and corresponding distances.
              Iterates over the distance array two elements at a time to compute x and y coordinates.
    Simple: Calculate face keypoints from center points and distances.
    """
    preds = []
    # Loop over the distances in steps of 2 (for x and y coordinates)
    for i in range(0, distance.shape[1], 2):
        # Compute the x-coordinate by adding the corresponding distance to the x value of the center point
        px = points[:, i % 2] + distance[:, i]  # Detailed: Calculate keypoint x-coordinate based on distance.
                                                 # Simple: Get keypoint x-coordinate.
        # Compute the y-coordinate by adding the next distance value to the y value of the center point
        py = points[:, i % 2 + 1] + distance[:, i + 1]  # Detailed: Calculate keypoint y-coordinate based on distance.
                                                        # Simple: Get keypoint y-coordinate.
        if max_shape is not None:
            # Clamp the keypoints to ensure they are within image boundaries
            px = px.clamp(min=0, max=max_shape[1])  # Detailed: Limit x-coordinate to the valid range.
                                                    # Simple: Keep keypoint x inside.
            py = py.clamp(min=0, max=max_shape[0])  # Detailed: Limit y-coordinate to the valid range.
                                                    # Simple: Keep keypoint y inside.
        preds.append(px)  # Detailed: Append computed x-coordinate to the predictions list.
                          # Simple: Save keypoint x.
        preds.append(py)  # Detailed: Append computed y-coordinate to the predictions list.
                          # Simple: Save keypoint y.
    # Stack all keypoints along the last axis to form an array of keypoint coordinates.
    return np.stack(preds, axis=-1)  # Detailed: Combine all keypoint x and y coordinates into one array.
                                     # Simple: Put the keypoints together.

#############################################
# Face Detector Class using SCRFD model
#############################################

class FaceDetector:
    def __init__(self, onnx_file=None, session=None):
        """
        Detailed: Initialize the FaceDetector using an ONNX model. If a session is not provided,
                  it loads the ONNX model from the provided file path with CUDA support.
        Simple: Set up the face detector with a pre-trained ONNX model.
        """
        from onnxruntime import InferenceSession  # Detailed: Import InferenceSession for running the ONNX model.
                                                  # Simple: Lets us run the model.
        self.session = session  # Detailed: Use an existing session if provided.
                               # Simple: Use the given session if available.

        self.batched = False  # Detailed: Initialize a flag to indicate if the model supports batching.
                              # Simple: Start with batched mode off.
        if self.session is None:
            # Detailed: If no session is provided, ensure an ONNX file is given and exists, then create a new session.
            # Simple: If no session is given, load the model from file.
            assert onnx_file is not None
            assert os.path.exists(onnx_file)
            self.session = InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
        
        self.nms_thresh = 0.4  # Detailed: Set the threshold for non-maximum suppression (NMS) to filter overlapping boxes.
                              # Simple: Define a threshold to remove duplicate boxes.
        self.center_cache = {}  # Detailed: Initialize a cache to store computed anchor centers for different feature map sizes.
                               # Simple: Cache computed center points.
        self._init_vars()  # Detailed: Initialize additional variables based on the model's input and output configurations.
                          # Simple: Set up model-specific variables.

    def _init_vars(self):
        """
        Detailed: Internal function to initialize variables like input size, output names, and flags
                  for using keypoints, based on the ONNX session's model configuration.
        Simple: Set up important model parameters.
        """
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape  # Detailed: Retrieve the shape of the input tensor.
                                      # Simple: Get the input size.
        if isinstance(input_shape[2], str):
            self.input_size = None  # Detailed: If the input dimensions are dynamic (represented as strings), set input_size to None.
                                   # Simple: If size is not fixed, use None.
        else:
            self.input_size = tuple(input_shape[2:4][::-1])  # Detailed: Otherwise, set the input size using the fixed dimensions (width, height).
                                                           # Simple: Define the fixed input size.
        input_name = input_cfg.name  # Detailed: Get the name of the input tensor.
                                    # Simple: Save the input tensor's name.
        outputs = self.session.get_outputs()  # Detailed: Retrieve the model's output tensors.
                                             # Simple: Get the outputs.
        if len(outputs[0].shape) == 3:
            self.batched = True  # Detailed: If the output tensor shape indicates batching (3 dimensions), mark the model as batched.
                                # Simple: Set batched mode if the output has 3 dimensions.
        output_names = []
        for o in outputs:
            output_names.append(o.name)  # Detailed: Collect the names of all output tensors.
                                        # Simple: Save each output's name.
        self.input_name = input_name  # Detailed: Store the input tensor name.
                                     # Simple: Remember the input name.
        self.output_names = output_names  # Detailed: Store the output tensor names.
                                         # Simple: Remember the output names.
        self.use_kps = False  # Detailed: Initialize a flag indicating whether keypoints are used.
                            # Simple: Start without keypoints.
        self._num_anchors = 1  # Detailed: Initialize the number of anchors per spatial location.
                             # Simple: Default to one anchor.
        if len(outputs) == 6:
            # Detailed: Configure variables for models with 6 outputs (likely without keypoints).
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            # Detailed: Configure variables for models with 9 outputs (using keypoints).
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            # Detailed: Configure for models with 10 outputs.
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            # Detailed: Configure for models with 15 outputs (using keypoints and more feature map levels).
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, image, thresh):
        """
        Detailed: Perform a forward pass through the model using the provided image.
                  Generate scores, bounding boxes, and optionally keypoints for detected faces.
        Simple: Run the model on an image to get face scores, boxes, and keypoints.
        """
        scores_list = []  # Detailed: Initialize a list to store detection scores for each feature map level.
                         # Simple: List for face scores.
        bboxes_list = []  # Detailed: Initialize a list to store bounding boxes.
                          # Simple: List for face boxes.
        kps_list = []  # Detailed: Initialize a list to store keypoints if available.
                     # Simple: List for face keypoints.
        input_size = tuple(image.shape[0:2][::-1])  # Detailed: Get the input size as (width, height) from the image.
                                                   # Simple: Determine image size.
        # Create a blob from the image with specified preprocessing parameters
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128, input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        # Run inference on the blob using the ONNX session
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        fmc = self.fmc  # Detailed: Number of output feature maps for classification/regression.
        input_width = blob.shape[3]  # Detailed: Get the width of the blob.
        input_height = blob.shape[2]  # Detailed: Get the height of the blob.

        # Process each feature map level
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]  # Detailed: If batched, take the first element of the output.
                boxes = net_outs[idx + fmc][0]  # Detailed: Get the bounding box predictions.
                boxes = boxes * stride  # Detailed: Scale the boxes by the stride.
                if self.use_kps:
                    points = net_outs[idx + fmc * 2][0] * stride  # Detailed: Scale keypoints if used.
            else:
                scores = net_outs[idx]  # Detailed: If not batched, take outputs directly.
                boxes = net_outs[idx + fmc]
                boxes = boxes * stride
                if self.use_kps:
                    points = net_outs[idx + fmc * 2] * stride

            height = input_height // stride  # Detailed: Calculate the feature map height.
            width = input_width // stride  # Detailed: Calculate the feature map width.
            key = (height, width, stride)  # Detailed: Create a key for caching anchor centers.
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]  # Detailed: Retrieve cached anchor centers if available.
            else:
                # Detailed: Create a grid of anchor centers based on the feature map dimensions and stride.
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    # Duplicate centers if multiple anchors per location are used.
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers  # Detailed: Cache the computed anchor centers.
            # Find positions with scores above the threshold
            pos_index = np.where(scores >= thresh)[0]
            # Convert distances to bounding boxes using the anchor centers
            bboxes = distance2bbox(anchor_centers, boxes)
            pos_scores = scores[pos_index]
            pos_bboxes = bboxes[pos_index]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, points)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kps_list.append(kpss[pos_index])
        # Return lists of scores, bounding boxes, and keypoints (if available)
        return scores_list, bboxes_list, kps_list

    def detect(self, image, thresh=0.5, input_size=None, max_num=0, metric='default'):
        """
        Detailed: Detect faces in an image by resizing the image to the model's input size,
                  running forward pass, applying non-maximum suppression (NMS), and optionally limiting
                  the number of detections.
        Simple: Detect faces in the image, filter overlapping boxes, and return results.
        """
        # Ensure the input size is defined (either provided or from the model)
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        # Resize image while maintaining aspect ratio
        im_ratio = float(image.shape[0]) / image.shape[1]  # Detailed: Compute the aspect ratio of the input image.
        model_ratio = float(input_size[1]) / input_size[0]  # Detailed: Compute the aspect ratio of the model's expected input.
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        # Place the resized image into a blank image of the required input size
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Run forward pass to get scores, boxes, and keypoints
        scores_list, bboxes_list, kps_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]  # Detailed: Sort detections by score in descending order.
        bboxes = np.vstack(bboxes_list) / det_scale  # Detailed: Scale bounding boxes back to original image size.
        if self.use_kps:
            kpss = np.vstack(kps_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)  # Detailed: Apply non-maximum suppression to remove redundant boxes.
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        # Optionally limit the number of detections based on a metric
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # Extra weight for centering the detection.
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss  # Detailed: Return the final detections and keypoints.
                             # Simple: Output the face boxes and keypoints.

    def nms(self, dets):
        """
        Detailed: Apply non-maximum suppression (NMS) to eliminate overlapping detections.
                  It iteratively selects the highest-scoring detection and removes others with high overlap.
        Simple: Remove duplicate face boxes that overlap too much.
        """
        thresh = self.nms_thresh  # Detailed: Use the predefined NMS threshold.
        x1 = dets[:, 0]  # Detailed: Extract the x-coordinate of the top-left corner of each box.
        y1 = dets[:, 1]  # Detailed: Extract the y-coordinate of the top-left corner.
        x2 = dets[:, 2]  # Detailed: Extract the x-coordinate of the bottom-right corner.
        y2 = dets[:, 3]  # Detailed: Extract the y-coordinate of the bottom-right corner.
        scores = dets[:, 4]  # Detailed: Extract the confidence scores for each detection.

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # Detailed: Compute the area of each detection box.
        order = scores.argsort()[::-1]  # Detailed: Sort the detections by descending score.

        keep = []  # Detailed: Initialize a list to store indices of boxes to keep.
        while order.size > 0:
            i = order[0]  # Detailed: Select the detection with the highest remaining score.
            keep.append(i)  # Detailed: Keep this detection.
            # Compute the intersection coordinates with the remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)  # Detailed: Compute the width of the intersection area.
            h = np.maximum(0.0, yy2 - yy1 + 1)  # Detailed: Compute the height of the intersection area.
            inter = w * h  # Detailed: Calculate the intersection area.
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  # Detailed: Compute the Intersection over Union (IoU).

            index = np.where(ovr <= thresh)[0]  # Detailed: Find indices where IoU is less than the threshold.
            order = order[index + 1]  # Detailed: Update the order list by keeping only those detections.
        return keep  # Detailed: Return the indices of detections that survived NMS.
                   # Simple: Output the indices of good boxes.
