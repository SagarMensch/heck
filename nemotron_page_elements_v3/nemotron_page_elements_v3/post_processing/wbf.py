# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from:
# https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py

import warnings
from typing import Dict, List, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt


def prefilter_boxes(
    boxes: List[npt.NDArray[np.float64]],
    scores: List[npt.NDArray[np.float64]],
    labels: List[npt.NDArray[np.int_]],
    weights: List[float],
    thr: float,
    class_agnostic: bool = False,
) -> Dict[Union[str, int], npt.NDArray[np.float64]]:
    """
    Reformats and filters boxes.
    Output is a dict of boxes to merge separately.

    Args:
        boxes (list[np array[n x 4]]): List of boxes. One list per model.
        scores (list[np array[n]]): List of confidences.
        labels (list[np array[n]]): List of labels.
        weights (list): Model weights.
        thr (float): Confidence threshold
        class_agnostic (bool, optional): Merge boxes from different classes. Defaults to False.

    Returns:
        dict[np array [? x 8]]: Filtered boxes.
    """
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):
        assert len(boxes[t]) == len(scores[t]), "len(boxes) != len(scores)"
        assert len(boxes[t]) == len(labels[t]), "len(boxes) != len(labels)"

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swap them.")
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn("Y2 < Y1 value in box. Swap them.")
                y1, y2 = y2, y1

            array = np.array([x1, x2, y1, y2])
            if array.min() < 0 or array.max() > 1:
                warnings.warn("Coordinates outside [0, 1]")
                array = np.clip(array, 0, 1)
                x1, x2, y1, y2 = array

            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]

            label_k = "*" if class_agnostic else label
            if label_k not in new_boxes:
                new_boxes[label_k] = []
            new_boxes[label_k].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def merge_labels(
    labels: npt.NDArray[np.int_], confs: npt.NDArray[np.float64]
) -> int:
    """
    Custom function for merging labels.
    If all labels are the same, return the unique value.
    Else, return the label of the most confident non-title (class 2) box.

    Args:
        labels (np array [n]): Labels.
        confs (np array [n]): Confidence.

    Returns:
        int: Label.
    """
    if len(np.unique(labels)) == 1:
        return labels[0]
    else:  # Most confident and not a title
        confs = confs[confs != 2]
        labels = labels[labels != 2]
        return labels[np.argmax(confs)]


def get_weighted_box(
    boxes: npt.NDArray[np.float64], conf_type: Literal["avg", "max"] = "avg"
) -> npt.NDArray[np.float64]:
    """
    Merges boxes by using the weighted fusion.

    Args:
        boxes (np array [n x 8]): Boxes to merge.
        conf_type (str, optional): Confidence merging type. Defaults to "avg".

    Returns:
        np array [8]: Merged box.
    """
    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]

    box[0] = merge_labels(
        np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
    )

    box[1] = np.max(conf_list) if conf_type == "max" else np.mean(conf_list)
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def get_biggest_box(
    boxes: npt.NDArray[np.float64], conf_type: Literal["avg", "max"] = "avg"
) -> npt.NDArray[np.float64]:
    """
    Merges boxes by using the biggest box.

    Args:
        boxes (np array [n x 8]): Boxes to merge.
        conf_type (str, optional): Confidence merging type. Defaults to "avg".

    Returns:
        np array [8]: Merged box.
    """
    box = np.zeros(8, dtype=np.float32)
    box[4:] = boxes[0][4:]
    conf_list = []
    w = 0
    for b in boxes:
        box[4] = min(box[4], b[4])
        box[5] = min(box[5], b[5])
        box[6] = max(box[6], b[6])
        box[7] = max(box[7], b[7])
        conf_list.append(b[1])
        w += b[2]

    box[0] = merge_labels(
        np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])
    )
    #     print(box[0], np.array([b[0] for b in boxes]))

    box[1] = np.max(conf_list) if conf_type == "max" else np.mean(conf_list)
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    return box


def find_matching_box_fast(
    boxes_list: npt.NDArray[np.float64],
    new_box: npt.NDArray[np.float64],
    match_iou: float,
) -> Tuple[int, float]:
    """
    Reimplementation of find_matching_box with numpy instead of loops.
    Gives significant speed up for larger arrays (~100x).
    This was previously the bottleneck since the function is called for every entry in the array.

    Args:
        boxes_list (np.ndarray): Array of boxes with shape (N, 8).
        new_box (np.ndarray): New box to match with shape (8,).
        match_iou (float): IoU threshold for matching.

    Returns:
        Tuple[int, float]: Index of best matching box (-1 if no match) and IoU value.
    """

    def bb_iou_array(
        boxes: npt.NDArray[np.float64], new_box: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    ious = bb_iou_array(boxes_list[:, 4:], new_box[4:])
    # ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
    boxes_list: List[npt.NDArray[np.float64]],
    labels_list: List[npt.NDArray[np.int_]],
    scores_list: List[npt.NDArray[np.float64]],
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0,
    conf_type: Literal["avg", "max"] = "avg",
    merge_type: Literal["weighted", "biggest"] = "weighted",
    class_agnostic: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """
    Custom WBF implementation that supports a class_agnostic mode and a biggest box fusion.
    Boxes are expected to be in normalized (x0, y0, x1, y1) format.

    Args:
        boxes_list (list[np.ndarray[n x 4]]): List of boxes. One list per model.
        labels_list (list[np.ndarray[n]]): List of labels.
        scores_list (list[np.ndarray[n]]): List of confidences.
        iou_thr (float, optional): IoU threshold for matching. Defaults to 0.55.
        skip_box_thr (float, optional): Exclude boxes with score < skip_box_thr. Defaults to 0.0.
        conf_type (str, optional): Confidence merging type ("avg" or "max"). Defaults to "avg".
        merge_type (str, optional): Merge type ("weighted" or "biggest"). Defaults to "weighted".
        class_agnostic (bool, optional): Merge boxes from different classes. Defaults to False.

    Returns:
        numpy.ndarray [N x 4]: Array of bounding boxes.
        numpy.ndarray [N]: Array of labels.
        numpy.ndarray [N]: Array of scores.
    """
    weights = np.ones(len(boxes_list))

    assert conf_type in ["avg", "max"], 'Conf type must be "avg" or "max"'
    assert merge_type in ["weighted", "biggest"], 'Conf type must be "weighted" or "biggest"'

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        weights,
        skip_box_thr,
        class_agnostic=class_agnostic,
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        clusters = []

        # Clusterize boxes
        for j in range(len(boxes)):
            ids = [i for i in range(len(boxes)) if i != j]
            index, best_iou = find_matching_box_fast(boxes[ids], boxes[j], iou_thr)

            if index != -1:
                index = ids[index]
                cluster_idx = [
                    clust_idx
                    for clust_idx, clust in enumerate(clusters)
                    if (j in clust or index in clust)
                ]
                if len(cluster_idx):
                    cluster_idx = cluster_idx[0]
                    clusters[cluster_idx] = list(
                        set(clusters[cluster_idx] + [index, j])
                    )
                else:
                    clusters.append([index, j])
            else:
                clusters.append([j])

        for j, c in enumerate(clusters):
            if merge_type == "weighted":
                weighted_box = get_weighted_box(boxes[c], conf_type)
            elif merge_type == "biggest":
                weighted_box = get_biggest_box(boxes[c], conf_type)

            if conf_type == "max":
                weighted_box[1] = weighted_box[1] / weights.max()
            else:  # avg
                weighted_box[1] = weighted_box[1] * len(c) / weights.sum()
            overall_boxes.append(weighted_box)

    overall_boxes = np.array(overall_boxes)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, labels, scores
