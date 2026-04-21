# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional


def expand_boxes(
    boxes: npt.NDArray[np.float64],
    r_x: Tuple[float, float] = (1, 1),
    r_y: Tuple[float, float] = (1, 1),
    size_agnostic: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Expands bounding boxes by a specified ratio.
    Expected box format is normalized [x_min, y_min, x_max, y_max].

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        r_x (tuple, optional): Left, right expansion ratios. Defaults to (1, 1) (no expansion).
        r_y (tuple, optional): Up, down expansion ratios. Defaults to (1, 1) (no expansion).
        size_agnostic (bool, optional): Expand independently of the box shape. Defaults to True.

    Returns:
        numpy.ndarray: Adjusted bounding boxes clipped to the [0, 1] range.
    """
    old_boxes = boxes.copy()

    if not size_agnostic:
        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
    else:
        h, w = 1, 1

    boxes[:, 0] -= w * (r_x[0] - 1)  # left
    boxes[:, 2] += w * (r_x[1] - 1)  # right
    boxes[:, 1] -= h * (r_y[0] - 1)  # up
    boxes[:, 3] += h * (r_y[1] - 1)  # down

    boxes = np.clip(boxes, 0, 1)

    # Enforce non-overlapping boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = bb_iou_array(boxes[i][None], boxes[j])[0]
            old_iou = bb_iou_array(old_boxes[i][None], old_boxes[j])[0]
            # print(iou, old_iou)
            if iou > 0.05 and old_iou < 0.1:
                if boxes[i, 1] < boxes[j, 1]:  # i above j
                    boxes[j, 1] = min(old_boxes[j, 1], boxes[i, 3])
                    if old_iou > 0:
                        boxes[i, 3] = max(old_boxes[i, 3], boxes[j, 1])
                else:
                    boxes[i, 1] = min(old_boxes[i, 1], boxes[j, 3])
                    if old_iou > 0:
                        boxes[j, 3] = max(old_boxes[j, 3], boxes[i, 1])

    return boxes


def merge_boxes(
    b1: npt.NDArray[np.float64], b2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Merges two bounding boxes into a single box that encompasses both.

    Args:
        b1 (numpy.ndarray): First bounding box [x_min, y_min, x_max, y_max].
        b2 (numpy.ndarray): Second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        numpy.ndarray: A single bounding box that covers both input boxes.
    """
    b = b1.copy()
    b[0] = min(b1[0], b2[0])
    b[1] = min(b1[1], b2[1])
    b[2] = max(b1[2], b2[2])
    b[3] = max(b1[3], b2[3])
    return b


def bb_iou_array(
    boxes: npt.NDArray[np.float64], new_box: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculates the Intersection over Union (IoU) between a box and an array of boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        new_box (numpy.ndarray): A single bounding box [x_min, y_min, x_max, y_max].

    Returns:
        numpy.ndarray: Array of IoU values between the new_box and each box in the array.
    """
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


def match_with_title(
    box: npt.NDArray[np.float64],
    title_boxes: npt.NDArray[np.float64],
    match_dist: float = 0.1,
    delta: float = 1.,
    already_matched: List[int] = [],
) -> Tuple[Optional[npt.NDArray[np.float64]], Optional[List[int]]]:
    """
    Matches a bounding box with a title bounding box based on IoU or proximity.

    Args:
        box (numpy.ndarray): Bounding box to match with title [x_min, y_min, x_max, y_max].
        title_boxes (numpy.ndarray): Array of title bounding boxes with shape (N, 4).
        match_dist (float, optional): Maximum distance for matching. Defaults to 0.1.
        delta (float, optional): Multiplier for matching several titles. Defaults to 1..
        already_matched (list, optional): List of already matched title indices. Defaults to [].

    Returns:
        tuple or None: If matched, returns a tuple of (merged_bbox, updated_title_boxes).
                       If no match is found, returns None, None.
    """
    if not len(title_boxes):
        return None, None

    dist_above = np.abs(title_boxes[:, 3] - box[1])
    dist_below = np.abs(box[3] - title_boxes[:, 1])

    dist_left = np.abs(title_boxes[:, 0] - box[0])
    dist_center = np.abs(title_boxes[:, 0] + title_boxes[:, 2] - box[0] - box[2]) / 2

    dists = np.min([dist_above, dist_below], 0)
    dists += np.min([dist_left, dist_center], 0) / 2

    ious = bb_iou_array(title_boxes, box)
    dists = np.where(ious > 0, min(match_dist - 0.01, np.min(dists)) / delta, dists)

    if len(already_matched):
        dists[already_matched] = match_dist * 10  # Remove already matched titles

    matches = None
    if np.min(dists) <= match_dist:
        matches = np.where(
            dists <= min(match_dist, np.min(dists) * delta)
        )[0]

    if matches is not None:
        new_bbox = box
        for match in matches:
            new_bbox = merge_boxes(new_bbox, title_boxes[match])
        return new_bbox, list(matches)
    else:
        return None, None


def match_boxes_with_title(
    boxes: npt.NDArray[np.float64],
    confs: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    classes: List[str],
    to_match_labels: List[str] = ["chart"],
    remove_matched_titles: bool = False,
    match_dist: float = 0.1,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    List[int],
]:
    """
    Matches charts with title.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        confs (numpy.ndarray): Array of confidence scores with shape (N,).
        labels (numpy.ndarray): Array of labels with shape (N,).
        classes (list): List of class names.
        to_match_labels (list): List of class names to match with titles.
        remove_matched_titles (bool): Whether to remove matched titles from the boxes.

    Returns:
        boxes (numpy.ndarray): Array of bounding boxes with shape (M, 4).
        confs (numpy.ndarray): Array of confidence scores with shape (M,).
        labels (numpy.ndarray): Array of labels with shape (M,).
        found_title (list): List of indices of matched titles.
        no_found_title (list): List of indices of unmatched titles.
        match_dist (float, optional): Maximum distance for matching. Defaults to 0.1.
    """
    # Put titles at the end
    title_ids = np.where(labels == classes.index("title"))[0]
    order = np.concatenate([np.delete(np.arange(len(boxes)), title_ids), title_ids])
    boxes = boxes[order]
    confs = confs[order]
    labels = labels[order]

    # Ids
    title_ids = np.where(labels == classes.index("title"))[0]
    to_match = np.where(np.isin(labels, [classes.index(c) for c in to_match_labels]))[0]

    # Matching
    found_title, already_matched = [], []
    for i in range(len(boxes)):
        if i not in to_match:
            continue
        merged_box, matched_title_ids = match_with_title(
            boxes[i],
            boxes[title_ids],
            already_matched=already_matched,
            match_dist=match_dist,
        )
        if matched_title_ids is not None:
            # print(f'Merged {classes[int(labels[i])]} at idx #{i} with title {matched_title_ids[-1]}')  # noqa
            boxes[i] = merged_box
            already_matched += matched_title_ids
            found_title.append(i)

    if remove_matched_titles and len(already_matched):
        boxes = np.delete(boxes, title_ids[already_matched], axis=0)
        confs = np.delete(confs, title_ids[already_matched], axis=0)
        labels = np.delete(labels, title_ids[already_matched], axis=0)

    return boxes, confs, labels, found_title
