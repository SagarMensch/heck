# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Literal


def get_overlaps(
    boxes: npt.NDArray[np.float64],
    other_boxes: npt.NDArray[np.float64],
    normalize: Literal["box_only", "all"] = "box_only",
) -> npt.NDArray[np.float64]:
    """
    Checks if a box overlaps with any other box.
    Boxes are expeceted in format (x0, y0, x1, y1)

    Args:
        boxes (np array [4] or [n x 4]): Boxes.
        other_boxes (np array [m x 4]): Other boxes.

    Returns:
        np array [n x m]: Overlaps.
    """
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    x0, y0, x1, y1 = (
        boxes[:, 0][:, None],
        boxes[:, 1][:, None],
        boxes[:, 2][:, None],
        boxes[:, 3][:, None],
    )
    areas = (y1 - y0) * (x1 - x0)

    x0_other, y0_other, x1_other, y1_other = (
        other_boxes[:, 0][None, :],
        other_boxes[:, 1][None, :],
        other_boxes[:, 2][None, :],
        other_boxes[:, 3][None, :],
    )
    areas_other = (y1_other - y0_other) * (x1_other - x0_other)

    # Intersection
    inter_y0 = np.maximum(y0, y0_other)
    inter_y1 = np.minimum(y1, y1_other)
    inter_x0 = np.maximum(x0, x0_other)
    inter_x1 = np.minimum(x1, x1_other)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    # Overlap
    if normalize == "box_only":  # Only consider box included in other box
        overlaps = inter_area / areas
    elif (
        normalize == "all"
    ):  # Consider box included in other box and other box included in box
        overlaps = inter_area / np.minimum(areas, areas_other[:, None])
    else:
        raise ValueError(f"Invalid normalization: {normalize}")
    return overlaps


def get_distances(
    title_boxes: npt.NDArray[np.float64], other_boxes: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Computes the distances between title and table/chart boxes.
    Distance is computed as the sum of the vertical and horizontal distances.
    Horizontal distance uses min(boxes center dist, boxes left dist).
    Vertical distance uses min(top_title to bottom_other dists, bottom_title to top_other dists).

    Args:
        title_boxes (np array [n_titles x 4]): Title boxes.
        other_boxes (np array [n_other x 4]): Other boxes.

    Returns:
        np array [n_titles x n_other]: Distances between titles and other boxes.
    """
    x0_title, xc_title, y0_title, y1_title = (
        title_boxes[:, 0],
        (title_boxes[:, 0] + title_boxes[:, 2]) / 2,
        title_boxes[:, 1],
        title_boxes[:, 3],
    )
    x0_other, xc_other, y0_other, y1_other = (
        other_boxes[:, 0],
        (other_boxes[:, 0] + other_boxes[:, 2]) / 2,
        other_boxes[:, 1],
        other_boxes[:, 3],
    )

    x_dists = np.min(
        [
            np.abs(
                xc_title[:, None] - xc_other[None, :]
            ),  # Title center to other center
            np.abs(x0_title[:, None] - x0_other[None, :]),  # Title left to other left
        ],
        axis=0,
    )

    y_dists = np.min(
        [
            np.abs(y1_title[:, None] - y0_other[None, :]),  # Title above other
            np.abs(y0_title[:, None] - y1_other[None, :]),  # Title below other
        ],
        axis=0,
    )

    dists = y_dists + x_dists / 2
    return dists


def find_titles(
    title_boxes: npt.NDArray[np.float64],
    table_boxes: npt.NDArray[np.float64],
    chart_boxes: npt.NDArray[np.float64],
    max_dist: float = 0.1,
) -> Dict[int, Tuple[str, int]]:
    """
    Associates titles to tables and charts.

    Args:
        title_boxes (np array [n_titles x 4]): Title boxes.
        table_boxes (np array [n_tables x 4]): Table boxes.
        chart_boxes (np array [n_charts x 4]): Chart boxes.
        max_dist (float, optional): Maximum distance between title and table/chart. Defaults to 0.1.

    Returns:
        dict: Dictionary of assigned titles.
            - Keys are the indices of the titles,
            - Values are tuples of:
                - str: Whether the title is assigned to a "chart" or "table"
                - int: index of the assigned table/chart
    """
    if not len(title_boxes) or not (len(table_boxes) or len(chart_boxes)):
        return {}

    # print(title_boxes.shape, table_boxes.shape, chart_boxes.shape)

    # Get distances
    chart_distances = np.ones((len(title_boxes), 0))
    if len(chart_boxes):
        chart_distances = get_distances(title_boxes, chart_boxes)
        chart_overlaps = get_overlaps(title_boxes, chart_boxes, normalize="box_only")
        # print(chart_overlaps, "chart_overlaps", chart_overlaps.shape)
        # print(chart_distances, "chart_distances", chart_distances.shape)
        chart_distances = np.where(chart_overlaps > 0.25, 0, chart_distances)

    # print(chart_distances)

    table_distances = np.ones((len(title_boxes), 0))
    if len(table_boxes):
        table_distances = get_distances(title_boxes, table_boxes)
        if len(chart_boxes):  # Penalize table titles that are inside charts
            table_distances = np.where(
                chart_overlaps.max(1, keepdims=True) > 0.25,
                table_distances * 10,
                table_distances,
            )

    # print(table_distances, "table_distances")

    # Assign to tables
    assigned_titles = {}
    for i, table in enumerate(table_boxes):
        best_match = np.argmin(table_distances[:, i])
        if table_distances[best_match, i] < max_dist:
            assigned_titles[best_match] = ("table", i)
            table_distances[best_match] = np.inf
            chart_distances[best_match] = np.inf

    # Assign to charts
    for i, chart in enumerate(chart_boxes):
        best_match = np.argmin(chart_distances[:, i])
        if chart_distances[best_match, i] < max_dist:
            assigned_titles[best_match] = ("chart", i)
            chart_distances[best_match] = np.inf

    return assigned_titles


def postprocess_included(
    boxes: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    confs: npt.NDArray[np.float64],
    class_: str = "title",
    classes: List[str] = ["table", "chart", "title", "infographic"],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """
    Post process title predictions.
    - Remove titles that are included in other boxes

    Args:
        boxes (numpy.ndarray [N, 4]): Array of bounding boxes.
        labels (numpy.ndarray [N]): Array of labels.
        confs (numpy.ndarray [N]): Array of confidences.
        class_ (str, optional): Class to postprocess. Defaults to "title".
        classes (list, optional): Classes. Defaults to ["table", "chart", "title", "infographic"].

    Returns:
        boxes (numpy.ndarray): Array of bounding boxes.
        labels (numpy.ndarray): Array of labels.
        confs (numpy.ndarray): Array of confidences.
    """
    boxes_to_pp = boxes[labels == classes.index(class_)]
    confs_to_pp = confs[labels == classes.index(class_)]

    order = np.argsort(confs_to_pp)  # least to most confident for NMS
    boxes_to_pp, confs_to_pp = boxes_to_pp[order], confs_to_pp[order]

    if len(boxes_to_pp) == 0:
        return boxes, labels, confs

    # other_boxes = boxes[labels != classes.index("title")]

    inclusion_classes = ["table", "infographic", "chart"]
    if class_ in ["header_footer", "title"]:
        inclusion_classes.append("text")

    other_boxes = boxes[np.isin(labels, [classes.index(c) for c in inclusion_classes])]

    # Remove boxes included in other_boxes
    kept_boxes, kept_confs = [], []
    for i, b in enumerate(boxes_to_pp):
        # # Inclusion NMS
        # if i < len(titles) - 1:
        #     overlaps_titles = get_overlaps(t, titles[i + 1:], normalize="all")
        #     if overlaps_titles.max() > 0.9:
        #         continue

        # print(t)
        # print(other_boxes)
        if len(other_boxes) > 0:
            overlaps = get_overlaps(b, other_boxes, normalize="box_only")
            if overlaps.max() > 0.9:
                continue

        kept_boxes.append(b)
        kept_confs.append(confs_to_pp[i])

    # Aggregate
    kept_boxes = np.stack(kept_boxes) if len(kept_boxes) else np.empty((0, 4))
    kept_confs = np.stack(kept_confs) if len(kept_confs) else np.empty(0)

    boxes_pp = np.concatenate([boxes[labels != classes.index(class_)], kept_boxes])
    confs_pp = np.concatenate([confs[labels != classes.index(class_)], kept_confs])
    labels_pp = np.concatenate(
        [
            labels[labels != classes.index(class_)],
            np.ones(len(kept_boxes)) * classes.index(class_),
        ]
    )

    return boxes_pp, labels_pp, confs_pp
