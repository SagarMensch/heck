# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple


COLORS = [
    "#003EFF",
    "#FF8F00",
    "#079700",
    "#A123FF",
    "#87CEEB",
    "#FF5733",
    "#C70039",
    "#900C3F",
    "#581845",
    "#11998E",
]


def reformat_for_plotting(
    boxes: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    scores: npt.NDArray[np.float64],
    shape: Tuple[int, int],
    num_classes: int,
) -> Tuple[List[npt.NDArray[np.int_]], List[npt.NDArray[np.float64]]]:
    """
    Reformat YOLOX predictions for plotting.
    - Unnormalizes boxes to original image size.
    - Reformats boxes to [xmin, ymin, width, height].
    - Converts to list of boxes and scores per class.

    Args:
        boxes (np.ndarray [N, 4]): Array of bounding boxes in format [xmin, ymin, xmax, ymax].
        labels (np.ndarray [N]): Array of labels.
        scores (np.ndarray [N]): Array of confidence scores.
        shape (tuple [2]): Shape of the image (height, width).
        num_classes (int): Number of classes.

    Returns:
        list[np.ndarray[N]]: List of box bounding boxes per class.
        list[np.ndarray[N]]: List of confidence scores per class.
    """
    boxes_plot = boxes.copy()
    boxes_plot[:, [0, 2]] *= shape[1]
    boxes_plot[:, [1, 3]] *= shape[0]
    boxes_plot = boxes_plot.astype(int)
    boxes_plot[:, 2] -= boxes_plot[:, 0]
    boxes_plot[:, 3] -= boxes_plot[:, 1]
    boxes_plot = [boxes_plot[labels == c] for c in range(num_classes)]
    confs = [scores[labels == c] for c in range(num_classes)]
    return boxes_plot, confs


def plot_sample(
    img: npt.NDArray[np.uint8],
    boxes_list: List[npt.NDArray[np.int_]],
    confs_list: List[npt.NDArray[np.float64]],
    labels: List[str],
    show_text: bool = True,
) -> None:
    """
    Plots an image with bounding boxes.
    Coordinates are expected in format [x_min, y_min, width, height].

    Args:
        img (numpy.ndarray): The input image to be plotted.
        boxes_list (list[np.ndarray]): List of box bounding boxes per class.
        confs_list (list[np.ndarray]): List of confidence scores per class.
        labels (list): List of class labels.
        show_text (bool, optional): Whether to show the text. Defaults to True.
    """
    plt.imshow(img, cmap="gray")
    plt.axis(False)

    for boxes, confs, col, l in zip(boxes_list, confs_list, COLORS, labels):
        for box_idx, box in enumerate(boxes):
            # Better display around boundaries
            h, w, _ = img.shape
            box = np.copy(box)
            box[:2] = np.clip(box[:2], 2, max(h, w))
            box[2] = min(box[2], w - 2 - box[0])
            box[3] = min(box[3], h - 2 - box[1])

            rect = Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                linewidth=2,
                facecolor="none",
                edgecolor=col,
            )
            plt.gca().add_patch(rect)

            # Add class and index label with proper alignment
            if show_text:
                plt.text(
                    box[0], box[1],
                    f"{l}_{box_idx}   conf={confs[box_idx]:.3f}",
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=col, alpha=1, edgecolor=col, pad=0, linewidth=2),
                    verticalalignment='bottom',
                    horizontalalignment='left'
                )


def postprocess_preds_page_element(
    preds: Dict[str, npt.NDArray],
    thresholds_per_class: Dict[str, float],
    class_labels: List[str],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """
    Post process predictions for the page element task.
    - Applies thresholding

    Args:
        preds (dict): Predictions. Keys are "scores", "boxes", "labels".
        thresholds_per_class (dict): Thresholds per class.
        class_labels (list): List of class labels.

    Returns:
        numpy.ndarray [N x 4]: Array of bounding boxes.
        numpy.ndarray [N]: Array of labels.
        numpy.ndarray [N]: Array of scores.
    """
    boxes = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    # Threshold per class
    thresholds = np.array(
        [thresholds_per_class[class_labels[int(x)]] for x in labels]
    )
    boxes = boxes[scores > thresholds]
    labels = labels[scores > thresholds]
    scores = scores[scores > thresholds]

    return boxes, labels, scores
