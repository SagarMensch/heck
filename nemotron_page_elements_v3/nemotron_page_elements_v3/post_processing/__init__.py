# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing utilities for page element predictions.

This module provides utilities for advanced post-processing of page element
detection results, including box expansion, matching with titles, and
weighted box fusion.
"""

# Import from page_elt_pp
from .page_elt_pp import (
    expand_boxes,
    merge_boxes,
    bb_iou_array,
    match_with_title,
    match_boxes_with_title,
)

# Import from text_pp
from .text_pp import (
    get_overlaps,
    get_distances,
    find_titles,
    postprocess_included,
)

# Import from wbf
from .wbf import (
    weighted_boxes_fusion,
    prefilter_boxes,
    merge_labels,
    get_weighted_box,
    get_biggest_box,
    find_matching_box_fast,
)

__all__ = [
    # page_elt_pp
    "expand_boxes",
    "merge_boxes",
    "bb_iou_array",
    "match_with_title",
    "match_boxes_with_title",
    # text_pp
    "get_overlaps",
    "get_distances",
    "find_titles",
    "postprocess_included",
    # wbf
    "weighted_boxes_fusion",
    "prefilter_boxes",
    "merge_labels",
    "get_weighted_box",
    "get_biggest_box",
    "find_matching_box_fast",
]


