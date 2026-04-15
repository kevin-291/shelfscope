from __future__ import annotations

import numpy as np
from PIL import Image

from src.services.grouping import (
    GroupingService,
    cluster_embeddings,
    extract_visual_embedding,
)


def test_extract_visual_embedding_returns_expected_shape():
    image = Image.new("RGB", (100, 100), color="white")

    embedding = extract_visual_embedding(image, [10, 10, 70, 80])

    assert embedding.shape == (75,)
    assert embedding.dtype == np.float32


def test_cluster_embeddings_groups_similar_vectors():
    base = np.ones(10, dtype=np.float32)
    similar = base * 0.99
    different = np.zeros(10, dtype=np.float32)
    different[0] = 1.0

    assignments = cluster_embeddings([base, similar, different], threshold=0.95)

    assert assignments[0] == assignments[1]
    assert assignments[2] != assignments[0]


def test_grouping_service_assigns_group_fields():
    image = Image.new("RGB", (120, 120), color="white")
    detections = [
        {
            "detection_id": "det_0001",
            "bbox": [5, 5, 40, 60],
            "score": 0.9,
            "class_name": "product",
        },
        {
            "detection_id": "det_0002",
            "bbox": [50, 5, 85, 60],
            "score": 0.85,
            "class_name": "product",
        },
    ]

    grouped = GroupingService().assign_groups(image, detections)

    assert len(grouped) == 2
    assert all("group_id" in item for item in grouped)
    assert all("group_color" in item for item in grouped)


def test_grouping_summary_counts_groups():
    detections = [
        {"group_id": "group_001"},
        {"group_id": "group_001"},
        {"group_id": "group_002"},
    ]

    summary = GroupingService().summarize_groups(detections)

    assert summary == {"group_001": 2, "group_002": 1}
