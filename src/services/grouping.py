from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any

import cv2
import numpy as np
from PIL import Image


GROUP_PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#F7B32B",
    "#3A86FF",
    "#6A4C93",
    "#2A9D8F",
    "#E76F51",
    "#457B9D",
    "#8D99AE",
    "#588157",
    "#EF476F",
    "#118AB2",
]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def extract_visual_embedding(image: Image.Image, bbox: list[int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    crop = image.crop((x_min, y_min, x_max, y_max)).convert("RGB")
    crop_np = np.array(crop)
    if crop_np.size == 0:
        return np.zeros(87, dtype=np.float32)

    resized = cv2.resize(crop_np, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()

    hist = np.concatenate([hist_h, hist_s, hist_v, hist_gray]).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-8

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    height = max(1, y_max - y_min)
    width = max(1, x_max - x_min)
    aspect = np.array([width / max(height, 1)], dtype=np.float32)
    area = np.array([sqrt(float(width * height)) / 512.0], dtype=np.float32)

    return np.concatenate([hist, edge_density, aspect, area]).astype(np.float32)


def cluster_embeddings(
    embeddings: list[np.ndarray], threshold: float = 0.92
) -> list[int]:
    if not embeddings:
        return []

    groups: list[dict[str, Any]] = []
    assignments: list[int] = []

    for embedding in embeddings:
        best_index = -1
        best_score = -1.0

        for index, group in enumerate(groups):
            score = cosine_similarity(embedding, group["centroid"])
            if score > best_score:
                best_score = score
                best_index = index

        if best_index == -1 or best_score < threshold:
            groups.append({"centroid": embedding.copy(), "members": 1})
            assignments.append(len(groups) - 1)
            continue

        group = groups[best_index]
        members = int(group["members"]) + 1
        group["centroid"] = (
            (group["centroid"] * group["members"]) + embedding
        ) / members
        group["members"] = members
        assignments.append(best_index)

    return assignments


class GroupingService:
    def assign_groups(
        self, image: Image.Image, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not detections:
            return detections

        embeddings = [
            extract_visual_embedding(image, item["bbox"]) for item in detections
        ]
        assignments = cluster_embeddings(embeddings)

        group_order: dict[int, str] = {}
        color_lookup: dict[str, str] = {}

        for cluster_index in assignments:
            if cluster_index not in group_order:
                group_id = f"group_{len(group_order) + 1:03d}"
                group_order[cluster_index] = group_id
                color_lookup[group_id] = GROUP_PALETTE[
                    (len(group_order) - 1) % len(GROUP_PALETTE)
                ]

        for detection, cluster_index in zip(detections, assignments):
            group_id = group_order[cluster_index]
            detection["group_id"] = group_id
            detection["group_color"] = color_lookup[group_id]

        return detections

    def summarize_groups(self, detections: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for detection in detections:
            counts[detection["group_id"]] += 1
        return dict(counts)
