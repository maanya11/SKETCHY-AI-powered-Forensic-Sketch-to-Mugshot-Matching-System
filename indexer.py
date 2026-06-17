import os, json
import numpy as np
import cv2
import faiss
from .config import (INDEX_DIR, ARC_INDEX_PATH, ARC_DB_PATH, CLIP_DB_PATH, META_MAP_PATH,
                     HNSW_M, EF_CONSTRUCTION)
from .db import suspects
from .embeddings import Models, l2_normalize_np, photo_for_clip

def build_index(models: Models) -> int:
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = list(suspects.find({}, {
        "suspect_id": 1, "case_id": 1, "name": 1, "notes": 1, "tags": 1, "photo_path": 1
    }))

    ARC_vecs = []
    CLIP_vecs = []
    meta_map = {}

    for doc in docs:
        photo_path = doc.get("photo_path")
        if not photo_path or not os.path.exists(photo_path):
            continue

        bgr = cv2.imread(photo_path)
        if bgr is None:
            continue

        arc = models.arcface(bgr)
        if arc is None:
            continue

        # adapted CLIP feature for photos (trained on Kaggle)
        clip_adapt = models.adapted_feat(photo_for_clip(bgr))

        ARC_vecs.append(arc.astype("float32"))
        CLIP_vecs.append(l2_normalize_np(clip_adapt).astype("float32"))

        faiss_id = len(ARC_vecs) - 1
        meta_map[str(faiss_id)] = {
            "suspect_id": doc.get("suspect_id"),
            "case_id": doc.get("case_id"),
            "name": doc.get("name"),
            "notes": doc.get("notes"),
            "tags": doc.get("tags", []),
            "photo_path": photo_path
        }

    if not ARC_vecs:
        raise RuntimeError("No valid faces indexed. Add clear face photos in real DB.")

    ARC_DB = np.vstack(ARC_vecs).astype("float32")
    CLIP_DB = np.vstack(CLIP_vecs).astype("float32")

    d = ARC_DB.shape[1]
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(ARC_DB)

    faiss.write_index(index, ARC_INDEX_PATH)
    np.save(ARC_DB_PATH, ARC_DB)
    np.save(CLIP_DB_PATH, CLIP_DB)
    with open(META_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, ensure_ascii=False)

    return ARC_DB.shape[0]
