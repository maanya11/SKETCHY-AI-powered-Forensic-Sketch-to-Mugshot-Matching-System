import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "forensic_db")

STORAGE_DIR = os.getenv("STORAGE_DIR", r"C:\SuspectDB")
PHOTOS_DIR = os.path.join(STORAGE_DIR, "photos")
SKETCHES_DIR = os.path.join(STORAGE_DIR, "sketches")
INDEX_DIR = os.path.join(STORAGE_DIR, "index_store")

ADAPTER_PATH = os.getenv("ADAPTER_PATH", "training/saved/adapter.pt")

ARC_INDEX_PATH = os.path.join(INDEX_DIR, "arc_hnsw.index")
ARC_DB_PATH = os.path.join(INDEX_DIR, "ARC_DB.npy")
CLIP_DB_PATH = os.path.join(INDEX_DIR, "CLIP_DB.npy")
META_MAP_PATH = os.path.join(INDEX_DIR, "meta_map.json")

TOPK_CANDIDATES = int(os.getenv("TOPK_CANDIDATES", "200"))
TOPK_FINAL = int(os.getenv("TOPK_FINAL", "3"))

W_ARC = float(os.getenv("W_ARC", "0.65"))
W_CLIP = float(os.getenv("W_CLIP", "0.35"))

EF_SEARCH = int(os.getenv("EF_SEARCH", "128"))
HNSW_M = int(os.getenv("HNSW_M", "32"))
EF_CONSTRUCTION = int(os.getenv("EF_CONSTRUCTION", "200"))
