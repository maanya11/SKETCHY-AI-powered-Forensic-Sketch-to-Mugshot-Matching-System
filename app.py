# ================================
# SketchyV1.26 Backend (AUTH + ADMIN + BULK UPLOAD + SEARCH + MEDIA)
# ================================

import os
import io
import json
import time
import uuid
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, Request, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pymongo import MongoClient
from jose import jwt, JWTError
from passlib.context import CryptContext

import faiss
from transformers import CLIPModel, CLIPProcessor

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

from backend.config import *

# ================================
# AUTH CONFIG
# ================================
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME")
JWT_ALG = "HS256"
JWT_EXP = 86400  # 24h
ADMIN_SIGNUP_CODE = os.getenv("ADMIN_SIGNUP_CODE", "CHANGE_ME")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"

# ================================
# ADAPTER (optional, for sketch->photo alignment)
# ================================
class Adapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

# ================================
# APP INIT
# ================================
app = FastAPI(title="Sketchy Backend v4 (Search + Bulk Upload)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# STORAGE DIRS
# ================================
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(SKETCHES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ================================
# DB INIT
# ================================
mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]
users_col = db["users"]
suspects_col = db["suspects"]

users_col.create_index("username", unique=True)
suspects_col.create_index("suspect_id", unique=True)

# ================================
# UTILS
# ================================
def hash_pw(pw): return pwd_context.hash(pw)
def verify_pw(pw, h): return pwd_context.verify(pw, h)

def create_token(data):
    data["exp"] = int(time.time()) + JWT_EXP
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

def get_user(req: Request):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    try:
        payload = decode_token(auth.split()[1])
    except JWTError:
        raise HTTPException(401, "Invalid token")

    user = users_col.find_one(
        {"username": payload["username"], "disabled": {"$ne": True}},
        {"_id": 0, "password": 0}
    )
    if not user:
        raise HTTPException(401, "User not found or disabled")
    return user

def admin_only(user=Depends(get_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admin only")
    return user

def safe_suspect_id_from_filename(name: str) -> str:
    # If user uploads "SUS-001.jpg" keep "SUS-001"
    base = os.path.splitext(os.path.basename(name))[0].strip()
    if base:
        return base
    return f"SUS-{uuid.uuid4().hex[:8].upper()}"

def pil_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(400, "Invalid image file")

def save_upload_to(path: str, upload: UploadFile):
    upload.file.seek(0)
    with open(path, "wb") as f:
        f.write(upload.file.read())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a,b (D,)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

# ================================
# MODEL ASSETS (lazy loaded)
# ================================
ASSETS_LOADED = False
clip_model = None
clip_proc = None
adapter = None
adapter_dim = None
face_app = None

def load_assets():
    global ASSETS_LOADED, clip_model, clip_proc, adapter, adapter_dim, face_app

    if ASSETS_LOADED:
        return

    # CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    clip_model.eval()

    # Adapter (optional)
    if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
        try:
            ckpt = torch.load(ADAPTER_PATH, map_location=DEVICE)
            adapter_dim = int(ckpt["dim"])
            adapter = Adapter(dim=adapter_dim).to(DEVICE)
            adapter.load_state_dict(ckpt["state_dict"])
            adapter.eval()
        except Exception:
            adapter = None
            adapter_dim = None

    # ArcFace (optional)
    if FaceAnalysis is not None:
        try:
            face_app = FaceAnalysis(name="buffalo_l")
            # ctx_id=-1 => CPU, 0 => GPU
            face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
        except Exception:
            face_app = None
    else:
        face_app = None

    ASSETS_LOADED = True

@torch.no_grad()
def clip_embed_images(pil_images: List[Image.Image]) -> np.ndarray:
    load_assets()
    inputs = clip_proc(images=pil_images, return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)

    # Apply adapter ONLY to sketch/query side in search (not here)
    return feats.detach().cpu().numpy().astype("float32")

def clip_embed_query(pil_img: Image.Image) -> np.ndarray:
    load_assets()
    inputs = clip_proc(images=[pil_img], return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)

    if adapter is not None:
        try:
            feats = adapter(feats)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
        except Exception:
            pass

    return feats.detach().cpu().numpy().astype("float32")[0]

def arcface_embed(pil_img: Image.Image) -> Optional[np.ndarray]:
    """
    Returns (512,) float32 unit vector if face found and insightface available, else None.
    """
    load_assets()
    if face_app is None:
        return None

    # insightface expects BGR numpy
    img = np.array(pil_img)[:, :, ::-1]
    faces = face_app.get(img)
    if not faces:
        return None

    # take largest face
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].embedding
    emb = emb.astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

# ================================
# INDEX STORE (loaded on demand)
# ================================
ARC_INDEX = None
ARC_DB = None
CLIP_DB = None
META_MAP = None

def load_index_store():
    global ARC_INDEX, ARC_DB, CLIP_DB, META_MAP
    if os.path.exists(ARC_INDEX_PATH) and os.path.exists(ARC_DB_PATH) and os.path.exists(CLIP_DB_PATH) and os.path.exists(META_MAP_PATH):
        ARC_INDEX = faiss.read_index(ARC_INDEX_PATH)
        ARC_DB = np.load(ARC_DB_PATH).astype("float32")
        CLIP_DB = np.load(CLIP_DB_PATH).astype("float32")
        with open(META_MAP_PATH, "r", encoding="utf-8") as f:
            META_MAP = json.load(f)
        return True
    return False

def clear_index_store():
    global ARC_INDEX, ARC_DB, CLIP_DB, META_MAP
    ARC_INDEX = None
    ARC_DB = None
    CLIP_DB = None
    META_MAP = None

# ================================
# AUTH ENDPOINTS
# ================================
@app.post("/auth/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    admin_code: str = Form("")
):
    role = "admin" if admin_code == ADMIN_SIGNUP_CODE else "user"
    doc = {
        "username": username.lower(),
        "password": hash_pw(password),
        "role": role,
        "disabled": False,
        "created_at": time.time()
    }
    try:
        users_col.insert_one(doc)
    except Exception:
        raise HTTPException(400, "Username exists")

    token = create_token({"username": doc["username"], "role": role})
    return {"token": token, "role": role}

@app.post("/auth/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    u = users_col.find_one({"username": username.lower()})
    if not u or u.get("disabled"):
        raise HTTPException(401, "Invalid credentials")
    if not verify_pw(password, u["password"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"username": u["username"], "role": u["role"]})
    return {"token": token, "role": u["role"]}

@app.get("/auth/me")
def me(user=Depends(get_user)):
    return user

# ================================
# ADMIN USER MANAGEMENT
# ================================
@app.get("/admin/users")
def list_users(_: dict = Depends(admin_only)):
    return list(users_col.find({}, {"_id": 0, "password": 0}))

@app.post("/admin/users/create")
def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
    _: dict = Depends(admin_only)
):
    doc = {
        "username": username.lower(),
        "password": hash_pw(password),
        "role": role if role in ["admin", "user"] else "user",
        "disabled": False,
        "created_at": time.time()
    }
    try:
        users_col.insert_one(doc)
    except Exception:
        raise HTTPException(400, "User exists")
    return {"created": True}

@app.put("/admin/users/{username}")
def change_username(
    username: str,
    new_username: str = Form(...),
    _: dict = Depends(admin_only)
):
    username = username.lower()
    new_username = new_username.lower()

    # prevent collisions
    if users_col.find_one({"username": new_username}):
        raise HTTPException(400, "New username already exists")

    res = users_col.update_one(
        {"username": username},
        {"$set": {"username": new_username}}
    )
    if res.matched_count == 0:
        raise HTTPException(404, "User not found")
    return {"updated": True}

@app.post("/admin/users/{username}/reset-password")
def reset_password(
    username: str,
    new_password: str = Form(...),
    _: dict = Depends(admin_only)
):
    res = users_col.update_one(
        {"username": username.lower()},
        {"$set": {"password": hash_pw(new_password)}}
    )
    if res.matched_count == 0:
        raise HTTPException(404, "User not found")
    return {"password_reset": True}

@app.post("/admin/users/{username}/disable")
def disable_user(
    username: str,
    disabled: bool = Form(True),
    _: dict = Depends(admin_only)
):
    res = users_col.update_one(
        {"username": username.lower()},
        {"$set": {"disabled": bool(disabled)}}
    )
    if res.matched_count == 0:
        raise HTTPException(404, "User not found")
    return {"disabled": bool(disabled)}

# ================================
# ADMIN: BULK UPLOAD SUSPECTS
# ================================
@app.post("/admin/bulk-upload")
async def bulk_upload(
    photos: List[UploadFile] = File(...),
    case_id: str = Form(""),
    tags: str = Form(""),
    notes: str = Form(""),
    _: dict = Depends(admin_only)
):
    """
    Upload multiple suspect photos at once.
    - Saves images to PHOTOS_DIR
    - Inserts/updates metadata in Mongo
    - Does NOT rebuild index automatically (call /admin/rebuild-index after)
    """
    uploaded = 0
    skipped = 0

    tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    for up in photos:
        sid = safe_suspect_id_from_filename(up.filename or f"SUS-{uuid.uuid4().hex[:8].upper()}")

        # Save as .jpg always (consistent)
        save_path = os.path.join(PHOTOS_DIR, f"{sid}.jpg")

        # If already exists in DB, skip (you can change this behavior if you want overwrite)
        existing = suspects_col.find_one({"suspect_id": sid})
        if existing:
            skipped += 1
            continue

        # Save file
        try:
            up.file.seek(0)
            img = pil_from_upload(up)
            img.save(save_path, format="JPEG", quality=95)
        except Exception:
            skipped += 1
            continue

        doc = {
            "suspect_id": sid,
            "case_id": case_id or "",
            "name": "Unknown",
            "notes": notes or "",
            "tags": tags_list,
            "photo_path": save_path,
            "created_at": time.time()
        }
        try:
            suspects_col.insert_one(doc)
            uploaded += 1
        except Exception:
            # if collision occurs
            skipped += 1

    return {"uploaded": uploaded, "skipped": skipped}

# ================================
# ADMIN: REBUILD INDEX
# ================================
@app.post("/admin/rebuild-index")
def rebuild_index(_: dict = Depends(admin_only)):
    """
    Rebuilds:
    - ArcFace FAISS HNSW index (if ArcFace available + faces found)
    - CLIP embeddings DB (always)
    Saves:
      arc_hnsw.index, ARC_DB.npy, CLIP_DB.npy, meta_map.json
    """
    load_assets()
    clear_index_store()

    suspects = list(suspects_col.find({}, {"_id": 0}))
    if not suspects:
        # still write empty stores
        for p in [ARC_INDEX_PATH, ARC_DB_PATH, CLIP_DB_PATH, META_MAP_PATH]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        return {"indexed": 0, "assets_loaded": True}

    arc_vecs = []
    clip_vecs = []
    meta_map = []

    # Load images and compute embeddings
    for s in suspects:
        path = s.get("photo_path")
        if not path or not os.path.exists(path):
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        # CLIP always
        c = clip_embed_images([img])[0]
        clip_vecs.append(c)

        # ArcFace optional
        a = arcface_embed(img)
        if a is None:
            # If no arc, use zeros (keeps alignment)
            a = np.zeros((512,), dtype="float32")
        arc_vecs.append(a)

        meta_map.append({
            "suspect_id": s.get("suspect_id"),
            "case_id": s.get("case_id", ""),
            "name": s.get("name", "Unknown"),
            "notes": s.get("notes", ""),
            "tags": s.get("tags", []),
            "photo_path": path
        })

    if not meta_map:
        return {"indexed": 0, "assets_loaded": True}

    ARC_DB_arr = np.vstack(arc_vecs).astype("float32")
    CLIP_DB_arr = np.vstack(clip_vecs).astype("float32")

    # Normalize Arc DB for cosine-like behavior
    norms = np.linalg.norm(ARC_DB_arr, axis=1, keepdims=True) + 1e-9
    ARC_DB_arr = ARC_DB_arr / norms

    dim = ARC_DB_arr.shape[1]  # 512

    # Build HNSW index
    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index.add(ARC_DB_arr)

    # Save to disk
    faiss.write_index(index, ARC_INDEX_PATH)
    np.save(ARC_DB_PATH, ARC_DB_arr)
    np.save(CLIP_DB_PATH, CLIP_DB_arr)
    with open(META_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, ensure_ascii=False, indent=2)

    # Load into RAM
    ARC_INDEX = index

    return {"indexed": int(ARC_DB_arr.shape[0]), "assets_loaded": True}

# ================================
# MEDIA: SERVE SUSPECT PHOTO
# ================================
@app.get("/media/{suspect_id}")
def get_suspect_photo(suspect_id: str, _user=Depends(get_user)):
    s = suspects_col.find_one({"suspect_id": suspect_id}, {"_id": 0})
    if not s:
        raise HTTPException(404, "Suspect not found")
    path = s.get("photo_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Photo file not found")
    return FileResponse(path)

# ================================
# SEARCH: UPLOAD SKETCH -> TOP-3 MATCHES
# ================================
@app.post("/search")
async def search(
    sketch: UploadFile = File(...),
    user: str = Form("operator"),
    _u=Depends(get_user)
):
    """
    Returns top-3 suspects:
    - Candidate retrieval: ArcFace FAISS HNSW (if available)
    - Rerank: weighted fusion ArcFace + CLIP
    Fallback:
    - If ArcFace not usable: CLIP-only search on stored CLIP_DB
    """
    load_assets()

    if not load_index_store():
        raise HTTPException(400, "Index not found. Admin must run /admin/rebuild-index first.")

    # Save sketch to sketches dir (optional log artifact)
    sketch_id = uuid.uuid4().hex
    sketch_save_path = os.path.join(SKETCHES_DIR, f"{sketch_id}.jpg")
    try:
        sketch.file.seek(0)
        img = pil_from_upload(sketch)
        img.save(sketch_save_path, format="JPEG", quality=95)
    except Exception:
        raise HTTPException(400, "Invalid sketch image")

    # Embeddings
    q_clip = clip_embed_query(img)  # (D,)
    q_arc = arcface_embed(img)      # (512,) or None

    # Load stores
    arc_db = np.load(ARC_DB_PATH).astype("float32")
    clip_db = np.load(CLIP_DB_PATH).astype("float32")
    with open(META_MAP_PATH, "r", encoding="utf-8") as f:
        meta_map = json.load(f)

    # CLIP-only fallback mode
    if q_arc is None or ARC_INDEX is None:
        # brute cosine against CLIP_DB (OK for <100k)
        sims = np.dot(clip_db, q_clip)  # clip_db is already normalized by CLIP
        top_idx = np.argsort(-sims)[:TOPK_FINAL]
        results = []
        for rank, idx in enumerate(top_idx, start=1):
            m = meta_map[int(idx)]
            results.append({
                "rank": rank,
                "score": float(sims[int(idx)]),
                "suspect_id": m["suspect_id"],
                "case_id": m.get("case_id", ""),
                "name": m.get("name", "Unknown"),
                "notes": m.get("notes", ""),
                "tags": m.get("tags", []),
                "photo_path": m.get("photo_path", ""),
                "photo_url": f"/media/{m['suspect_id']}"
            })
        return {"mode": "clip_only", "results": results}

    # Arc + CLIP fusion mode
    q_arc = q_arc.astype("float32")
    q_arc = q_arc / (np.linalg.norm(q_arc) + 1e-9)
    q_arc = q_arc.reshape(1, -1)

    # FAISS search (candidate retrieval)
    index = faiss.read_index(ARC_INDEX_PATH)
    index.hnsw.efSearch = EF_SEARCH

    D, I = index.search(q_arc, TOPK_CANDIDATES)
    cand_idx = [int(x) for x in I[0] if int(x) >= 0 and int(x) < len(meta_map)]
    if not cand_idx:
        raise HTTPException(404, "No candidates found")

    # Rerank candidates with fusion
    scored = []
    for idx in cand_idx:
        arc_sim = float(np.dot(arc_db[idx], q_arc[0]))  # cosine since normalized
        clip_sim = float(np.dot(clip_db[idx], q_clip))  # cosine since normalized
        score = (W_ARC * arc_sim) + (W_CLIP * clip_sim)
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOPK_FINAL]

    results = []
    for rank, (score, idx) in enumerate(top, start=1):
        m = meta_map[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "suspect_id": m["suspect_id"],
            "case_id": m.get("case_id", ""),
            "name": m.get("name", "Unknown"),
            "notes": m.get("notes", ""),
            "tags": m.get("tags", []),
            "photo_path": m.get("photo_path", ""),
            "photo_url": f"/media/{m['suspect_id']}"
        })

    return {"mode": "arc_clip_fusion", "results": results}

# ================================
# HEALTH
# ================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "auth": True,
        "admin": True,
        "search": True,
        "storage": {
            "PHOTOS_DIR": PHOTOS_DIR,
            "SKETCHES_DIR": SKETCHES_DIR,
            "INDEX_DIR": INDEX_DIR,
        }
    }
