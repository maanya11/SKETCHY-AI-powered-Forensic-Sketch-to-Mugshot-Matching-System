import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from insightface.app import FaceAnalysis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"

def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)

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

class Models:
    def __init__(self, adapter_path: str):
        self.clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
        self.proc = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        ckpt = torch.load(adapter_path, map_location=DEVICE)
        self.dim = ckpt["dim"]
        self.adapter = Adapter(dim=self.dim).to(DEVICE)
        self.adapter.load_state_dict(ckpt["state_dict"])
        self.adapter.eval()

        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1)

    @torch.no_grad()
    def clip_feat(self, pil_img: Image.Image) -> np.ndarray:
        inputs = self.proc(images=pil_img, return_tensors="pt").to(DEVICE)
        feat = self.clip.get_image_features(**inputs)[0]   # (D,)
        feat = feat / feat.norm(p=2)
        return feat.detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def adapted_feat(self, pil_img: Image.Image) -> np.ndarray:
        inputs = self.proc(images=pil_img, return_tensors="pt").to(DEVICE)
        feat = self.clip.get_image_features(**inputs)      # (1,D)
        z = self.adapter(feat)[0]
        z = F.normalize(z, dim=-1)
        return z.detach().cpu().numpy().astype("float32")

    def arcface(self, bgr: np.ndarray) -> np.ndarray | None:
        faces = self.face_app.get(bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = faces[0].embedding.astype("float32")
        return l2_normalize_np(emb)

def preprocess_sketch_for_clip(bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 180)
    inv = 255 - edges
    rgb = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def photo_for_clip(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
