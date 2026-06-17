import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from dataset_kaggle_pairs import KaggleSketchPhotoPairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"


class Adapter(nn.Module):
    """Small trainable MLP on top of frozen CLIP image features."""
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def info_nce_loss(q, k_all, temperature: float = 0.07):
    """
    q: (B,D) query embeddings (sketch, after adapter)
    k_all: (B,D) key embeddings (photos, frozen CLIP)
    Positives are aligned by batch index.
    """
    q = F.normalize(q, dim=-1)
    k_all = F.normalize(k_all, dim=-1)

    logits = (q @ k_all.T) / temperature  # (B,B)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def clip_image_features(model: CLIPModel, processor: CLIPProcessor, images):
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)  # (B,D)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_root", required=True, help="Path containing photos/ and sketches/")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="training/saved/adapter.pt")
    args = ap.parse_args()

    photos_dir = os.path.join(args.kaggle_root, "photos")
    sketches_dir = os.path.join(args.kaggle_root, "sketches")

    ds = KaggleSketchPhotoPairs(photos_dir, sketches_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    # Freeze CLIP
    for p in clip.parameters():
        p.requires_grad = False
    clip.eval()

    # Determine feature dim (ds returns paths now)
    sample_sketch_path, _ = ds[0]
    sample_sketch = Image.open(sample_sketch_path).convert("RGB")
    with torch.no_grad():
        feat = clip_image_features(clip, proc, [sample_sketch])
        dim = feat.shape[-1]

    adapter = Adapter(dim=dim).to(DEVICE)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0

        for sketch_paths, photo_paths in pbar:
            # DataLoader collates strings (paths) fine
            sketches = [Image.open(p).convert("RGB") for p in sketch_paths]
            photos = [Image.open(p).convert("RGB") for p in photo_paths]

            # Frozen CLIP features
            with torch.no_grad():
                f_sk = clip_image_features(clip, proc, sketches)  # (B,D)
                f_ph = clip_image_features(clip, proc, photos)    # (B,D)

            # Train adapter: map sketch features -> closer to matching photo features
            z_sk = adapter(f_sk)                # (B,D)
            loss = info_nce_loss(z_sk, f_ph)    # InfoNCE over batch

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=running / max(1, pbar.n))

        # Save after each epoch
        torch.save({"dim": dim, "state_dict": adapter.state_dict()}, args.out)

    print("Saved adapter to:", args.out)
    print("Device:", DEVICE)


if __name__ == "__main__":
    main()
