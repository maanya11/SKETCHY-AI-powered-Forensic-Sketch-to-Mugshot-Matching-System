import os
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png")


def _is_img(f: str) -> bool:
    return f.lower().endswith(IMG_EXT)


def _build_pairs(photos_dir: str, sketches_dir: str):
    """
    Matches Kaggle sketch-photo pairs.

    Example:
      photo:   f-005-01.jpg
      sketch:  M2-005-01-sz1.jpg
    Matching key: 005-01
    """
    photos = [f for f in os.listdir(photos_dir) if _is_img(f)]
    sketches = [f for f in os.listdir(sketches_dir) if _is_img(f)]

    photo_map = {}
    for p in photos:
        stem = os.path.splitext(p)[0].lower()
        parts = stem.split("-")
        if len(parts) >= 3:
            key = f"{parts[1]}-{parts[2]}"
            photo_map[key] = p

    pairs = []
    for s in sketches:
        stem = os.path.splitext(s)[0].lower()
        parts = stem.split("-")
        if len(parts) >= 3:
            key = f"{parts[1]}-{parts[2]}"
            if key in photo_map:
                pairs.append(
                    (
                        os.path.join(sketches_dir, s),
                        os.path.join(photos_dir, photo_map[key]),
                    )
                )

    if not pairs:
        raise RuntimeError("No sketch-photo pairs found. Check filenames.")

    return pairs


class KaggleSketchPhotoPairs(Dataset):
    def __init__(self, photos_dir: str, sketches_dir: str):
        self.pairs = _build_pairs(photos_dir, sketches_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.pairs[idx]
        return sketch_path, photo_path
