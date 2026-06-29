import os
from PIL import Image

img_dir = "data/raw/kitti/images"
lbl_dir = "data/raw/kitti/labels"
split_file = "data/raw/kitti/splits/train.txt"

bad_images = 0
missing_labels = 0
empty_labels = 0

with open(split_file, "r", encoding="utf-8") as f:
    ids = [x.strip() for x in f if x.strip()]

for _id in ids:
    img_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join(img_dir, _id + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        bad_images += 1
        continue

    try:
        Image.open(img_path).verify()
    except Exception:
        bad_images += 1

    lbl_path = os.path.join(lbl_dir, _id + ".txt")
    if not os.path.exists(lbl_path):
        missing_labels += 1
        continue

    with open(lbl_path, "r", encoding="utf-8") as lf:
        lines = [ln.strip() for ln in lf if ln.strip()]
    if len(lines) == 0:
        empty_labels += 1

print("bad_images:", bad_images)
print("missing_labels:", missing_labels)
print("empty_labels:", empty_labels)