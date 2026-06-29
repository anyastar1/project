import os, random

random.seed(42)

img_dir = "data/raw/kitti/images"
out_dir = "data/raw/kitti/splits"
os.makedirs(out_dir, exist_ok=True)

ids = []
for f in os.listdir(img_dir):
    if f.lower().endswith((".png", ".jpg", ".jpeg")):
        ids.append(os.path.splitext(f)[0])

ids = sorted(ids)
random.shuffle(ids)

n = len(ids)
n_train = int(0.7 * n)
n_val = int(0.2 * n)

train_ids = ids[:n_train]
val_ids = ids[n_train:n_train+n_val]
test_ids = ids[n_train+n_val:]

for name, arr in [("train.txt", train_ids), ("val.txt", val_ids), ("test.txt", test_ids)]:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        for x in arr:
            f.write(x + "\n")

print(f"Total={n} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")