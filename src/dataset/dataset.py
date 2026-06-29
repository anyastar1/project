import os
from PIL import Image
import torch
from torch.utils.data import Dataset

KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(KITTI_CLASSES)}  


def parse_kitti_label_file(label_path: str):
    boxes, labels = [], []

    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls not in CLASS_TO_ID:
                continue

            x1, y1, x2, y2 = map(float, parts[4:8])

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[cls])

    return boxes, labels


class KittiDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, split_file, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        with open(split_file, "r", encoding="utf-8") as f:
            self.ids = [x.strip() for x in f.readlines() if x.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, f"{sample_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{sample_id}.jpg")

        label_path = os.path.join(self.labels_dir, f"{sample_id}.txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes, labels = parse_kitti_label_file(label_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.shape[0] > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
        }

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def run_preprocess(config):
    print("[INFO] KITTI preprocess step: using ready images/labels/splits from config paths.")