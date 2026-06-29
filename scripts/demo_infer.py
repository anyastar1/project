import os
import cv2
import yaml
import torch
import random
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from src.models.model_factory import get_model

CLASSES = ["__background__", "Car", "Pedestrian", "Cyclist"]
COLORS = {
    "Car": (0, 255, 0),
    "Pedestrian": (0, 165, 255),
    "Cyclist": (255, 0, 0),
}

def load_model(config_path, model_name):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg["project"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    num_classes = len(cfg["data"]["classes"]) + 1
    model = get_model(model_name, num_classes=num_classes)

    ckpt = os.path.join(cfg["paths"]["checkpoints"], f"{model_name}_final.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, device, cfg

def draw_preds(img_bgr, boxes, labels, scores, thr=0.5):
    for b, l, s in zip(boxes, labels, scores):
        if s < thr:
            continue
        x1, y1, x2, y2 = map(int, b)
        cls_name = CLASSES[int(l)] if int(l) < len(CLASSES) else str(int(l))
        color = COLORS.get(cls_name, (255, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, f"{cls_name} {s:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="faster_rcnn")
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--num_images", type=int, default=30)
    args = ap.parse_args()

    model, device, cfg = load_model(args.config, args.model)

    split_file = os.path.join(cfg["data"]["split_dir"], "test.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [x.strip() for x in f if x.strip()]

    random.shuffle(ids)
    ids = ids[:args.num_images]

    out_dir = "results/demo_frames"
    os.makedirs(out_dir, exist_ok=True)

    for sid in ids:
        img_path_png = os.path.join(cfg["data"]["images_dir"], f"{sid}.png")
        img_path_jpg = os.path.join(cfg["data"]["images_dir"], f"{sid}.jpg")
        img_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg

        pil = Image.open(img_path).convert("RGB")
        t = F.to_tensor(pil).to(device)

        with torch.no_grad():
            pred = model([t])[0]

        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img_bgr = draw_preds(
            img_bgr,
            pred["boxes"].detach().cpu().numpy(),
            pred["labels"].detach().cpu().numpy(),
            pred["scores"].detach().cpu().numpy(),
            thr=args.score_thr,
        )

        cv2.imwrite(os.path.join(out_dir, f"{sid}.jpg"), img_bgr)

    print(f"[INFO] Saved visual results to: {out_dir}")

if __name__ == "__main__":
    main()