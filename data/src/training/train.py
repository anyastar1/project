import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import KittiDetectionDataset, collate_fn
from src.models.model_factory import get_model


def _build_device(device_str: str):
    d = (device_str or "cpu").lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_train(model_name: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    device = _build_device(cfg["project"].get("device", "cpu"))

    train_ds = KittiDetectionDataset(
        images_dir=data_cfg["images_dir"],
        labels_dir=data_cfg["labels_dir"],
        split_file=os.path.join(data_cfg["split_dir"], "train.txt"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )

    num_classes = len(data_cfg["classes"]) + 1
    model = get_model(model_name, num_classes=num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=tr_cfg["learning_rate"])

    epochs = tr_cfg["epochs"]
    model.train()

    os.makedirs(paths_cfg["checkpoints"], exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"[INFO] Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        ckpt_path = os.path.join(paths_cfg["checkpoints"], f"{model_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    final_path = os.path.join(paths_cfg["checkpoints"], f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved final checkpoint: {final_path}")