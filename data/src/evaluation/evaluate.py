# src/evaluation/evaluate.py
import os
import csv
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# Импортируем официальную метрику для детекции
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset.dataset import KittiDetectionDataset, collate_fn
from src.models.model_factory import get_model
from src.evaluation.metrics import detection_prf1


def _build_device(device_str: str):
    d = (device_str or "cpu").lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_eval(model_name: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    device = _build_device(cfg["project"].get("device", "cpu"))

    val_ds = KittiDetectionDataset(
        images_dir=data_cfg["images_dir"],
        labels_dir=data_cfg["labels_dir"],
        split_file=os.path.join(data_cfg["split_dir"], "val.txt"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )

    num_classes = len(data_cfg["classes"]) + 1
    model = get_model(model_name, num_classes=num_classes).to(device)

    if "yolo" in model_name.lower():
        ckpt_path = os.path.join(paths_cfg["checkpoints"], "yolo11_final.pt")
    else:
        ckpt_path = os.path.join(paths_cfg["checkpoints"], f"{model_name}_final.pth")

    if "yolo" in model_name.lower():
        model.load_state_dict(ckpt_path)
    else:
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            
    model.eval()

    # Инициализируем честный калькулятор mAP (учитывает координаты в формате xyxy)
    metric_calculator = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    # Оставляем классические TP/FP для расчета базовых Precision/Recall/F1
    total_tp, total_fp, total_fn = 0, 0, 0
    iou_threshold = 0.5
    score_threshold = 0.3  # Рекомендую снизить до 0.3, чтобы поднять Recall у YOLO!

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

    with torch.no_grad():
        for images, _targets in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            # Форматируем данные для torchmetrics mAP
            preds_metric = []
            targets_metric = []

            for out, target in zip(outputs, _targets):
                pred_boxes = out["boxes"].cpu()
                pred_scores = out["scores"].cpu()
                pred_labels = out["labels"].cpu() if "labels" in out else torch.zeros(len(pred_scores), dtype=torch.long)
                
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                # Фильтруем по порогу уверенности для torchmetrics
                keep = pred_scores >= score_threshold
                
                preds_metric.append({
                    "boxes": pred_boxes[keep],
                    "scores": pred_scores[keep],
                    "labels": pred_labels[keep]
                })
                targets_metric.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels
                })

                # Наш старый цикл для классических TP/FP/FN
                keep_boxes = pred_boxes[keep]
                matched_gt = set()
                tp, fp = 0, 0

                for p_box in keep_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for idx, g_box in enumerate(gt_boxes):
                        if idx in matched_gt:
                            continue
                        iou = calculate_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou >= iou_threshold:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1

                fn = len(gt_boxes) - len(matched_gt)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            # Обновляем torchmetrics на каждом батче
            metric_calculator.update(preds_metric, targets_metric)

    # Вычисляем финальный честный mAP
    computed_metrics = metric_calculator.compute()
    real_map50 = computed_metrics["map_50"].item()

    # Считаем базовые Precision/Recall/F1
    precision, recall, f1 = detection_prf1(total_tp, total_fp, total_fn)

    os.makedirs(paths_cfg["tables"], exist_ok=True)
    out_csv = os.path.join(paths_cfg["tables"], "final_comparison.csv")

    file_exists = os.path.exists(out_csv)
    is_old_format = False
    if file_exists:
        with open(out_csv, 'r') as f_check:
            first_line = f_check.readline()
            if "num_predictions_on_val" in first_line:
                is_old_format = True

    # Перезапишем файл полностью (режим 'w'), чтобы очистить старые неверные данные
    mode = 'w' if (not file_exists or is_old_format) else 'a'

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == 'w':
            w.writerow(["model", "mAP@0.5", "precision", "recall", "f1"])
        w.writerow([model_name, f"{real_map50:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

    print(f"\n[INFO] Успех! Честный mAP@0.5 = {real_map50:.4f}")
    print(f"[INFO] Результаты сохранены в: {out_csv}")