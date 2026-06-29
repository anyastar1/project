# src/models/yolo_model.py
import torch
import torch.nn as nn
from ultralytics import YOLO

class OriginalYOLO11Wrapper(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self._yolo_container = [YOLO("yolo11n.pt")]
        self.num_classes = num_classes
        self.data_yaml = "data/yolo_data.yaml"
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.is_trained = False

    @property
    def yolo(self):
        return self._yolo_container[0]

    def train(self, mode=True):
        super().train(mode)
        return self

    def load_state_dict(self, state_dict, strict=True):
        """
        Принимает путь к файлу весов (строку) из обновленного evaluate.py
        """
        if isinstance(state_dict, str):
            self._yolo_container = [YOLO(state_dict)]
            print(f"[YOLO11] Веса успешно загружены напрямую из: {state_dict}")
        else:
            # На случай, если пришел словарь
            expected_path = "results/checkpoints/yolo11_final.pt"
            import os
            if os.path.exists(expected_path):
                self._yolo_container = [YOLO(expected_path)]
                print(f"[YOLO11] Нативно подгружен файл: {expected_path}")
        
        from torch.nn.modules.module import _IncompatibleKeys
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])

    def forward(self, images, targets=None):
        if self.training:
            if not self.is_trained:
                print("\n[YOLO11] Запуск нативного процесса обучения...")
                self.yolo.train(
                    data=self.data_yaml,
                    epochs=5,
                    imgsz=640,
                    device="cpu",
                    workers=2,
                    project="results/checkpoints",
                    name="yolo11_run",
                    exist_ok=True
                )
                self.is_trained = True
            
            loss_value = 0.0 * self.dummy_param.sum()
            return {
                "loss_classifier": torch.tensor(0.0, requires_grad=True, device=images[0].device) + loss_value,
                "loss_box_reg": torch.tensor(0.0, requires_grad=True, device=images[0].device)
            }
        else:
            # --- ИСПРАВЛЕННЫЙ И БЕЗОПАСНЫЙ БЛОК ИНФЕРЕНСА (EVAL) ---
            output = []
            
            for img in images:
                # Переводим из (C, H, W) в (H, W, C) и возвращаем в диапазон 0-255
                img_numpy = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(__import__("numpy").uint8)
                
                with torch.no_grad():
                    results = self.yolo.predict(img_numpy, imgsz=640, verbose=False)
                
                for res in results:
                    # Защита от типов: проверяем, тензор перед нами или numpy-массив
                    if torch.is_tensor(res.boxes.xyxy):
                        boxes = res.boxes.xyxy.clone().detach().to(img.device)
                        scores = res.boxes.conf.clone().detach().to(img.device)
                        labels = res.boxes.cls.clone().detach().to(img.device) + 1  # <-- ДОБАВЛЯЕМ + 1
                    else:
                        boxes = torch.from_numpy(res.boxes.xyxy).to(img.device)
                        scores = torch.from_numpy(res.boxes.conf).to(img.device)
                        labels = torch.from_numpy(res.boxes.cls).long().to(img.device) + 1  # <-- ДОБАВЛЯЕМ + 1
                    
                    output.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels
                    })
            return output