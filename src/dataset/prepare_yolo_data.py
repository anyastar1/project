# src/dataset/prepare_yolo_data.py
import os
import glob
import shutil
import random
from PIL import Image
from tqdm import tqdm

CLASSES_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

def convert_kitti_line_to_yolo(parts, img_w, img_h):
    # Координаты KITTI: xmin, ymin, xmax, ymax
    xmin = float(parts[4])
    ymin = float(parts[5])
    xmax = float(parts[6])
    ymax = float(parts[7])
    
    # Конвертация в YOLO: x_center, y_center, width, height (нормализованные от 0 до 1)
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    return f"{CLASSES_MAP[parts[0]]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def make_splits_and_convert(src_images, src_labels, split_dir, dest_root, train_ratio=0.8):
    os.makedirs(split_dir, exist_ok=True)
    
    # Получаем все доступные базовые имена файлов (например, '000000', '000001')
    all_label_files = glob.glob(os.path.join(src_labels, "*.txt"))
    all_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_label_files]
    
    # Проверяем, есть ли вообще данные
    if not all_ids:
        print(f"[Ошибка] Папка с исходной разметкой {src_labels} пуста или не найдена!")
        return
        
    # Перемешиваем случайным образом для честного разделения
    random.seed(42)  # Фиксируем сид для воспроизводимости экспериментов по требованию практики
    random.shuffle(all_ids)
    
    # Считаем индекс разделения
    split_idx = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]
    
    # Сохраняем текстовые файлы сплитов (чтобы ваш исходный KittiDetectionDataset тоже мог их читать)
    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(split_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_ids))
        
    print(f"[ОК] Созданы списки сплитов в {split_dir}: {len(train_ids)} для train, {len(val_ids)} для val")
    
    # Обрабатываем каждый сплит
    splits = {"train": train_ids, "val": val_ids}
    for split_name, file_ids in splits.items():
        print(f"Конвертация и копирование сплита: {split_name}...")
        
        dest_images_dir = os.path.join(dest_root, "images", split_name)
        dest_labels_dir = os.path.join(dest_root, "labels", split_name)
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        
        for file_id in tqdm(file_ids):
            img_src_path = os.path.join(src_images, f"{file_id}.png")
            label_src_path = os.path.join(src_labels, f"{file_id}.txt")
            
            if not os.path.exists(img_src_path) or not os.path.exists(label_src_path):
                continue
                
            # Извлекаем размеры для нормализации
            with Image.open(img_src_path) as img:
                img_w, img_h = img.size
                
            # Копируем изображение в структуру YOLO
            shutil.copy(img_src_path, os.path.join(dest_images_dir, f"{file_id}.png"))
            
            # Конвертируем разметку KITTI -> YOLO
            yolo_lines = []
            with open(label_src_path, "r") as f_in:
                for line in f_in:
                    parts = line.strip().split(" ")
                    if parts[0] in CLASSES_MAP:
                        yolo_line = convert_kitti_line_to_yolo(parts, img_w, img_h)
                        yolo_lines.append(yolo_line)
                        
            # Записываем файл разметки YOLO
            with open(os.path.join(dest_labels_dir, f"{file_id}.txt"), "w") as f_out:
                f_out.write("\n".join(yolo_lines))

if __name__ == "__main__":
    # Конфигурация путей
    DATA_ROOT = "data/raw/kitti"
    SRC_IMAGES = os.path.join(DATA_ROOT, "images")
    SRC_LABELS = os.path.join(DATA_ROOT, "labels")
    SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
    YOLO_DATASET_ROOT = os.path.join(DATA_ROOT, "yolo_dataset")
    
    # Запуск разделения и конвертации
    make_splits_and_convert(SRC_IMAGES, SRC_LABELS, SPLIT_DIR, YOLO_DATASET_ROOT, train_ratio=0.8)
    
    # Создание yolo_data.yaml
    yaml_content = f"""path: {os.path.abspath(YOLO_DATASET_ROOT)}
train: images/train
val: images/val

names:
  0: Car
  1: Pedestrian
  2: Cyclist
"""
    with open(os.path.join(DATA_ROOT, "yolo_data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"\n[УСПЕХ] Данные полностью подготовлены!")
    print(f"Файл конфигурации для YOLO: {os.path.join(DATA_ROOT, 'yolo_data.yaml')}")