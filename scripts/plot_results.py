import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

# 1) Графики loss (если есть логи)
for fp in glob.glob("results/logs/*_train_log.csv"):
    df = pd.read_csv(fp)
    model = os.path.basename(fp).replace("_train_log.csv", "")
    plt.figure()
    plt.plot(df["epoch"], df["avg_loss"], marker='o')
    plt.title(f"{model} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.grid(True)
    plt.savefig(f"results/plots/{model}_loss.png", dpi=150)
    plt.close()
    print(f"✅ Saved: results/plots/{model}_loss.png")

# 2) Сравнение моделей (по количеству предсказаний)
cmp_path = "results/tables/final_comparison.csv"
if os.path.exists(cmp_path):
    df = pd.read_csv(cmp_path)
    
    # Проверяем, какие колонки есть
    if "num_predictions_on_val" in df.columns:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df["model"], df["num_predictions_on_val"])
        plt.title("Model Comparison: Number of Detections on Validation Set")
        plt.xlabel("Model")
        plt.ylabel("Number of Predictions")
        plt.xticks(rotation=30)
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("results/plots/model_comparison.png", dpi=150)
        plt.close()
        print("✅ Saved: results/plots/model_comparison.png")
    else:
        print("⚠️  Колонка 'num_predictions_on_val' не найдена в CSV")
        print("   Доступные колонки:", list(df.columns))
else:
    print("⚠️  Файл results/tables/final_comparison.csv не найден")

print("\n📁 Все графики сохранены в results/plots/")