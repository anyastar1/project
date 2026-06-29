import pandas as pd
p = "results/tables/final_comparison.csv"
df = pd.read_csv(p)
print(df.sort_values("mAP@0.5:0.95", ascending=False))