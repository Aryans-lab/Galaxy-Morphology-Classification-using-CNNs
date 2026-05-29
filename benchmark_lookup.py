import time
import pandas as pd
import numpy as np

# Create a dummy dataframe
N = 10000
df = pd.DataFrame({
    'asset_id': np.arange(N),
    'label': np.random.randint(0, 2, size=N)
})

# Baseline
start = time.perf_counter()
labels = []
for asset_id in df['asset_id'].unique():
    labels.append(df[df['asset_id'] == asset_id]['label'].values[0])
end = time.perf_counter()
print(f"Baseline (O(N) lookup): {end - start:.4f} seconds")

# Optimized
start = time.perf_counter()
labels_opt = []
label_dict = df.set_index('asset_id')['label'].to_dict()
for asset_id in df['asset_id'].unique():
    labels_opt.append(label_dict[asset_id])
end = time.perf_counter()
print(f"Optimized (dict lookup): {end - start:.4f} seconds")

# Assert correctness
assert labels == labels_opt, "Results do not match!"
