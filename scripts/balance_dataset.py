import numpy as np
import os
import gc
import logging
from skimage.transform import resize
from imblearn.over_sampling import RandomOverSampler
from utils import BASE_DIR

def balance_dataset():
    try:
        # 1. Load dataset with memory optimization
        input_path = os.path.join(BASE_DIR, "data", "processed", "galaxy_dataset.npz")
        with np.load(input_path) as data:
            X, y = data['images'], data['labels']
        
        # 2. Report initial balance
        unique, counts = np.unique(y, return_counts=True)
        logging.info(f"Initial class counts: {dict(zip(unique, counts))}")
        
        if len(unique) != 2:
            raise ValueError("Expected binary classification data")

        # 3. Calculate target size (100*100 preserves details while saving memory)
        TARGET_SIZE = (100, 100)
        logging.info(f"Downscaling images to {TARGET_SIZE} for memory efficiency")
        
        # 4. Downsample in batches to avoid memory spikes
        batch_size = 500  # Conservative batch size for 15GB RAM
        X_ds = np.empty((X.shape[0], *TARGET_SIZE, X.shape[3]), dtype=np.uint8)
        
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch = X[i:end_idx]
            
            # Vectorized resizing with anti-aliasing
            resized_batch = np.zeros((len(batch), *TARGET_SIZE, 3), dtype=np.float32)
            for j in range(len(batch)):
                resized_batch[j] = resize(
                    batch[j], 
                    TARGET_SIZE,
                    preserve_range=True,
                    anti_aliasing=True
                )
            X_ds[i:end_idx] = resized_batch.astype(np.uint8)
            
            # Explicit memory cleanup
            del batch, resized_batch
            gc.collect()
        
        del X  # Free original images (saves 3.3GB+)
        gc.collect()
        
        # 5. Balance with RandomOverSampler (best for image data)
        ros = RandomOverSampler(random_state=42)
        X_flat = X_ds.reshape(len(X_ds), -1)  # No memory copy
        X_bal_flat, y_bal = ros.fit_resample(X_flat, y)
        
        # 6. Reshape back to images
        X_bal = X_bal_flat.reshape(-1, *TARGET_SIZE, 3)
        
        # 7. Save memory-optimized balanced dataset
        output_path = os.path.join(BASE_DIR, "data", "processed", "galaxy_dataset_balanced_96x96.npz")
        np.savez_compressed(output_path, images=X_bal, labels=y_bal)
        
        # 8. Verify and log results
        unique_bal, counts_bal = np.unique(y_bal, return_counts=True)
        logging.info(f"Balanced dataset counts: {dict(zip(unique_bal, counts_bal))}")
        logging.info(f"Final dataset shape: {X_bal.shape}")
        return True

    except Exception as e:
        logging.error(f"Balancing failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if balance_dataset():
        print("✅ Balancing completed successfully. Output: galaxy_dataset_balanced_96x96.npz")
    else:
        print("❌ Balancing failed - check logs")