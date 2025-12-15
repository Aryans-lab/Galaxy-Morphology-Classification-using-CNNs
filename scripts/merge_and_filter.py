import pandas as pd
import numpy as np
import logging
import os
from utils import BASE_DIR, get_image_path

def load_and_merge():
    try:
        raw_dir = os.path.join(BASE_DIR, "data", "raw")
        
        # Load datasets with memory-efficient dtypes
        mapping = pd.read_csv(
            os.path.join(raw_dir, "gz2_filename_mapping.csv"),
            dtype={'objid': 'str', 'asset_id': 'int32'}
        )
        hart = pd.read_csv(
            os.path.join(raw_dir, "gz2_hart16.csv"),
            usecols=['dr7objid', 't01_smooth_or_features_a01_smooth_debiased', 
                     't01_smooth_or_features_a02_features_or_disk_debiased'],
            dtype={'dr7objid': 'str'}
        )
        
        # Merge datasets
        merged = hart.merge(
            mapping,
            left_on="dr7objid",
            right_on="objid",
            how="inner"
        )
        logging.info(f"Merged dataset size: {len(merged)}")
        
        # Check image existence
        merged['image_exists'] = merged['asset_id'].apply(
            lambda x: os.path.exists(get_image_path(x)))
        valid = merged[merged['image_exists']].copy()
        
        # Create high-confidence labels
        conditions = [
            valid['t01_smooth_or_features_a01_smooth_debiased'] > 0.8,
            valid['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.8
        ]
        choices = [0, 1]  # 0=Elliptical, 1=Spiral
        valid['label'] = np.select(conditions, choices, default=-1)
        labeled = valid[valid['label'] != -1]
        
        # Save results
        output_path = os.path.join(BASE_DIR, "data", "processed", "filtered_labels.csv")
        labeled.to_csv(output_path, index=False)
        logging.info(f"Saved {len(labeled)} labeled galaxies")
        return True
        
    except Exception as e:
        logging.error(f"Merge failed: {str(e)}")
        return False

if __name__ == "__main__":
    if load_and_merge():
        print("Merge completed successfully")
    else:
        print("Merge failed - check logs")