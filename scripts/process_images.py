import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import logging
import os
from utils import BASE_DIR, get_image_path

SIZE = 128  # process_images standardizes to 128x128; make_splits resizes to 100


def process_images():
    try:
        # Load labels
        labels_path = os.path.join(BASE_DIR, "data", "processed", "filtered_labels.csv")
        df = pd.read_csv(labels_path)
        logging.info(f"Processing {len(df)} galaxies")

        asset_ids = df['asset_id'].unique()

        # Pre-compute label mapping for O(1) lookup
        label_dict = df.set_index('asset_id')['label'].to_dict()

        # Pre-allocate instead of list-then-copy: peak memory ~8 GB instead
        # of ~15 GB for the full 156k-image dataset (matters on 20 GB boxes).
        images = np.empty((len(asset_ids), SIZE, SIZE, 3), dtype=np.uint8)
        labels = np.empty(len(asset_ids), dtype=np.int64)
        asset_out = np.empty(len(asset_ids), dtype=asset_ids.dtype)
        n = 0
        failed_assets = []

        # Process with progress tracking
        for asset_id in tqdm(asset_ids, desc="Processing galaxies"):
            img_path = get_image_path(asset_id)
            try:
                img = Image.open(img_path)
                img = img.resize((SIZE, SIZE)).convert('RGB')
                images[n] = np.asarray(img)
                labels[n] = label_dict[asset_id]
                asset_out[n] = asset_id
                n += 1
            except Exception as e:
                failed_assets.append(asset_id)
                logging.warning(f"Failed {asset_id}: {str(e)}")

        images, labels, asset_out = images[:n], labels[:n], asset_out[:n]

        # Save dataset (also store asset ids for provenance; make_splits
        # carries them through to the splits manifest)
        output_path = os.path.join(BASE_DIR, "data", "processed", "galaxy_dataset.npz")
        np.savez_compressed(output_path, images=images, labels=labels,
                            asset_ids=asset_out)

        logging.info(f"Saved {len(images)} images | Failed: {len(failed_assets)}")
        return True

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    if process_images():
        print("Image processing completed")
    else:
        print("Processing failed - check logs")