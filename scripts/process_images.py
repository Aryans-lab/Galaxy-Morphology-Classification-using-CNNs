import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import logging
import os
from utils import BASE_DIR, IMAGE_DIR, get_image_path

def process_images():
    try:
        # Load labels
        labels_path = os.path.join(BASE_DIR, "data", "processed", "filtered_labels.csv")
        df = pd.read_csv(labels_path)
        logging.info(f"Processing {len(df)} galaxies")
        
        # Initialize storage
        images = []
        labels = []
        failed_assets = []
        
        # Process with progress tracking
        for asset_id in tqdm(df['asset_id'].unique(), desc="Processing galaxies"):
            img_path = get_image_path(asset_id)
            try:
                img = Image.open(img_path)
                img = img.resize((128, 128)).convert('RGB')
                images.append(np.array(img))
                labels.append(df[df['asset_id'] == asset_id]['label'].values[0])
            except Exception as e:
                failed_assets.append(asset_id)
                logging.warning(f"Failed {asset_id}: {str(e)}")
        
        # Convert to arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Save dataset
        output_path = os.path.join(BASE_DIR, "data", "processed", "galaxy_dataset.npz")
        np.savez_compressed(output_path, images=images, labels=labels)
        
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