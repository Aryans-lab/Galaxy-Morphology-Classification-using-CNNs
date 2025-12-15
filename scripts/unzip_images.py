import zipfile
import logging
import os
from utils import BASE_DIR, IMAGE_DIR, get_image_count

def unzip_with_nesting():
    try:
        src_zip = os.path.join(BASE_DIR, "data", "raw", "images_gz2.zip")
        
        # Skip if images already exist
        if get_image_count() > 0:
            logging.info(f"Found {get_image_count()} images - skipping extraction")
            return True

        logging.info(f"Starting extraction to {IMAGE_DIR}")
        with zipfile.ZipFile(src_zip, 'r') as z:
            # Extract while preserving structure
            z.extractall(os.path.dirname(IMAGE_DIR))
        
        # Verify extraction
        extracted_count = get_image_count()
        if extracted_count == 0:
            raise RuntimeError("No images extracted - check ZIP structure")
        
        logging.info(f"Extracted {extracted_count} images")
        return True
        
    except Exception as e:
        logging.error(f"Extraction failed: {str(e)}")
        return False

if __name__ == "__main__":
    if unzip_with_nesting():
        print("Unzip completed successfully")
    else:
        print("Unzip failed - check logs")