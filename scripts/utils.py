import os
import logging
from datetime import datetime

# Base configuration
# Calculate base directory relative to the script location (one level up from 'scripts/')
DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.getenv("PROJECT_BASE_DIR", DEFAULT_BASE_DIR)

LOG_DIR = os.path.join(BASE_DIR, "logs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")

def setup_environment():
    """Create required directories and initialize logging"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f"processing_{datetime.now().strftime('%Y%m%d')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Environment setup complete")

def get_image_path(asset_id):
    """Get path for specific asset ID"""
    return os.path.join(IMAGE_DIR, f"{int(asset_id)}.jpg")

def get_image_count():
    """Count available JPEG images"""
    return len([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

# Initialize on import
setup_environment()