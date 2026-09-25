"""Package the project + your local GZ2 images for upload to Kaggle.

Usage:
    python make_upload_zips.py [REPO_DIR] [IMAGES_DIR]

Defaults: REPO_DIR = this folder, IMAGES_DIR = data/processed/images
(adjust to wherever your {asset_id}.jpg files actually are).

Produces ./upload/:
    repo.zip        - the project (skips .git, venvs, big .npz, images)
    images_a.zip    - first half of the JPEGs (filenames at zip root)
    images_b.zip    - second half
"""
import os
import sys
import zipfile

SKIP_DIRS = {".git", ".venv-test", "venv", "__pycache__", ".pytest_cache",
             "node_modules"}
SKIP_EXT = {".npz"}  # keep large array artifacts out of the repo zip


def make_repo_zip(repo_dir, out):
    n = 0
    with zipfile.ZipFile(os.path.join(out, "repo.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for r, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if os.path.splitext(f)[1].lower() in SKIP_EXT:
                    continue
                p = os.path.join(r, f)
                z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(repo_dir))))
                n += 1
    print(f"repo.zip: {n} files")


def make_image_zips(images_dir, out, two_parts=True):
    jpgs = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    if not jpgs:
        sys.exit(f"No .jpg files found in {images_dir}")
    parts = (jpgs[: len(jpgs) // 2], jpgs[len(jpgs) // 2:]) if two_parts else (jpgs,)
    for i, part in enumerate(parts):
        name = f"images_{chr(97 + i)}.zip" if two_parts else "images.zip"
        with zipfile.ZipFile(os.path.join(out, name), "w", zipfile.ZIP_STORED) as z:
            for p in part:  # jpgs already compressed -> STORED is fastest
                z.write(p, os.path.basename(p))
        print(f"{name}: {len(part)} images")
    return len(jpgs)


def main():
    repo_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    images_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo_dir, "data", "processed", "images")
    out = "upload"
    os.makedirs(out, exist_ok=True)

    if not os.path.isfile(os.path.join(repo_dir, "scripts", "train_model.py")):
        sys.exit(f"{repo_dir} does not look like the project root (no scripts/train_model.py)")

    make_repo_zip(repo_dir, out)
    n = make_image_zips(images_dir, out)
    print(f"\ndone. Upload these {1 + (2 if n > 1 else 1)} files to a Kaggle dataset named 'galaxy-morph-v2':")
    for f in sorted(os.listdir(out)):
        print(f"  {f}  ({os.path.getsize(os.path.join(out, f)) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
