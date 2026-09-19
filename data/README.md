## Dataset Description

This directory contains processed data products derived from **Galaxy Zoo 2 /
Galaxy Zoo DR7** (images from the SDSS ugriz stack; labels from the
crowdsourced, debiased vote fractions in Hart et al. 2016).

### Label task

Binary morphological classification from the GZ2 *t01* question
("completely smooth" vs. "features or a disk"), using the **debiased**
vote fractions:

| Label | GZ2 criterion                       | Physical content                          |
|-------|-------------------------------------|-------------------------------------------|
| 0     | `t01_smooth_debiased > 0.8`         | Smooth light: ellipticals + lenticulars   |
| 1     | `t01_features_debiased > 0.8`       | Disk / feature-rich: mostly spirals       |

Galaxies that are confident in *neither* class (or intermediate) are
dropped. Note that the "smooth" class is *not* a pure elliptical sample -
lenticulars (S0s) sit in it - so we display the classes as
**Smooth** and **Disk/Feature** rather than "Elliptical/Spiral".

### Files

- `filtered_labels.csv` - merged + filtered labels (committed; small)
- `galaxy_dataset.npz` - 128x128 RGB images + labels (created by
  `scripts/process_images.py`; too large for git)
- `galaxy_dataset_splits_100x100.npz` - the single seeded train/val/test
  split created by `scripts/make_splits.py` (see `splits_manifest.json`)
- `galaxy_dataset_train_balanced.npz` - the **training split only**,
  oversampled with `RandomOverSampler` by `scripts/balance_dataset.py`
  (val/test are never resampled)

Raw Galaxy Zoo images are not included due to size and licensing
constraints; see the README "Data Setup" section for where to obtain
them.
