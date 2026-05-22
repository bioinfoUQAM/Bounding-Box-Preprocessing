# Bounding Box Preprocessing — ROI Assignment & Detection Rearrangement

Preprocessing modules for spatially filtering and reorganising cattle body-part detections prior to HMM-based behaviour analysis. Part of the [WELL-E](https://github.com/bioinfoUQAM) animal welfare computer vision pipeline.

***

## Overview

Three sequential modules transform raw bounding box detections (body, head, snout) into temporally coherent, noise-free feature sequences ready for Hidden Markov Model (HMM) behaviour classification:

1. **`rearrange_boxes.py`** groups body/head/snout detections per animal per frame using a greedy overlap-based assignment.
2. **`assign_boxes.py`** lets the user draw ROIs on the first video frame to spatially filter detections to a single animal.
3. **`inter_filt.py`** interpolates missing frames, smooths trajectories, and computes derivatives to produce the final HMM feature vectors.

```
Raw detection CSV
      │
      ▼
rearrange_boxes.py   →   *_arranged_boxes_.csv   (one row = one animal/frame)
      │
      ▼
assign_boxes.py      →   roi_<id>_new.csv         (one file per animal)
      │
      ▼
inter_filt.py        →   *_interpolated_filtered_derivates_*.csv
      │
      ▼
HMM feature extraction
```

***

## Requirements

- Python 3.8+
- OpenCV
- pandas
- numpy
- tqdm
- scipy

```bash
pip install opencv-python pandas numpy tqdm scipy
```

> **Note:** `assign_boxes.py` opens an interactive window and requires a display environment (desktop, X forwarding, or VNC on remote servers).

***

## Module 1 — `rearrange_boxes.py`

Groups raw detections into coherent per-animal rows using bounding box overlap scores.

### How it works

For each frame, the algorithm evaluates all combinations of body, head, and snout boxes and assigns them greedily in three passes:

| Pass | Components | Overlap score |
|------|-----------|---------------|
| **Triplet** | body + head + snout | sum of all 3 pairwise overlaps |
| **Duet** | body + head *or* body + snout | single pairwise overlap |
| **Singleton** | body only | no score needed |

The highest-scoring combination is selected first, its boxes are removed from the pool, and the process repeats. Missing landmarks are filled with `NaN`.

### Usage

```python
import pandas as pd
import numpy as np
from rearrange_boxes import separate_boxes

df = pd.read_csv("detections.csv").replace('none', np.nan).astype(float)
arranged = separate_boxes(df)
arranged.to_csv("output/detections_arranged_boxes_.csv", index=False)
```

### Inputs / Outputs

| | File | Description |
|--|------|-------------|
| **Input** | `detections.csv` | Raw detector output multiple rows per frame |
| **Output** | `*_arranged_boxes_.csv` | One row per animal per frame |

***

## Module 2 — `assign_boxes.py`

Interactive ROI drawing tool that filters arranged detections to a single animal zone.

### How it works

1. Opens the first video frame in an OpenCV window.
2. User draws rectangular ROIs by clicking and dragging. Each ROI is labelled with an ID.
3. Press **ESC** to confirm.
4. For each detection row, the body box centre is tested against each ROI. Head and snout centres are validated against the ROI *or* the body bounding box (posture tolerance).
5. Matching rows are exported to per-ROI CSV files.

### Interactive controls

| Action | Control |
|--------|---------|
| Draw ROI | Click + drag |
| Confirm | **ESC** |

### Usage

```python
from assign_boxes import select_rois_from_first_frame, process_and_export

video_path = "videos/animal_01.mp4"
csv_path   = "arranged/animal_01_arranged_boxes_.csv"
output_dir = "roi_output/animal_01/"

rois = select_rois_from_first_frame(video_path)
if rois:
    process_and_export(csv_path, rois, output_dir)
```

### Inputs / Outputs

| | File | Description |
|--|------|-------------|
| **Input** | `*_arranged_boxes_.csv` | Output of `rearrange_boxes.py` |
| **Input** | `.mp4` video | Fixed-camera recording |
| **Output** | `roi_<id>_new.csv` | One file per drawn ROI |

***

## Module 3 — `inter_filt.py`

Produces the final **HMM feature vectors** from per-animal ROI CSVs. Raw bounding box trajectories contain missing frames and high-frequency detector noise that would corrupt HMM state estimation. This module fills temporal gaps, smooths coordinate signals, and computes frame-to-frame derivatives — capturing both **position** and **motion** of each body part. The resulting 26-column feature CSV (coordinates + derivatives) is the direct input to the HMM observation model.

### How it works

The processing pipeline runs in four steps per animal file:

| Step | Function | Description |
|------|----------|-------------|
| **1. Fill gaps** | `fill_missing_frames()` | Inserts `NaN` rows for any frame number absent from the CSV |
| **2. Interpolate** | `interpolate_data()` | Fills `NaN` values using linear interpolation |
| **3. Filter** | `butter_lowpass_filter()` | Applies a zero-phase Butterworth low-pass filter (`cutoff=0.1`, `fs=30 Hz`, `order=1`) to remove high-frequency noise |
| **4. Derivatives** | `calculate_derivatives()` | Computes numerical derivatives (via `np.gradient`) for each coordinate column — appended as `d_cow_L`, `d_head_T`, etc. |

### Usage

```python
import pandas as pd
import numpy as np
from inter_filt import fill_missing_frames, interpolate_data, butter_lowpass_filter, calculate_derivatives

data = pd.read_csv("roi_output/animal_01/roi_1_new.csv")
original_df = data[['frame','cow_L','cow_T','cow_W','cow_H',
                     'head_L','head_T','head_W','head_H',
                     'snout_L','snout_T','snout_W','snout_H']].copy()

full_df      = fill_missing_frames(original_df)
interpolated = interpolate_data(full_df)

filtered = interpolated.copy()
for col in [c for c in interpolated.columns if c != 'frame']:
    filtered[col] = butter_lowpass_filter(filtered[col].values, cutoff=0.1, fs=30, order=1)

derivatives = calculate_derivatives(filtered)
output = pd.concat([filtered, derivatives], axis=1).iloc[:-1]
output.to_csv("features/animal_01_interpolated_filtered_derivates.csv", index=False)
```

### Inputs / Outputs

| | File | Description |
|--|------|-------------|
| **Input** | `roi_<id>_new.csv` | Output of `assign_boxes.py`: per-animal filtered detections |
| **Output** | `*_interpolated_filtered_derivates_*.csv` | 26-column HMM feature CSV: 13 smoothed coordinates + 13 derivatives |

### Output CSV format

| Columns | Description |
|---------|-------------|
| `frame`, `cow_L` … `snout_H` | Smoothed bounding box coordinates |
| `d_cow_L` … `d_snout_H` | Frame-to-frame derivatives of each coordinate |

***

## CSV Format

### Intermediate CSVs (Modules 1 & 2)

All intermediate CSVs share the same 13-column structure. Coordinates are in pixels from the top-left corner of the frame.

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `cow_L`, `cow_T`, `cow_W`, `cow_H` | Body box — left, top, width, height |
| `head_L`, `head_T`, `head_W`, `head_H` | Head box |
| `snout_L`, `snout_T`, `snout_W`, `snout_H` | Snout box |

Missing detections are stored as `'none'` in raw files and converted to `NaN` on load.

### Final Feature CSV (Module 3 output)

The final output contains 26 coordinate/derivative columns plus behavioural annotation columns. Each row corresponds to one frame.

| Column | Type | Description |
|--------|------|-------------|
| `Frame` | `int` | Frame number |
| `Times` | `float` | Timestamp in seconds (frame / fps) |
| `body box L/T/W/H` | `float` | Smoothed body bounding box coordinates |
| `head box L/T/W/H` | `float` | Smoothed head bounding box coordinates |
| `snout box L/T/W/H` | `float` | Smoothed snout bounding box coordinates |
| `ddt body box L/T/W/H` | `float` | Frame-to-frame derivatives of body coordinates |
| `ddt head box L/T/W/H` | `float` | Frame-to-frame derivatives of head coordinates |
| `ddt snout box L/T/W/H` | `float` | Frame-to-frame derivatives of snout coordinates |
| `Defecation` | `int` | Binary behaviour label (0/1) |
| `Drinking` | `int` | Binary behaviour label (0/1) |
| `Eating` | `int` | Binary behaviour label (0/1) |
| `Exploration` | `int` | Binary behaviour label (0/1) |
| `Grooming` | `int` | Binary behaviour label (0/1) |
| `Head Shaking (as if to detach)` | `int` | Binary behaviour label (0/1) |
| `Kneeling` | `int` | Binary behaviour label (0/1) |
| `Lying` | `int` | Binary behaviour label (0/1) |
| `Lying Down` | `int` | Binary behaviour label (0/1) |
| `Not Visible` | `int` | Binary behaviour label (0/1) |
| `Other (including inactive)` | `int` | Binary behaviour label (0/1) |
| `Scratching` | `int` | Binary behaviour label (0/1) |
| `Social Interaction` | `int` | Binary behaviour label (0/1) |
| `Standing` | `int` | Binary behaviour label (0/1) |
| `Standing Up` | `int` | Binary behaviour label (0/1) |
| `Urination` | `int` | Binary behaviour label (0/1) |
| `Vigilance toward the door` | `int` | Binary behaviour label (0/1) |

**Example rows:**

| Frame | Times | body box L | body box T | ... | Lying | Other (including inactive) | Standing |
|-------|-------|------------|------------|-----|-------|---------------------------|----------|
| 1 | 0.000 | 600.029 | 501.975 | ... | 1 | 1 | 0 |
| 2 | 0.033 | 600.067 | 502.139 | ... | 1 | 1 | 0 |
| 3 | 0.067 | 600.103 | 502.303 | ... | 1 | 1 | 0 |

> Behaviour labels are mutually inclusive multiple behaviours can be active simultaneously (e.g., `Lying = 1` and `Other = 1`).

***

## Limitations

- Fixed-camera only: ROIs must be redrawn if the camera moves between sessions.
- Rectangular ROIs only: non-rectangular zones are not supported.
- Body box is required as anchor: frames with no body detection are skipped.
- Hardcoded column names (`cow_L`, `head_L`, etc.) must match the input CSV exactly.

***

*Part of the WELL-E animal welfare research pipeline — UQAM / McGill.*
