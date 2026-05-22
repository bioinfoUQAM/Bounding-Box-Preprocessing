"""
assign_boxes.py
---------------
Interactive ROI definition tool for fixed-camera animal tracking videos.
Displays the first frame of a video, lets the user draw rectangular ROIs,
then spatially filters bounding box detections to produce one CSV per ROI.

Usage:
    python assign_boxes.py --video <path> --csv <path> --output_dir <folder> [options]

Arguments:
    --video         Path to the .mp4 video file.
    --csv           Path to the arranged bounding box CSV (output of rearrange_boxes.py).
    --output_dir    Folder where per-ROI CSV files will be saved.
    --body_cols     Column names for body box (default: cow_L cow_T cow_W cow_H).
    --head_cols     Column names for head box (default: head_L head_T head_W head_H).
    --snout_cols    Column names for snout box (default: snout_L snout_T snout_W snout_H).

Controls (interactive window):
    Click + drag    Draw a rectangular ROI.
    ESC             Confirm all ROIs and close the window.
"""

import argparse
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Module-level state for the mouse callback ---
_rois = []
_current_roi = []
_roi_id = 1


def _draw_roi(event, x, y, flags, param):
    """OpenCV mouse callback to capture click-drag rectangles as ROIs."""
    global _current_roi, _rois, _roi_id
    if event == cv2.EVENT_LBUTTONDOWN:
        _current_roi = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        _current_roi.append((x, y))
        x1, y1 = _current_roi[0]
        x2, y2 = _current_roi[1]
        _rois.append({
            'id': _roi_id,
            'x1': min(x1, x2), 'y1': min(y1, y2),
            'x2': max(x1, x2), 'y2': max(y1, y2)
        })
        _roi_id += 1
        _current_roi = []


def select_rois_from_first_frame(video_path):
    """
    Display the first frame of a video and let the user draw ROIs interactively.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    list of dict or None
        List of ROI dicts with keys {id, x1, y1, x2, y2}, or None if the frame
        could not be read.
    """
    global _rois, _current_roi, _roi_id
    _rois = []
    _current_roi = []
    _roi_id = 1

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read the first frame from '{video_path}'.")
        return None

    print("Draw ROIs on the frame. Press ESC when done.")
    clone = frame.copy()
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", _draw_roi)

    while True:
        temp = clone.copy()
        for roi in _rois:
            cv2.rectangle(temp, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (0, 255, 0), 2)
            cv2.putText(temp, f"ID {roi['id']}", (roi['x1'], roi['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print(f"{len(_rois)} ROI(s) defined.")
    return _rois


def _center_in_roi(cx, cy, roi):
    """Return True if point (cx, cy) falls inside the ROI rectangle."""
    return roi['x1'] <= cx <= roi['x2'] and roi['y1'] <= cy <= roi['y2']


def _center_in_box(px, py, box_L, box_T, box_W, box_H):
    """Return True if point (px, py) falls inside a bounding box (L, T, W, H)."""
    return box_L <= px <= (box_L + box_W) and box_T <= py <= (box_T + box_H)


def filter_detections_by_roi(df, rois,
                              body_cols=('cow_L', 'cow_T', 'cow_W', 'cow_H'),
                              head_cols=('head_L', 'head_T', 'head_W', 'head_H'),
                              snout_cols=('snout_L', 'snout_T', 'snout_W', 'snout_H')):
    """
    Assign each detection row to the ROI whose boundary contains the body box centre.
    Head and snout centres are accepted if they fall inside the ROI *or* inside
    the body bounding box (posture tolerance).

    Parameters
    ----------
    df : pd.DataFrame
    rois : list of dict
    body_cols, head_cols, snout_cols : tuple of str

    Returns
    -------
    dict {roi_id: list of pd.Series}
    """
    roi_data = {roi['id']: [] for roi in rois}

    bL, bT, bW, bH = body_cols
    hL, hT, hW, hH = head_cols
    sL, sT, sW, sH = snout_cols

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning detections to ROIs"):
        if pd.isna(row[bL]) or pd.isna(row[bT]) or pd.isna(row[bW]) or pd.isna(row[bH]):
            continue

        cbx = row[bL] + row[bW] / 2
        cby = row[bT] + row[bH] / 2

        hbx = row[hL] + row[hW] / 2 if pd.notna(row[hL]) else None
        hby = row[hT] + row[hH] / 2 if pd.notna(row[hT]) else None
        sbx = row[sL] + row[sW] / 2 if pd.notna(row[sL]) else None
        sby = row[sT] + row[sH] / 2 if pd.notna(row[sT]) else None

        for roi in rois:
            if not _center_in_roi(cbx, cby, roi):
                continue
            # Validate head centre
            if hbx is not None:
                head_ok = (_center_in_roi(hbx, hby, roi) or
                           _center_in_box(hbx, hby, row[bL], row[bT], row[bW], row[bH]))
                if not head_ok:
                    continue
            # Validate snout centre
            if sbx is not None:
                snout_ok = (_center_in_roi(sbx, sby, roi) or
                            _center_in_box(sbx, sby, row[bL], row[bT], row[bW], row[bH]))
                if not snout_ok:
                    continue
            roi_data[roi['id']].append(row)
            break

    return roi_data


def process_and_export(csv_path, rois, output_dir,
                       body_cols=('cow_L', 'cow_T', 'cow_W', 'cow_H'),
                       head_cols=('head_L', 'head_T', 'head_W', 'head_H'),
                       snout_cols=('snout_L', 'snout_T', 'snout_W', 'snout_H')):
    """Load CSV, filter by ROIs, and export one CSV per ROI."""
    df = pd.read_csv(csv_path).replace('none', np.nan)
    for col in list(body_cols):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    os.makedirs(output_dir, exist_ok=True)
    roi_data = filter_detections_by_roi(df, rois,
                                        body_cols=body_cols,
                                        head_cols=head_cols,
                                        snout_cols=snout_cols)

    for roi_id, rows in roi_data.items():
        if rows:
            out_path = os.path.join(output_dir, f"roi_{roi_id}_new.csv")
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"ROI {roi_id}: {len(rows)} rows saved to {out_path}")
        else:
            print(f"ROI {roi_id}: no detections found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ROI selector and detection filter.")
    parser.add_argument("--video",      required=True, help="Path to the .mp4 video file.")
    parser.add_argument("--csv",        required=True, help="Path to the arranged detection CSV.")
    parser.add_argument("--output_dir", required=True, help="Output folder for per-ROI CSVs.")
    parser.add_argument("--body_cols",  nargs=4, default=["cow_L", "cow_T", "cow_W", "cow_H"],  metavar=("L", "T", "W", "H"))
    parser.add_argument("--head_cols",  nargs=4, default=["head_L", "head_T", "head_W", "head_H"], metavar=("L", "T", "W", "H"))
    parser.add_argument("--snout_cols", nargs=4, default=["snout_L", "snout_T", "snout_W", "snout_H"], metavar=("L", "T", "W", "H"))
    args = parser.parse_args()

    rois = select_rois_from_first_frame(args.video)
    if rois:
        process_and_export(args.csv, rois, args.output_dir,
                           body_cols=tuple(args.body_cols),
                           head_cols=tuple(args.head_cols),
                           snout_cols=tuple(args.snout_cols))
    else:
        print("No ROIs selected. Exiting.")
