"""
rearrange_boxes.py
------------------
Reassigns body-part bounding boxes (body, head, snout) to individual animals
per frame using a greedy overlap-based assignment strategy (triplets > duets > singletons).

Usage:
    python rearrange_boxes.py --input_dir <folder> --output_dir <folder> [options]

    Or process a single file:
    python rearrange_boxes.py --input_dir path/to/file.csv --output_dir path/to/output/

Arguments:
    --input_dir     Path to folder containing detection CSV files, or path to a single CSV.
    --output_dir    Path to folder where arranged CSV files will be saved.
    --body_cols     Column names for body box (default: cow_L cow_T cow_W cow_H).
    --head_cols     Column names for head box (default: head_L head_T head_W head_H).
    --snout_cols    Column names for snout box (default: snout_L snout_T snout_W snout_H).
    --frame_col     Name of the frame column (default: frame).
    --suffix        Output filename suffix (default: _arranged_boxes_).
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_overlap(box1, box2):
    """Calculate the intersection area between two bounding boxes in (L, T, W, H) format."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[0]) + float(box1[2]), float(box2[0]) + float(box2[2]))
    y2 = min(float(box1[1]) + float(box1[3]), float(box2[1]) + float(box2[3]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def separate_boxes(df, frame_col='frame',
                   body_cols=('cow_L', 'cow_T', 'cow_W', 'cow_H'),
                   head_cols=('head_L', 'head_T', 'head_W', 'head_H'),
                   snout_cols=('snout_L', 'snout_T', 'snout_W', 'snout_H')):
    """
    Reassign body/head/snout bounding boxes per frame using greedy overlap scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Detection DataFrame with one row per detection event.
    frame_col : str
        Name of the frame column.
    body_cols, head_cols, snout_cols : tuple of str
        Column names for each body part box (L, T, W, H).

    Returns
    -------
    pd.DataFrame
        Rearranged DataFrame — one row per animal per frame.
    """
    all_cols = [frame_col] + list(body_cols) + list(head_cols) + list(snout_cols)
    arranged_data = pd.DataFrame(columns=all_cols)

    df = df.copy()
    df['id'] = df.index

    body_df  = df[['id'] + list(body_cols)].values
    head_df  = df[['id'] + list(head_cols)].values
    snout_df = df[['id'] + list(snout_cols)].values

    for frame in tqdm(df[frame_col].unique(), desc="Processing frames"):
        frame_df = df[df[frame_col] == frame]

        body_boxes  = frame_df[['id'] + list(body_cols)].dropna().values
        head_boxes  = frame_df[['id'] + list(head_cols)].dropna().values
        snout_boxes = frame_df[['id'] + list(snout_cols)].dropna().values

        # --- Triplets ---
        tmp = []
        for b in body_boxes:
            for h in head_boxes:
                for s in snout_boxes:
                    score = (calculate_overlap(b[1:], h[1:]) +
                             calculate_overlap(b[1:], s[1:]) +
                             calculate_overlap(h[1:], s[1:]))
                    tmp.append([frame, b[0], h[0], s[0], score])
        tmp = sorted(tmp, key=lambda x: x[-1], reverse=True)

        best_triplets = []
        while tmp:
            best_triplets.append(tmp.pop(0))
            b_ids = [x[1] for x in best_triplets]
            h_ids = [x[2] for x in best_triplets]
            s_ids = [x[3] for x in best_triplets]
            tmp = [x for x in tmp if x[1] not in b_ids and x[2] not in h_ids and x[3] not in s_ids]

        used_b = [x[1] for x in best_triplets]
        used_h = [x[2] for x in best_triplets]
        used_s = [x[3] for x in best_triplets]
        body_boxes  = [b for b in body_boxes  if b[0] not in used_b]
        head_boxes  = [h for h in head_boxes  if h[0] not in used_h]
        snout_boxes = [s for s in snout_boxes if s[0] not in used_s]

        # --- Duets ---
        tmp = []
        for b in body_boxes:
            for h in head_boxes:
                tmp.append([frame, b[0], h[0], None, calculate_overlap(b[1:], h[1:])])
            for s in snout_boxes:
                tmp.append([frame, b[0], None, s[0], calculate_overlap(b[1:], s[1:])])
        tmp = sorted(tmp, key=lambda x: x[-1], reverse=True)

        best_duets = []
        while tmp:
            best_duets.append(tmp.pop(0))
            b_ids = [x[1] for x in best_duets]
            h_ids = [x[2] for x in best_duets if x[2] is not None]
            s_ids = [x[3] for x in best_duets if x[3] is not None]
            tmp = [x for x in tmp if x[1] not in b_ids and x[2] not in h_ids and x[3] not in s_ids]

        used_b = [x[1] for x in best_duets]
        used_h = [x[2] for x in best_duets if x[2] is not None]
        used_s = [x[3] for x in best_duets if x[3] is not None]
        body_boxes  = [b for b in body_boxes  if b[0] not in used_b]
        head_boxes  = [h for h in head_boxes  if h[0] not in used_h]
        snout_boxes = [s for s in snout_boxes if s[0] not in used_s]

        # --- Singletons ---
        best_singles = []
        for b in body_boxes:
            best_singles.append([frame, b[0], None, None, 0])

        # --- Reconstruct rows ---
        boxes_rearranged = []

        for triplet in best_triplets:
            bb = body_df[int(triplet[1])]
            hb = head_df[int(triplet[2])]
            sb = snout_df[int(triplet[3])]
            row = {frame_col: frame}
            for i, col in enumerate(body_cols):  row[col] = bb[i + 1]
            for i, col in enumerate(head_cols):  row[col] = hb[i + 1]
            for i, col in enumerate(snout_cols): row[col] = sb[i + 1]
            boxes_rearranged.append(row)

        for duet in best_duets:
            bb = body_df[int(duet[1])]
            row = {frame_col: frame}
            for i, col in enumerate(body_cols): row[col] = bb[i + 1]
            if duet[2] is not None:
                hb = head_df[int(duet[2])]
                for i, col in enumerate(head_cols):  row[col] = hb[i + 1]
                for col in snout_cols: row[col] = np.nan
            else:
                sb = snout_df[int(duet[3])]
                for col in head_cols: row[col] = np.nan
                for i, col in enumerate(snout_cols): row[col] = sb[i + 1]
            boxes_rearranged.append(row)

        for single in best_singles:
            bb = body_df[int(single[1])]
            row = {frame_col: frame}
            for i, col in enumerate(body_cols): row[col] = bb[i + 1]
            for col in head_cols + snout_cols: row[col] = np.nan
            boxes_rearranged.append(row)

        arranged_data = pd.concat([arranged_data, pd.DataFrame(boxes_rearranged)], ignore_index=True)

    return arranged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rearrange body-part bounding boxes per animal per frame.")
    parser.add_argument("--input_dir",  required=True, help="Folder of detection CSVs, or path to a single CSV.")
    parser.add_argument("--output_dir", required=True, help="Output folder for arranged CSVs.")
    parser.add_argument("--body_cols",  nargs=4, default=["cow_L", "cow_T", "cow_W", "cow_H"],  metavar=("L", "T", "W", "H"))
    parser.add_argument("--head_cols",  nargs=4, default=["head_L", "head_T", "head_W", "head_H"], metavar=("L", "T", "W", "H"))
    parser.add_argument("--snout_cols", nargs=4, default=["snout_L", "snout_T", "snout_W", "snout_H"], metavar=("L", "T", "W", "H"))
    parser.add_argument("--frame_col",  default="frame", help="Name of the frame column.")
    parser.add_argument("--suffix",     default="_arranged_boxes_", help="Output filename suffix.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input_dir):
        csv_files = [args.input_dir]
    else:
        csv_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".csv")]

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        df = pd.read_csv(csv_path).replace('none', np.nan).astype(float)

        arranged = separate_boxes(df,
                                  frame_col=args.frame_col,
                                  body_cols=tuple(args.body_cols),
                                  head_cols=tuple(args.head_cols),
                                  snout_cols=tuple(args.snout_cols))

        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}{args.suffix}.csv")
        arranged.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}  |  shape: {arranged.shape}")
