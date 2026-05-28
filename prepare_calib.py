#!/usr/bin/env python3
"""
Generate raw float32 calibration files + input_list.txt for qnn-onnx-converter.

Usage:
  python3 prepare_calib.py --images "./coco2017/val2017/*.jpg" --num 100 --output ./calib_raw
"""
import numpy as np
import cv2
import glob
import os
import argparse

def generate(image_pattern, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(image_pattern))[:num_images]
    print(f"Found {len(paths)} images")

    raw_files = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0          # [0,1]
        img = np.transpose(img, (2, 0, 1))            # HWC → CHW
        img = np.expand_dims(img, axis=0)             # add batch dim

        raw_path = os.path.join(output_dir, f"calib_{i:04d}.raw")
        img.astype(np.float32).tofile(raw_path)
        raw_files.append(raw_path)

    # Write input_list.txt
    list_path = os.path.join(output_dir, "input_list.txt")
    with open(list_path, "w") as f:
        for r in raw_files:
            f.write(f"images:={os.path.abspath(r)}\n")

    print(f"Wrote {len(raw_files)} raw files + {list_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    generate(args.images, args.num, args.output)
