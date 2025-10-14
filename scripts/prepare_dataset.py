#!/usr/bin/env python3
"""
Prepare dataset CSV splits from a directory organized with one folder per class.

Usage:
  python3 scripts/prepare_dataset.py --root src/data/Image_Library --out_dir data/splits --train_frac 0.8 --val_frac 0.1 --test_frac 0.1

Produces CSV files: train.csv, val.csv, test.csv with columns: image_path,label
"""
from __future__ import annotations

import argparse
import csv
from ctypes.util import test
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def gather_images(root: str) -> List[Tuple[str, str]]:
    root = Path(root)
    pairs = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        label = class_dir.name
        for p in class_dir.rglob('*'):
            if p.is_file():
                pairs.append((str(p), label))
    return pairs


def stratified_split(pairs: List[Tuple[str,str]],
                     train_frac: float, val_frac: float, test_frac: float,
                     seed: int = 0):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    by_cls = defaultdict(list)
    for path, cls in pairs:
        by_cls[cls].append((path, cls))

    rng = random.Random(seed)
    train, val, test = [], [], []

    for cls, items in by_cls.items():
        n = len(items)
        rng.shuffle(items)

        if n == 1:
            # keep single sample in train so class is represented
            train.extend(items)
            continue

        n_train = max(1, int(round(n * train_frac)))
        if n_train >= n:
            n_train = n - 1  # reserve at least one for val/test if possible
        r = n - n_train
        # allocate remaining proportionally between val/test
        if val_frac + test_frac == 0:
            n_val = 0
        else:
            n_val = int(round(r * (val_frac / (val_frac + test_frac))))
        n_test = r - n_val

        # slice
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def write_csv(pairs: List[Tuple[str, str]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        for p, label in pairs:
            writer.writerow([p, label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='src/data/Image_Library')
    parser.add_argument('--out_dir', default='data/splits')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pairs = gather_images(args.root)
    print(f'Found {len(pairs)} images across {len(set([l for _, l in pairs]))} classes')

    train, val, test = stratified_split(pairs, args.train_frac, args.val_frac, args.test_frac, seed=args.seed)

    # after stratified_split:
    def write_split(pairs, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])  # header
            writer.writerows(pairs)  # each pair is (image_path, label)

        write_split(train, 'data/splits/train.csv')
        write_split(val, 'data/splits/val.csv')
        write_split(test, 'data/splits/test.csv')

    print(f'Wrote {len(train)} train, {len(val)} val, {len(test)} test to {args.out_dir}')


if __name__ == '__main__':
    main()
