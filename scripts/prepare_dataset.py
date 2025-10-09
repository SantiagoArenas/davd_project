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


def stratified_split(pairs: List[Tuple[str, str]], train_frac: float, val_frac: float, test_frac: float, seed: int = 42):
    # Group by label
    by_label = defaultdict(list)
    for path, label in pairs:
        by_label[label].append(path)

    train, val, test = [], [], []
    for label, paths in by_label.items():
        if len(paths) < 3:
            # small class: put 60/20/20 by rounding
            t, v, te = [], [], []
            for i, p in enumerate(paths):
                if i % 5 < 3:
                    t.append(p)
                elif i % 5 == 3:
                    v.append(p)
                else:
                    te.append(p)
        else:
            p_train = train_frac / (train_frac + val_frac + test_frac)
            p_temp = 1 - p_train
            t, temp = train_test_split(paths, train_size=p_train, random_state=seed, shuffle=True)
            # split temp into val/test
            v_size = val_frac / (val_frac + test_frac)
            v, te = train_test_split(temp, train_size=v_size, random_state=seed, shuffle=True)

        train += [(p, label) for p in t]
        val += [(p, label) for p in v]
        test += [(p, label) for p in te]

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

    write_csv(train, os.path.join(args.out_dir, 'train.csv'))
    write_csv(val, os.path.join(args.out_dir, 'val.csv'))
    write_csv(test, os.path.join(args.out_dir, 'test.csv'))

    print(f'Wrote {len(train)} train, {len(val)} val, {len(test)} test to {args.out_dir}')


if __name__ == '__main__':
    main()
