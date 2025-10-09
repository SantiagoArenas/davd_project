#!/usr/bin/env python3
"""
Recursively rename image files in folders to: foldername_number.ext

Usage:
  # dry-run (default)
  python3 scripts/rename_images.py --root src/data/Image_Library

  # actually apply renames
  python3 scripts/rename_images.py --root src/data/Image_Library --apply

The script preserves file extensions, zero-pads numbers based on files-per-folder,
and skips non-image files. It prints what it would do in dry-run mode.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def natural_sort_key(s: str):
    # split string into list of ints and strings for natural sort
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def collect_images(dirpath: str) -> List[str]:
    files = []
    for entry in os.listdir(dirpath):
        full = os.path.join(dirpath, entry)
        if os.path.isfile(full):
            _, ext = os.path.splitext(entry)
            if ext.lower() in IMAGE_EXTS:
                files.append(entry)
    files.sort(key=natural_sort_key)
    return files


def make_target_name(folder: str, idx: int, pad: int, ext: str) -> str:
    base = f"{folder}_{str(idx).zfill(pad)}"
    return base + ext


def rename_in_directory(dirpath: str, apply: bool) -> Tuple[int, int]:
    """Rename images in a single directory. Returns (renamed_count, skipped_count)."""
    images = collect_images(dirpath)
    if not images:
        return 0, 0

    pad = max(1, len(str(len(images))))
    folder = os.path.basename(dirpath.rstrip(os.sep))

    renamed = 0
    skipped = 0

    # Build a set of current filenames for collision checking
    current_set = set(os.listdir(dirpath))

    for i, name in enumerate(images, start=1):
        src = os.path.join(dirpath, name)
        _, ext = os.path.splitext(name)
        target_name = make_target_name(folder, i, pad, ext.lower())
        target = os.path.join(dirpath, target_name)

        if os.path.abspath(src) == os.path.abspath(target):
            skipped += 1
            continue

        # If target exists and it's not the same file, find an alternate to avoid overwriting
        if os.path.exists(target):
            # try to find a free slot by increasing a suffix counter
            alt_idx = 1
            alt_target = None
            while True:
                alt_name = f"{os.path.splitext(target_name)[0]}_{alt_idx}{ext.lower()}"
                alt_path = os.path.join(dirpath, alt_name)
                if not os.path.exists(alt_path):
                    alt_target = alt_path
                    break
                alt_idx += 1
            print(f"Collision: target {target_name} exists; will use {os.path.basename(alt_target)}")
            final_target = alt_target
        else:
            final_target = target

        if apply:
            os.rename(src, final_target)
            print(f"RENAMED: {name} -> {os.path.basename(final_target)}")
            renamed += 1
        else:
            print(f"DRY-RUN: {name} -> {os.path.basename(final_target)}")
            renamed += 0

    return renamed, skipped


def main():
    parser = argparse.ArgumentParser(description="Rename images to foldername_number.ext")
    parser.add_argument('--root', '-r', default='src/data/Image_Library', help='Root folder to process')
    parser.add_argument('--apply', action='store_true', help='Actually perform renames. Default is dry-run')
    parser.add_argument('--exts', nargs='*', help='Optional list of extensions to consider (e.g. .jpg .png)')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.exists(root):
        print(f"Root path does not exist: {root}")
        return

    global IMAGE_EXTS
    if args.exts:
        IMAGE_EXTS = set(e.lower() if e.startswith('.') else f'.{e.lower()}' for e in args.exts)

    total_dirs = 0
    total_renamed = 0
    total_skipped = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip the root itself if it contains many family folders; we want per-folder rename
        # But we still process every directory that contains image files.
        renamed, skipped = rename_in_directory(dirpath, args.apply)
        if renamed or skipped:
            total_dirs += 1
            total_renamed += renamed
            total_skipped += skipped

    print('\nSummary:')
    print(f'  Folders processed: {total_dirs}')
    print(f'  Files renamed (applied): {total_renamed}')
    print(f'  Files skipped (already correct): {total_skipped}')
    if not args.apply:
        print('\nDry-run only. Rerun with --apply to perform the renames.')


if __name__ == '__main__':
    main()
