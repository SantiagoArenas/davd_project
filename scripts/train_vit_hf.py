#!/usr/bin/env python3
"""
Train a ViT image classifier using Hugging Face transformers and datasets.

This is a baseline training script with the initial hyperparameters discussed.

Usage (example):
  python3 scripts/train_vit_hf.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --output_dir models/vit_baseline \
    --epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --fp16

"""
from __future__ import annotations

import argparse
import os
from io import BytesIO
from dataclasses import dataclass
from typing import Dict
import sys
import textwrap

import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = (preds == labels).mean()
    return {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--model_name_or_path', default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--output_dir', default='models/vit_baseline')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use sqrt-smoothed inverse-frequency class weights')
    args = parser.parse_args()

    # Safety: require Python 3.10+ because recent transformers rely on PEP604 syntax
    if sys.version_info < (3, 10):
        msg = textwrap.dedent(f"""
        This script requires Python 3.10 or newer (current: {sys.version.split()[0]}).

        Options:
        - Install Python 3.10 from https://python.org and create a venv:
            /path/to/python3.10 -m venv .venv310
            source .venv310/bin/activate
            pip install -r requirements.txt

        - Or use your preferred Python installer. After activating a Python 3.10+ venv,
          re-run this script.
        """)
        print(msg)
        sys.exit(1)

    # Detect available device/backends
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if use_cuda:
        device_str = f'cuda (count={torch.cuda.device_count()})'
    elif use_mps:
        device_str = 'mps'
    else:
        device_str = 'cpu'

    # fp16 only supported on CUDA. If user passed --fp16 but no CUDA, warn and disable it.
    if args.fp16 and not use_cuda:
        print(f"--fp16 requested but CUDA not available (device={device_str}). Disabling fp16.")
        args.fp16 = False

    print(f'Environment: python={sys.version.split()[0]}, device={device_str}, fp16={args.fp16}')

    # load CSVs via datasets
    data_files = {"train": args.train_csv, "validation": args.val_csv}
    ds = load_dataset('csv', data_files=data_files)

    # Filter out invalid image files before processing
    def is_valid_image(example):
        """Check if image_path points to a valid image file"""
        path = example['image_path']
        if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            return False
        try:
            # Quick validation without loading full image
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    print("Filtering valid images...")
    ds = ds.filter(is_valid_image, desc="Filtering valid images")
    print(f"After filtering: {len(ds['train'])} train, {len(ds['validation'])} val images")

    # map label column to ClassLabel
    labels = sorted(set(ds['train']['label']))
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    def convert_label(example):
        example['label'] = label2id[example['label']]
        return example

    ds = ds.map(convert_label)

    # Optionally compute sqrt-smoothed inverse-frequency class weights BEFORE applying transforms
    class_weights = None
    if args.use_weighted_loss:
        from collections import Counter
        import math

        # Extract labels before with_transform (which would trigger image loading)
        train_labels = ds['train']['label']
        counts = Counter(train_labels)
        freqs = [counts[i] if i in counts else 0 for i in range(num_labels)]
        # sqrt smoothing: weight = 1 / sqrt(freq)
        weights = [1.0 / math.sqrt(c) if c > 0 else 1.0 for c in freqs]
        class_weights = torch.tensor(weights, dtype=torch.float)
        print('Using sqrt-smoothed class weights (first 10):', class_weights[:10])

    # Now load feature extractor and preprocess images
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    
    # Preprocess images eagerly using map() instead of with_transform()
    # This ensures image_path is available during preprocessing
    def preprocess_batch(examples):
        """Batch preprocess function - all images should be valid after filtering"""
        images = [Image.open(p).convert('RGB') for p in examples['image_path']]
        encoding = feature_extractor(images, return_tensors='pt')
        encoding['labels'] = examples['label']
        return encoding
    
    # Apply preprocessing with map, removing image_path column after processing
    print("Preprocessing images...")
    ds = ds.map(
        preprocess_batch,
        batched=True,
        batch_size=100,
        remove_columns=['image_path'],
        desc="Preprocessing images"
    )

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    # If using weighted loss, subclass Trainer to inject weighted CrossEntropyLoss
    if class_weights is not None:
        class WeightedTrainer(Trainer):
            def __init__(self, *args, class_weights=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get('label') if 'label' in inputs else inputs.get('labels')
                # move labels to device
                labels = labels.to(model.device)
                # prepare model inputs (remove labels if present)
                model_inputs = {k: v for k, v in inputs.items() if k != 'label' and k != 'labels'}
                outputs = model(**model_inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        TrainerClass = WeightedTrainer
    else:
        TrainerClass = Trainer

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=False,
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
    )

    data_collator = DefaultDataCollator()

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **({'class_weights': class_weights} if class_weights is not None else {}),
    )

    # Print a short summary
    print(f'Num labels: {num_labels}')
    print('Starting training...')
    trainer.train()


if __name__ == '__main__':
    main()
