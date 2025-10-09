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
from dataclasses import dataclass
from typing import Dict

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


def make_transforms(feature_extractor):
    size = feature_extractor.size if hasattr(feature_extractor, 'size') else feature_extractor.image_mean.shape[-1]

    def preprocess(example):
        img = Image.open(example['image_path']).convert('RGB')
        example['pixel_values'] = feature_extractor(images=img, return_tensors='pt')['pixel_values'][0]
        return example

    return preprocess


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
    args = parser.parse_args()

    # load CSVs via datasets
    data_files = {"train": args.train_csv, "validation": args.val_csv}
    ds = load_dataset('csv', data_files=data_files)

    # map label column to ClassLabel
    labels = sorted(set(ds['train']['label']))
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    def convert_label(example):
        example['label'] = label2id[example['label']]
        return example

    ds = ds.map(convert_label)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    preprocess = make_transforms(feature_extractor)

    ds = ds.with_transform(preprocess)

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        push_to_hub=False,
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Print a short summary
    print(f'Num labels: {num_labels}')
    print('Starting training...')
    trainer.train()


if __name__ == '__main__':
    main()
