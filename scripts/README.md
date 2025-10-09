This folder contains helper scripts for dataset prep and training.

- `prepare_dataset.py` - Create CSV train/val/test splits from a folder-per-class image dataset. Example:

  ```zsh
  python3 scripts/prepare_dataset.py --root src/data/Image_Library --out_dir data/splits
  ```

- `train_vit_hf.py` - Baseline training using Hugging Face `transformers` ViT models. Example:

  ```zsh
  python3 scripts/train_vit_hf.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --output_dir models/vit_baseline \
    --epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --fp16
  ```

Install dependencies (pick correct CUDA build for `torch`):

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# upgrade torch separately if you need a specific CUDA build
``` 
