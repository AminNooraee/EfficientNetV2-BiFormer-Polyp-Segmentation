# EfficientNetV2-BiFormer Polyp Segmentation

Official research code for a hybrid CNN-Transformer model for colonoscopy polyp segmentation.

The model uses:

- **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k`) as a pretrained multi-scale encoder.
- **Squeeze-and-Excitation (SE)** attention on decoder skip-fusion features.
- **BiFormer / Bi-Level Routing Attention** blocks in the decoder.
- **DropBlock** regularization in the decoder.
- Five-fold cross-validation on **Kvasir-SEG** and **CVC-ClinicDB**.

The public implementation was cleaned and reorganized from the experiment notebooks used for the manuscript.

## Reported results

| Dataset | Dice (%) | IoU (%) | Precision (%) | Recall (%) |
|---|---:|---:|---:|---:|
| Kvasir-SEG | 92.85 | 88.02 | 93.07 | 93.92 |
| CVC-ClinicDB | 95.02 | 92.83 | 96.24 | 96.15 |

The manuscript additionally reports Accuracy = 94.04% on CVC-ClinicDB.

## Architecture

The encoder is created with:

```python
timm.create_model(
    "tf_efficientnetv2_s.in21k",
    pretrained=True,
    features_only=True,
)
```

The classification head is therefore not used. Multi-scale feature maps returned by `timm` are passed to the decoder through skip connections. The internal EfficientNetV2-S backbone is not structurally modified and is fine-tuned end-to-end.

Each of the four main decoder stages follows the same high-level sequence:

```text
Transposed Convolution
        ↓
Skip Concatenation
        ↓
SE Channel Attention
        ↓
3×3 Conv + BatchNorm + ReLU
        ↓
DropBlock
        ↓
BiFormer Decoder Block
```

A final upsampling stage reconstructs the output to the original spatial resolution.

## Repository structure

```text
.
├── train.py
├── evaluate.py
├── benchmark.py
├── requirements.txt
├── models/
│   ├── model.py
│   ├── biformer_block.py
│   └── bra_legacy.py
├── data/
│   └── datasets.py
├── utils/
│   ├── losses.py
│   ├── metrics.py
│   ├── training.py
│   └── reproducibility.py
└── configs/
    ├── kvasir.json
    └── cvc_clinicdb.json
```

## Environment

The original Kaggle notebooks used **Python 3.12.10** and NVIDIA Tesla T4 GPUs. The exact package versions were not recorded in the notebooks, so `requirements.txt` lists the required packages without inventing version pins.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Datasets are not redistributed in this repository.

### Kvasir-SEG

Expected layout:

```text
data/Kvasir-SEG/
├── images/
│   ├── *.jpg
│   └── ...
└── masks/
    ├── *.jpg
    └── ...
```

### CVC-ClinicDB

Expected layout:

```text
data/CVC-ClinicDB/
├── Original/
│   ├── *.png
│   └── ...
└── Ground-Truth/
    ├── *.png
    └── ...
```

All images and masks are resized to **256×256**. RGB images are scaled to `[0, 1]`. Masks are binarized using a threshold of 127.

## Cross-validation protocol

For each dataset:

- 5-fold cross-validation is used.
- One fold (20%) is held out as test data.
- The remaining 80% is split into 90% training and 10% validation.
- The split seed is **42**.

This matches the data-splitting logic in the experiment notebooks.

## Training

Common settings:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Maximum epochs | 200 |
| Loss | 0.3 Cross-Entropy + 0.7 foreground Dice Loss |
| Input size | 256×256 |
| Model selection | Best validation Dice |

Early-stopping patience differs by dataset to match the experiments:

- Kvasir-SEG: `patience = 20`
- CVC-ClinicDB: `patience = 50`

Train one fold:

```bash
python train.py \
  --config configs/kvasir.json \
  --image-dir /path/to/Kvasir-SEG/images \
  --mask-dir /path/to/Kvasir-SEG/masks \
  --fold 0
```

Train all five folds:

```bash
python train.py \
  --config configs/kvasir.json \
  --image-dir /path/to/Kvasir-SEG/images \
  --mask-dir /path/to/Kvasir-SEG/masks \
  --all-folds
```

For CVC-ClinicDB:

```bash
python train.py \
  --config configs/cvc_clinicdb.json \
  --image-dir /path/to/CVC-ClinicDB/Original \
  --mask-dir /path/to/CVC-ClinicDB/Ground-Truth \
  --all-folds
```

## Evaluation

Evaluate a saved fold checkpoint using the corresponding split:

```bash
python evaluate.py \
  --config configs/kvasir.json \
  --image-dir /path/to/Kvasir-SEG/images \
  --mask-dir /path/to/Kvasir-SEG/masks \
  --checkpoint runs/kvasir/fold_0/best_dice.ckpt \
  --fold 0
```

The evaluation script reports:

- Dice
- IoU
- Precision
- Recall
- Accuracy
- Loss

## Complexity and inference speed

```bash
python benchmark.py --device cuda
```

The benchmark reports parameter count, THOP MACs/FLOPs-style operation count, latency, and FPS for a `1×3×256×256` input.

The manuscript reports approximately **20.82 M executed parameters** and **5.31 G operations** using THOP on the executed forward graph. THOP may not account for every custom attention operation, so complexity values should be interpreted consistently across model variants.

## Pretrained weights

Pretrained segmentation checkpoints will be added separately after the repository release. The EfficientNetV2-S encoder weights are downloaded through `timm`.

## BiFormer implementation

`models/bra_legacy.py` is the Bi-Level Routing Attention implementation used by the experiments. It originates from the official BiFormer project by Lei Zhu and retains its original attribution header. See `THIRD_PARTY_NOTICES.md`.

## Citation

A `CITATION.cff` file is included for citing this repository. The manuscript citation can be added after publication.

## License

This repository is released under the MIT License. Third-party code retains its original attribution and license notices.
