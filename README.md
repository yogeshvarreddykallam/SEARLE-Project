# SEARLE — Zero-Shot Composed Image Retrieval

Fork and fine-tuning of **SEARLE / iSEARLE** for PSU CSE 597. Replicates and extends the [ICCV 2023 paper](https://arxiv.org/abs/2303.15247) *"Zero-Shot Composed Image Retrieval with Textual Inversion"* and its [extended version](https://arxiv.org/abs/2405.02951).

> **Original repo:** [miccunifi/SEARLE](https://github.com/miccunifi/SEARLE)

## What is Composed Image Retrieval?

Given a **reference image** and a **relative text caption** (e.g., *"same scene but at night"*), retrieve the target image from a gallery. Standard CIR requires expensive labeled triplets; SEARLE eliminates that requirement.

## How SEARLE Works

1. **Pre-training (OTI):** For each unlabeled image, optimize a pseudo-word token *S\** in CLIP's token embedding space that best represents the image
2. **Distillation:** Train a φ-network to predict pseudo-word tokens directly from image features (no optimization at inference time)
3. **Inference:** Encode the reference image → pseudo-token via φ → concatenate with relative caption → retrieve via CLIP text encoder

![Method Overview](assets/intro.png)

## Repository Structure

```
.
├── src/
│   ├── phi.py                          # φ-network architecture
│   ├── train_phi.py                    # Training loop
│   ├── oti_inversion.py                # Optimization-based textual inversion
│   ├── encode_with_pseudo_tokens.py    # Inference-time encoding
│   ├── validate.py                     # Recall@K evaluation
│   ├── generate_test_submission.py     # Submission generation
│   ├── datasets.py / data_utils.py     # FashionIQ, CIRR, CIRCO loaders
│   └── utils.py
├── hubconf.py                          # torch.hub entry point
├── assets/
│   ├── intro.png
│   ├── Poster.pdf
│   └── Slides.pptx
└── data/
    └── oidv7-class-descriptions.csv
```

## Setup

```bash
pip install torch torchvision open-clip-torch
```

Load pre-trained models:
```python
import torch
model = torch.hub.load('.', 'searle_model', source='local')
```

## Training

```bash
python src/train_phi.py   --dataset fashioniq   --clip-model-name ViT-B/32   --batch-size 64   --epochs 100
```

## Evaluation

```bash
python src/validate.py --dataset fashioniq --split val
python src/validate.py --dataset circo
```

## Citation

```bibtex
@inproceedings{baldrati2023zero,
  title={Zero-Shot Composed Image Retrieval with Textual Inversion},
  author={Baldrati, Alberto and others},
  booktitle={ICCV},
  year={2023}
}
```
