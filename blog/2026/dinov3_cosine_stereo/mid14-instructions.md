# Middlebury 2014 Evaluation + Code Refactor — Instructions for Claude Code

## Overview

We have an existing single-pair stereo comparison script (`stereo_comparison.py`, provided separately). We want to:

1. **Refactor** shared code (encoder loading, feature extraction, disparity matching, metrics) into reusable modules
2. **Create a new script** (`middlebury_eval.py`) that evaluates all encoders + block matching baselines on the Middlebury 2014 training set (15 pairs) at Quarter resolution
3. Keep the original single-pair script working — update it to import from the shared modules

## Dataset: Middlebury 2014 at Quarter resolution

- 15 training scenes with public ground truth
- Quarter resolution images are roughly 750×500 (varies per scene)
- Ground truth: `disp0.pfm` (float32 PFM) + non-occluded mask
- torchvision has a built-in loader:
  ```python
  from torchvision.datasets import Middlebury2014Stereo
  dataset = Middlebury2014Stereo(root="./data", split="train", download=True)
  # Returns (img_left, img_right, disparity, valid_mask)
  # disparity is numpy array shape (1, H, W), valid_mask is numpy bool array
  ```
- **IMPORTANT**: Check what torchvision returns for `valid_mask` — it may be `None` or a proper mask. If it doesn't provide the mask, load `mask0nocc.png` manually (pixels with value 255 are valid/non-occluded).
- **IMPORTANT**: Some GT disparity values may be `inf` or `0` — build a valid mask: `valid = np.isfinite(disp_gt) & (disp_gt > 0) & mask_nocc`
- **IMPORTANT**: Image sizes vary per scene — do not hardcode dimensions. Read size from each image.
- The Middlebury2014Stereo torchvision dataset may download images at a specific resolution. Check what resolution it gives you. If it doesn't support quarter resolution directly, you may need to download manually from `https://vision.middlebury.edu/stereo/data/scenes2014/` — each scene has `im0.png` and `im1.png` in `FullSize`, `HalfSize`, and `QuarterSize` subdirectories. Alternatively, download full and resize to quarter yourself.

## Metrics

Use **EPE** and **D1**:

```python
def compute_metrics(predicted, ground_truth, valid_mask=None):
    """
    Args:
        predicted: (H, W) disparity in pixels
        ground_truth: (H, W) disparity in pixels
        valid_mask: (H, W) bool — True where GT is valid
    Returns dict with: epe, d1
    """
    if valid_mask is not None:
        pred = predicted[valid_mask]
        gt = ground_truth[valid_mask]
    else:
        pred = predicted.ravel()
        gt = ground_truth.ravel()

    err = np.abs(pred - gt)

    epe = float(err.mean())

    # D1: percentage of pixels where error > max(3px, 5% of GT)
    threshold = np.maximum(3.0, 0.05 * np.abs(gt))
    d1 = float((err > threshold).mean() * 100.0)

    return {'epe': epe, 'd1': d1}
```

**Always evaluate only on valid (non-occluded, finite, positive GT) pixels.**

## File structure

```
stereo_experiment/
├── models.py                # Encoder loading and feature extraction
├── matching.py              # Disparity computation and metrics
├── stereo_comparison.py     # Original single-pair script (updated imports)
├── middlebury_eval.py       # NEW: Middlebury evaluation
├── block_matching.py        # Block matching implementations
├── im3.png, im4.png, groundtruth.png
├── data/                    # Middlebury downloads go here
├── croco/                   # CroCo repo
└── results/                 # Output JSON/CSV files
```

## Step 1: `models.py` — Encoder registry

Provide a clean interface for loading any encoder and extracting features.

```python
"""
Encoder registry for stereo matching experiments.

Usage:
    from models import load_encoder, extract_features, ENCODER_LIST

    model, meta = load_encoder("DINOv3 ViT-B", device="cuda")
    # meta = {'name': ..., 'training': ..., 'pos_embed': ..., 'patch_size': ..., 'feat_dim': ...}

    feat = extract_features(model, meta, img_tensor, hp, wp)
    # feat: (D, hp, wp) tensor on CPU
"""

ENCODER_LIST = [...]  # List of available encoder name strings

def load_encoder(name: str, device: str = "cpu") -> tuple[object, dict]:
    """Load a named encoder. Returns (model, metadata_dict)."""
    ...

def extract_features(model, meta: dict, img_tensor, hp: int, wp: int) -> torch.Tensor:
    """Extract (D, hp, wp) patch features from a (3, H, W) image tensor."""
    ...
```

Register all encoders:

| Name | timm model string | Training | Pos Embed | Patch Size | Notes |
|------|-------------------|----------|-----------|------------|-------|
| DINOv3 ViT-B | `vit_base_patch16_dinov3.lvd1689m` | DINO v3 | RoPE | 16 | No `dynamic_img_size` needed |
| DINOv2 ViT-B | `vit_base_patch14_dinov2.lvd142m` | DINO v2 | Learned | **14** | `dynamic_img_size=True` |
| DINOv1 ViT-B | `vit_base_patch16_224.dino` | DINO v1 | Learned | 16 | `dynamic_img_size=True` |
| MAE ViT-B | `vit_base_patch16_224.mae` | MAE | Learned | 16 | `dynamic_img_size=True` |
| CLIP ViT-B | `vit_base_patch16_clip_224.openai` | CLIP | Learned | 16 | `dynamic_img_size=True` |
| Supervised ViT-B | `vit_base_patch16_224.augreg_in21k_ft_in1k` | Supervised | Learned | 16 | `dynamic_img_size=True` |
| CroCo v2 ViT-B | Custom loading | CroCo v2 | RoPE | 16 | See existing script for loading code |
| Random ViT-B | `vit_base_patch16_dinov3.lvd1689m` with `pretrained=False` | Random | RoPE | 16 | Untrained baseline |

Key implementation notes (from existing working script):
- All timm models use `num_classes=0`
- Token stripping: use `out[:, -n:, :]` where `n = hp * wp` (works for all models)
- CroCo v2: clone repo, download checkpoint, monkey-patch PatchEmbed to remove size assertion, build positions manually — **copy exactly from the working code in the existing script**
- Each encoder's extract function must handle its own token layout (CLS tokens, register tokens, etc.)

## Step 2: `matching.py` — Disparity and metrics

```python
"""Scanline matching and evaluation metrics."""

def prepare_image(image, scale, patch_size, orig_h, orig_w):
    """Resize to scale × original, snapped to patch_size multiples.
    Returns (img_tensor, h_patches, w_patches, actual_h, actual_w)"""
    ...

def compute_disparity_scanline(feat_l, feat_r):
    """Cosine similarity scanline matching with non-negative disparity.
    feat_l, feat_r: (D, H, W)
    Returns (disparity, confidence) in patch units, both (H, W)"""
    ...

def disp_patches_to_pixels(disp_patches, patch_size, actual_scale, orig_h, orig_w):
    """Convert patch-unit disparity to original pixels, nearest-neighbor upsample.
    actual_scale = actual_image_height / orig_h
    Returns (disp_orig_px_grid, disp_full_res)"""
    ...

def compute_metrics(predicted, ground_truth, valid_mask=None):
    """Compute EPE and D1 on valid pixels only."""
    ...
```

## Step 3: `block_matching.py` — Pixel baselines

Two variants:

### Strided block matching (ViT-equivalent)
- Block sizes: 16, 8, 5, 4
- Stride = block_size (non-overlapping)
- Cosine similarity on flattened grayscale pixel blocks
- Same scanline + non-negative disparity constraint
- Same nearest-neighbor upsampling
- These correspond to ViT at 1x, 2x, ~3x, 4x

### Dense block matching (stride 1)
- Block sizes: 3, 5, 7, 9
- Stride = 1, SAD cost
- max_disp: set per scene. Use the max GT disparity for the scene + some margin, or cap at a reasonable value like `min(W // 3, 300)`
- Per-pixel output, no upsampling needed

Both should expose a function with this interface:
```python
def block_match_strided(left_gray, right_gray, block_size) -> np.ndarray:
    """Returns (H_blocks, W_blocks) disparity in pixel units."""
    ...

def block_match_dense(left_gray, right_gray, block_size, max_disp) -> np.ndarray:
    """Returns (H, W) disparity in pixel units."""
    ...
```

The strided version needs the same `disp_patches_to_pixels` upsampling. The dense version is already at full resolution.

## Step 4: `middlebury_eval.py` — Main evaluation script

### Structure

```python
"""
Evaluate stereo matching encoders on Middlebury 2014 Quarter-resolution training set.

Usage:
    python middlebury_eval.py
    python middlebury_eval.py --encoders "DINOv3 ViT-B" "MAE ViT-B"
    python middlebury_eval.py --scales 1 2 3 4
    python middlebury_eval.py --block-matching-only
    python middlebury_eval.py --device cuda
"""

import argparse
from models import load_encoder, extract_features, ENCODER_LIST
from matching import prepare_image, compute_disparity_scanline, disp_patches_to_pixels, compute_metrics
from block_matching import block_match_strided, block_match_dense
```

### Evaluation loop

```
For each encoder:
    Load model once
    For each scale in [1, 2, 3, 4]:
        For each of 15 Middlebury training scenes:
            - Get image dimensions for this scene
            - prepare_image at this scale
            - extract_features for left and right
            - compute_disparity_scanline
            - disp_patches_to_pixels (convert + upsample)
            - compute_metrics against GT with valid_mask
            - Store per-scene EPE and D1
            - Free tensors
        Compute mean EPE and mean D1 across 15 scenes
    Print per-scale summary for this encoder

For strided block matching:
    For each block_size in [16, 8, 5, 4]:
        For each scene:
            - Convert to grayscale
            - block_match_strided
            - Upsample to full res
            - compute_metrics
        Mean across scenes

For dense block matching:
    For each block_size in [3, 5, 7, 9]:
        For each scene:
            - Convert to grayscale
            - block_match_dense (set max_disp per scene)
            - compute_metrics (already full res)
        Mean across scenes
```

### Memory management
- Delete feature tensors and disparity maps after computing metrics for each scene
- Don't store full-resolution arrays across scenes
- Only accumulate the scalar metrics (EPE, D1) per scene

### Output

**1. Per-encoder table printed to console:**
```
DINOv3 ViT-B (DINO v3, RoPE, patch_size=16)
  Scale │  EPE  │  D1%
  ──────┼───────┼──────
    1x  │ XX.XX │ XX.X
    2x  │ XX.XX │ XX.X
    3x  │ XX.XX │ XX.X
    4x  │ XX.XX │ XX.X
```

**2. Final combined table (at best scale per encoder):**
```
Encoder            │ Training    │ Pos   │ Best │  EPE  │  D1%
───────────────────┼─────────────┼───────┼──────┼───────┼──────
DINOv3 ViT-B       │ DINO v3     │ RoPE  │  3x  │ XX.XX │ XX.X
DINOv1 ViT-B       │ DINO v1     │ Learn │  3x  │ XX.XX │ XX.X
MAE ViT-B          │ MAE         │ Learn │  3x  │ XX.XX │ XX.X
...                │             │       │      │       │
Block (stride=bs)  │ Pixels      │  N/A  │ bs=4 │ XX.XX │ XX.X
Block (dense)      │ Pixels      │  N/A  │ bs=7 │ XX.XX │ XX.X
```

**3. Save raw results to JSON:**
```python
results = {
    "encoder_name": {
        "meta": {"training": ..., "pos_embed": ..., "patch_size": ...},
        "scales": {
            "1": {
                "mean_epe": ..., "mean_d1": ...,
                "per_scene": {
                    "Adirondack": {"epe": ..., "d1": ...},
                    "Jadeplant": {"epe": ..., "d1": ...},
                    ...
                }
            },
            ...
        }
    },
    ...
}
# Save to results/middlebury_results.json
```

Per-scene results are important — they let us later analyze which scenes benefit most from learned features (likely textureless/repetitive) vs where block matching wins (fine detail, large textures).

**4. No plots needed** — we'll do plotting separately later.

### CLI

```python
parser = argparse.ArgumentParser()
parser.add_argument("--encoders", nargs="*", default=None,
                    help="Encoder names to run. Default: all")
parser.add_argument("--scales", nargs="*", type=int, default=[1, 2, 3, 4])
parser.add_argument("--device", default="cpu")
parser.add_argument("--block-matching-only", action="store_true")
parser.add_argument("--skip-block-matching", action="store_true")
parser.add_argument("--data-root", default="./data")
parser.add_argument("--output", default="./results/middlebury_results.json")
```

## Step 5: Update `stereo_comparison.py`

Update the original script to import from `models.py` and `matching.py`. It should still:
- Work on the single pair (im3.png, im4.png, groundtruth.png with /16 scaling)
- Produce all its plots
- Just with less duplicated code

## Things to watch out for

1. **Image sizes vary per Middlebury scene** — never hardcode dimensions
2. **DINOv2 patch_size=14** — resize math and disparity conversion must use correct patch size. Image dimensions must be divisible by 14 for DINOv2, by 16 for everything else.
3. **Valid mask** — always evaluate only on valid pixels. Build mask from: finite GT, positive GT, non-occluded
4. **PFM loading** — torchvision should handle this, but verify the values look reasonable (float32, range typically 0–300 px)
5. **Disparity upsampling** — use nearest-neighbor, not bilinear
6. **Dense block matching max_disp** — set per scene based on GT range. Something like `int(np.nanmax(disp_gt[valid]) * 1.2)` or cap at a reasonable maximum
7. **CroCo v2** — copy the working loading/extraction code exactly from the existing script. Don't try to simplify it.
8. **The torchvision Middlebury2014Stereo dataset** — it may download at a specific resolution. Check what resolution is returned and verify it's quarter. If not, the `__init__` may accept a `resolution` or `size` parameter, or you may need to download manually and write a simple loader.
9. **Scene names** — include them in the per-scene results. The 15 training scenes include: Adirondack, ArtL, Jadeplant, Motorcycle, MotorcycleE, Piano, PianoL, Pipes, Playroom, Playtable, PlaytableP, Recycle, Shelves, Teddy, Vintage (names may vary slightly — check what torchvision returns or what the directory names are)
