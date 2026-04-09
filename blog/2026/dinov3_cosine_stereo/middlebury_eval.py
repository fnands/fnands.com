#!/usr/bin/env python3
"""
Evaluate stereo matching encoders on Middlebury 2014 Quarter-resolution training set.

Usage:
    uv run middlebury_eval.py
    uv run middlebury_eval.py --encoders "DINOv3 ViT-B" "MAE ViT-B"
    uv run middlebury_eval.py --scales 1 2 3 4
    uv run middlebury_eval.py --device cpu
    uv run middlebury_eval.py --block-matching-only
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from encoders import load_encoder, extract_features, ENCODER_LIST, ENCODER_META_STATIC
from matching import (prepare_image, compute_disparity_scanline,
                      disp_patches_to_pixels, upsample_nearest, compute_metrics)
from block_matching import block_match_strided, block_match_dense

warnings.filterwarnings("ignore")

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

UPSCALE_FACTORS     = [1, 2, 3, 4]
STRIDED_BLOCK_SIZES = [16, 8, 5, 4]
DENSE_BLOCK_SIZES   = [3, 5, 7, 9]


# ── Dataset loading ────────────────────────────────────────────────────────

def _try_torchvision_dataset(data_root: Path):
    """Load Middlebury 2014 training set via torchvision (full resolution)."""
    from torchvision.datasets import Middlebury2014Stereo  # noqa: PLC0415
    ds = Middlebury2014Stereo(root=str(data_root), split="train",
                               calibration="perfect", download=True)
    print(f"  torchvision Middlebury2014Stereo loaded ({len(ds)} scenes, full res)")
    return ds


def _get_scene_name(dataset, idx: int) -> str:
    """Extract scene name from the dataset's internal file list."""
    try:
        left_path = dataset._images[idx][0]
        return Path(left_path).parent.name
    except (AttributeError, IndexError, TypeError):
        return f"scene_{idx:02d}"


def _nocc_mask_path(dataset, idx: int) -> Path | None:
    """Try to find the non-occlusion mask file for a given scene."""
    try:
        left_path = Path(dataset._images[idx][0])
        candidate = left_path.parent / "mask0nocc.png"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return None


QUARTER_DOWNSAMPLE = 4  # resize full-res images to 1/4 for evaluation


def _resize_to_quarter(img_left, img_right, disp_gt, valid):
    """Resize full-resolution data to quarter resolution."""
    import torch.nn.functional as F  # noqa: PLC0415
    orig_w, orig_h = img_left.size    # PIL: (W, H)
    q_w = orig_w // QUARTER_DOWNSAMPLE
    q_h = orig_h // QUARTER_DOWNSAMPLE

    img_l_q = img_left.resize((q_w, q_h), Image.BICUBIC)
    img_r_q = img_right.resize((q_w, q_h), Image.BICUBIC)

    # Scale disparity by 1/QUARTER_DOWNSAMPLE (pixel units scale with image width)
    disp_t = torch.from_numpy(disp_gt)[None, None].float()
    disp_q = F.interpolate(disp_t, size=(q_h, q_w), mode="nearest").squeeze().numpy()
    disp_q = disp_q / QUARTER_DOWNSAMPLE

    valid_t = torch.from_numpy(valid.astype(np.uint8))[None, None].float()
    valid_q = F.interpolate(valid_t, size=(q_h, q_w), mode="nearest").squeeze().numpy() > 0.5

    return img_l_q, img_r_q, disp_q, valid_q


def load_scene(dataset, idx: int):
    """
    Load one stereo scene, resize to quarter resolution.
    Returns (img_left, img_right, disp_gt, valid_mask, scene_name).
    disp_gt: float32 (H, W) at quarter resolution
    valid_mask: bool (H, W)
    """
    data = dataset[idx]
    img_left, img_right = data[0], data[1]

    # Disparity — torchvision returns (1, H, W) numpy
    raw_disp = data[2]
    if raw_disp is None:
        raise ValueError(f"No GT disparity for scene {idx}")
    disp_gt = np.asarray(raw_disp, dtype=np.float32)
    if disp_gt.ndim == 3:
        disp_gt = disp_gt[0]

    # Build valid mask: finite, positive
    valid = np.isfinite(disp_gt) & (disp_gt > 0)

    # Non-occlusion mask from torchvision (may be all-True if not available)
    raw_mask = data[3] if len(data) > 3 else None
    if raw_mask is not None:
        m = np.asarray(raw_mask, dtype=bool)
        if m.ndim == 3:
            m = m[0]
        # Only apply if it actually masks some pixels (not a trivial all-True mask)
        if not m.all():
            valid = valid & m

    # Try mask0nocc.png manually if torchvision didn't provide one
    nocc_path = _nocc_mask_path(dataset, idx)
    if nocc_path is not None:
        nocc = np.array(Image.open(nocc_path))
        valid = valid & (nocc == 255)

    scene_name = _get_scene_name(dataset, idx)

    # Resize full-resolution data to quarter resolution
    img_left, img_right, disp_gt, valid = _resize_to_quarter(
        img_left, img_right, disp_gt, valid)

    return img_left, img_right, disp_gt, valid, scene_name


# ── Encoder evaluation ─────────────────────────────────────────────────────

def eval_encoder_scene(model, meta, img_left, img_right,
                        disp_gt, valid_mask, scale: int):
    """Evaluate one encoder on one scene at one scale. Returns metrics dict."""
    orig_h, orig_w = img_left.size[1], img_left.size[0]   # PIL: (W, H)
    patch_size     = meta["patch_size"]

    img_l_t, hp, wp, new_h, _ = prepare_image(
        img_left,  scale, patch_size, orig_h, orig_w)
    img_r_t, _,  _,  _,    _ = prepare_image(
        img_right, scale, patch_size, orig_h, orig_w)
    actual_scale = new_h / orig_h

    feat_l = extract_features(model, meta, img_l_t, hp, wp)
    feat_r = extract_features(model, meta, img_r_t, hp, wp)

    disp_patches, _ = compute_disparity_scanline(feat_l, feat_r)
    _, disp_full    = disp_patches_to_pixels(
        disp_patches.numpy(), patch_size, actual_scale, orig_h, orig_w)

    metrics = compute_metrics(disp_full, disp_gt, valid_mask)

    # Clean up
    del feat_l, feat_r, disp_patches
    return metrics


def eval_encoder(name: str, model, meta, dataset, scales: list) -> dict:
    """Evaluate an encoder at multiple scales across all scenes."""
    print(f"\n{'='*65}")
    print(f"  {name}  (patch_size={meta['patch_size']})")
    print(f"{'='*65}")

    results = {"meta": {k: v for k, v in meta.items() if not k.startswith("_")},
               "scales": {}}

    for scale in scales:
        scene_results = {}
        epes, d1s = [], []
        t0 = time.time()

        for idx in range(len(dataset)):
            try:
                img_l, img_r, disp_gt, valid, scene_name = load_scene(dataset, idx)
            except Exception as e:
                print(f"    skip scene {idx}: {e}")
                continue

            m = eval_encoder_scene(model, meta, img_l, img_r, disp_gt, valid, scale)
            scene_results[scene_name] = m
            epes.append(m["epe"])
            d1s.append(m["d1"])

        mean_epe = float(np.mean(epes)) if epes else float("nan")
        mean_d1  = float(np.mean(d1s))  if d1s  else float("nan")
        elapsed  = time.time() - t0

        results["scales"][str(scale)] = {
            "mean_epe": mean_epe, "mean_d1": mean_d1, "per_scene": scene_results
        }
        print(f"  {scale}x  EPE={mean_epe:.2f}  D1={mean_d1:.1f}%  ({elapsed:.1f}s)", flush=True)

    # Print per-encoder summary
    print(f"\n  {'Scale':>6} │  {'EPE':>6}  │  {'D1%':>6}")
    print(f"  {'──────':>6}─┼──{'──────':>6}──┼──{'──────':>6}")
    for scale in scales:
        r = results["scales"][str(scale)]
        print(f"  {scale}x{' ':4} │  {r['mean_epe']:>6.2f}  │  {r['mean_d1']:>5.1f}")

    return results


# ── Block matching evaluation ──────────────────────────────────────────────

def eval_strided_scene(img_left, img_right, disp_gt, valid_mask,
                        block_size: int):
    orig_h, orig_w = img_left.size[1], img_left.size[0]
    l_gray = np.array(img_left.convert("L")).astype(np.float32) / 255.0
    r_gray = np.array(img_right.convert("L")).astype(np.float32) / 255.0

    disp_blocks = block_match_strided(l_gray, r_gray, block_size)
    disp_full   = upsample_nearest(disp_blocks, orig_h, orig_w)
    return compute_metrics(disp_full, disp_gt, valid_mask)


def eval_dense_scene(img_left, img_right, disp_gt, valid_mask,
                     block_size: int):
    orig_h, orig_w = img_left.size[1], img_left.size[0]
    l_gray = np.array(img_left.convert("L")).astype(np.float32) / 255.0
    r_gray = np.array(img_right.convert("L")).astype(np.float32) / 255.0

    valid_disp = disp_gt[valid_mask] if valid_mask.any() else disp_gt.ravel()
    max_disp   = min(int(np.nanmax(valid_disp) * 1.2) + 10, orig_w // 3, 300)

    disp_full = block_match_dense(l_gray, r_gray, block_size, max_disp)
    return compute_metrics(disp_full, disp_gt, valid_mask)


def eval_block_matching(dataset) -> dict:
    """Evaluate all block matching variants."""
    results = {"strided": {}, "dense": {}}

    print(f"\n{'='*65}")
    print("  Block Matching — Strided (ViT-equivalent)")
    print(f"{'='*65}")
    for bs in STRIDED_BLOCK_SIZES:
        epes, d1s = [], []
        scene_results = {}
        t0 = time.time()
        for idx in range(len(dataset)):
            try:
                img_l, img_r, disp_gt, valid, name = load_scene(dataset, idx)
            except Exception:
                continue
            m = eval_strided_scene(img_l, img_r, disp_gt, valid, bs)
            scene_results[name] = m
            epes.append(m["epe"]); d1s.append(m["d1"])
        elapsed = time.time() - t0
        mean_epe = float(np.mean(epes)); mean_d1 = float(np.mean(d1s))
        results["strided"][str(bs)] = {"mean_epe": mean_epe, "mean_d1": mean_d1,
                                        "per_scene": scene_results}
        print(f"  bs={bs:2d}  EPE={mean_epe:.2f}  D1={mean_d1:.1f}%  ({elapsed:.1f}s)")

    print(f"\n{'='*65}")
    print("  Block Matching — Dense (stride 1, SAD)")
    print(f"{'='*65}")
    for bs in DENSE_BLOCK_SIZES:
        epes, d1s = [], []
        scene_results = {}
        t0 = time.time()
        for idx in range(len(dataset)):
            try:
                img_l, img_r, disp_gt, valid, name = load_scene(dataset, idx)
            except Exception:
                continue
            m = eval_dense_scene(img_l, img_r, disp_gt, valid, bs)
            scene_results[name] = m
            epes.append(m["epe"]); d1s.append(m["d1"])
        elapsed = time.time() - t0
        mean_epe = float(np.mean(epes)); mean_d1 = float(np.mean(d1s))
        results["dense"][str(bs)] = {"mean_epe": mean_epe, "mean_d1": mean_d1,
                                      "per_scene": scene_results}
        print(f"  bs={bs:2d}  EPE={mean_epe:.2f}  D1={mean_d1:.1f}%  ({elapsed:.1f}s)")

    return results


# ── Summary printing ──────────────────────────────────────────────────────

def print_summary(all_enc_results: dict, bm_results: dict, scales: list):
    """Print final combined table at best scale per encoder."""
    print(f"\n{'='*75}")
    print("  Final Summary (best scale per encoder)")
    print(f"{'='*75}")
    hdr = f"  {'Encoder':<22} │ {'Training':<11} │ {'Pos':<6} │ {'Best':>4} │ {'EPE':>6} │ {'D1%':>6}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for name, res in all_enc_results.items():
        meta = res["meta"]
        best_scale = min(
            scales,
            key=lambda s: res["scales"][str(s)]["mean_epe"]
        )
        r = res["scales"][str(best_scale)]
        print(f"  {name:<22} │ {meta['training']:<11} │ {meta['pos_embed']:<6} │ "
              f"{best_scale}x   │ {r['mean_epe']:>6.2f} │ {r['mean_d1']:>5.1f}")

    if bm_results:
        print("  " + "─" * (len(hdr) - 2))
        # Best strided block size
        best_bs  = min(STRIDED_BLOCK_SIZES,
                       key=lambda b: bm_results["strided"][str(b)]["mean_epe"])
        r = bm_results["strided"][str(best_bs)]
        print(f"  {'Block strided':<22} │ {'Pixels':<11} │ {'N/A':<6} │ "
              f"bs={best_bs} │ {r['mean_epe']:>6.2f} │ {r['mean_d1']:>5.1f}")
        # Best dense block size
        best_bs  = min(DENSE_BLOCK_SIZES,
                       key=lambda b: bm_results["dense"][str(b)]["mean_epe"])
        r = bm_results["dense"][str(best_bs)]
        print(f"  {'Block dense':<22} │ {'Pixels':<11} │ {'N/A':<6} │ "
              f"bs={best_bs} │ {r['mean_epe']:>6.2f} │ {r['mean_d1']:>5.1f}")


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="*", default=None,
                        help="Encoder names to run. Default: all")
    parser.add_argument("--scales", nargs="*", type=int, default=UPSCALE_FACTORS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-matching-only", action="store_true")
    parser.add_argument("--skip-block-matching", action="store_true")
    parser.add_argument("--data-root", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "middlebury_results.json"))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoders_to_run = args.encoders if args.encoders else ENCODER_LIST
    scales          = args.scales

    print(f"Device: {args.device}")
    print(f"Encoders: {encoders_to_run}")
    print(f"Scales: {scales}")
    print(f"Data root: {data_root}")

    # Load dataset
    print("\nLoading Middlebury 2014 dataset …")
    try:
        dataset = _try_torchvision_dataset(data_root)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        raise

    n_scenes = len(dataset)
    print(f"  {n_scenes} scenes found.")

    # Verify by peeking at scene 0
    try:
        img_l, img_r, dgt, vmask, sname = load_scene(dataset, 0)
        orig_h, orig_w = img_l.size[1], img_l.size[0]
        print(f"  First scene: '{sname}'  size={orig_w}×{orig_h}  "
              f"valid_px={vmask.sum()}/{vmask.size}  "
              f"disp_range=[{dgt[vmask].min():.1f}, {dgt[vmask].max():.1f}]")
    except Exception as e:
        print(f"  WARNING: could not peek at scene 0: {e}")

    # Load existing results for resuming
    save_data: dict = {"encoders": {}, "block_matching": {},
                       "config": {"scales": scales, "device": args.device,
                                  "n_scenes": n_scenes}}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            save_data["encoders"]       = existing.get("encoders", {})
            save_data["block_matching"] = existing.get("block_matching", {})
            print(f"  Loaded existing results: {list(save_data['encoders'].keys())}")
        except Exception:
            pass

    def _save():
        out_path.write_text(json.dumps(save_data, indent=2))
        print(f"  → Saved to {out_path.name}", flush=True)

    all_results: dict = save_data["encoders"]
    bm_results: dict  = save_data["block_matching"]

    # Encoder evaluation
    if not args.block_matching_only:
        for name in encoders_to_run:
            if name in all_results:
                print(f"\nSkipping {name} (already in results)", flush=True)
                continue
            print(f"\nLoading {name} …", flush=True)
            try:
                model, meta = load_encoder(name, device=args.device)
            except Exception as e:
                print(f"  ERROR loading {name}: {e}", flush=True)
                continue

            all_results[name] = eval_encoder(name, model, meta, dataset, scales)

            # Save after each encoder
            _save()

            # Free model memory
            del model
            torch.cuda.empty_cache() if args.device != "cpu" else None

    # Block matching evaluation
    if not args.skip_block_matching and not args.encoders:
        bm_results.update(eval_block_matching(dataset))
        _save()

    # Summary
    if all_results or bm_results:
        print_summary(all_results, bm_results if bm_results else None, scales)

    _save()
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
