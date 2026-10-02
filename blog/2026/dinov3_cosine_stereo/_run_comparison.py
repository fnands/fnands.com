# %% [markdown]
# # Stereo Disparity Encoder Comparison
#
# Compare pretrained ViT encoders on stereo disparity estimation via
# scanline cosine similarity matching. Each encoder's patch features are
# matched across epipolar lines; MAE vs ground truth is the metric.

# %% Imports and configuration
import os
import sys
import warnings
import types
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LEFT_IMG   = SCRIPT_DIR / "im3.png"
RIGHT_IMG  = SCRIPT_DIR / "im4.png"
GT_IMG     = SCRIPT_DIR / "groundtruth.png"
CROCO_DIR  = SCRIPT_DIR / "croco"
CROCO_CKPT = SCRIPT_DIR / "CroCo_V2_ViTBase_SmallDecoder.pth"

# ── Run config ─────────────────────────────────────────────────────────────
UPSCALE_FACTORS = [1, 2, 3, 4]
IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)
device          = torch.device("cpu")

print(f"Device: {device}")

# %% Load images and ground truth
img_left_orig  = Image.open(LEFT_IMG).convert("RGB")
img_right_orig = Image.open(RIGHT_IMG).convert("RGB")
W_orig, H_orig = img_left_orig.size          # PIL gives (W, H)
print(f"Stereo image size: {W_orig}×{H_orig} px")

disp_gt      = np.array(Image.open(GT_IMG)).astype(np.float32) / 16.0
gt_valid     = disp_gt > 0                         # border pixels are 0 (invalid)
gt_vmin      = 0.0                                 # keep 0 so invalid border shows as black
gt_vmax      = disp_gt[gt_valid].max()
print(f"GT disparity range: [0.0, {gt_vmax:.1f}] px")


# %% Shared utilities

def prepare_image(image: Image.Image, scale: int, patch_size: int = 16):
    """Resize to scale × original, snapped to patch_size multiples."""
    new_h = (H_orig * scale // patch_size) * patch_size
    new_w = (W_orig * scale // patch_size) * patch_size
    img_r = TF.resize(image, (new_h, new_w),
                      interpolation=TF.InterpolationMode.BICUBIC)
    img_t = TF.normalize(TF.to_tensor(img_r),
                         mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img_t, new_h // patch_size, new_w // patch_size


def compute_disparity_scanline(feat_l: torch.Tensor, feat_r: torch.Tensor):
    """
    Scanline cosine similarity with non-negative disparity constraint.
    feat_l, feat_r: (D, H, W)  — already on cpu
    Returns disparity (H, W), confidence (H, W)
    """
    D, H, W = feat_l.shape
    fl = F.normalize(feat_l.permute(1, 2, 0), dim=-1)   # (H, W, D)
    fr = F.normalize(feat_r.permute(1, 2, 0), dim=-1)
    sim = torch.bmm(fl, fr.transpose(1, 2))               # (H, W_l, W_r)
    mask = torch.tril(torch.ones(W, W, dtype=torch.bool))
    sim.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    best_j     = sim.argmax(dim=-1)                        # (H, W)
    confidence = sim.max(dim=-1).values                    # (H, W)
    cols       = torch.arange(W).unsqueeze(0).expand(H, -1)
    disparity  = (cols - best_j).float()
    return disparity, confidence


def disp_to_orig(disp_patches: np.ndarray, patch_size: int, actual_scale: float):
    """Convert patch-unit disparity to original-pixel disparity and upsample."""
    disp_orig_px = disp_patches * (patch_size / actual_scale)
    disp_full = F.interpolate(
        torch.from_numpy(disp_orig_px)[None, None].float(),
        size=(H_orig, W_orig), mode="nearest",
    ).squeeze().numpy()
    return disp_orig_px, disp_full


def compute_metrics(disp_full: np.ndarray):
    """Return EPE, NMAD, D1 against global disp_gt (valid pixels only)."""
    pred    = disp_full[gt_valid]
    gt      = disp_gt[gt_valid]
    abs_err = np.abs(pred - gt)
    epe     = float(abs_err.mean())
    err     = pred - gt
    nmad    = float(1.4826 * np.median(np.abs(err - np.median(err))))
    d1      = float((abs_err > 3.0).mean() * 100.0)
    return epe, nmad, d1


def run_scales(extract_fn, patch_size: int = 16):
    """
    Run disparity estimation at all scales.
    extract_fn(img_t, hp, wp) -> feat (D, H, W) tensor on cpu
    Returns dict: scale -> {disp_full, conf_full, disp_patches, grid, mae, time_s}
    """
    results = {}
    for scale in UPSCALE_FACTORS:
        t0 = time.time()
        img_l_t, hp, wp = prepare_image(img_left_orig, scale, patch_size)
        img_r_t, _,  _  = prepare_image(img_right_orig, scale, patch_size)
        actual_scale     = (hp * patch_size) / H_orig

        feat_l = extract_fn(img_l_t, hp, wp)
        feat_r = extract_fn(img_r_t, hp, wp)

        disp_patches, conf = compute_disparity_scanline(feat_l, feat_r)
        disp_orig_px, disp_full = disp_to_orig(
            disp_patches.numpy(), patch_size, actual_scale)

        conf_full = F.interpolate(
            conf[None, None].float(), size=(H_orig, W_orig), mode="nearest"
        ).squeeze().numpy()

        epe, nmad, d1 = compute_metrics(disp_full)
        elapsed       = time.time() - t0

        results[scale] = dict(
            disp_full=disp_full, conf_full=conf_full,
            disp_patches=disp_orig_px, grid=(wp, hp),
            epe=epe, nmad=nmad, d1=d1, time_s=elapsed,
        )
        print(f"  {scale}x  grid {wp}×{hp}  "
              f"disp=[{disp_orig_px.min():.1f},{disp_orig_px.max():.1f}]  "
              f"EPE={epe:.2f}px  NMAD={nmad:.2f}px  D1={d1:.1f}%  ({elapsed:.1f}s)")
    return results


# %% PCA visualization helper

def pca_visualize(feat_l: torch.Tensor, feat_r: torch.Tensor,
                  hp: int, wp: int, encoder_name: str, scale: int):
    """
    Fit PCA(3, whiten) on L2-normalised left features, project both sides,
    apply sigmoid(2x), save to pca_correspondence_{encoder_name}.png
    """
    D = feat_l.shape[0]
    x_l = F.normalize(feat_l.reshape(D, -1).permute(1, 0), dim=-1).numpy()
    x_r = F.normalize(feat_r.reshape(D, -1).permute(1, 0), dim=-1).numpy()

    pca = PCA(n_components=3, whiten=True)
    pca.fit(x_l)
    proj_l = 1 / (1 + np.exp(-2.0 * pca.transform(x_l).reshape(hp, wp, 3)))
    proj_r = 1 / (1 + np.exp(-2.0 * pca.transform(x_r).reshape(hp, wp, 3)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    axes[0].imshow(proj_l); axes[0].set_title("Left — PCA features"); axes[0].axis("off")
    axes[1].imshow(proj_r); axes[1].set_title("Right — PCA features"); axes[1].axis("off")
    fig.suptitle(f"{encoder_name} PCA correspondence ({scale}x, {wp}×{hp})", fontsize=13)
    plt.tight_layout()
    out = SCRIPT_DIR / f"pca_correspondence_{encoder_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  PCA saved → {out.name}")


def run_encoder(name: str, extract_fn, patch_size: int = 16):
    """Full pipeline: all scales + PCA at best scale."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    results = run_scales(extract_fn, patch_size)

    best_scale = min(results, key=lambda s: results[s]["epe"])
    print(f"  Best scale: {best_scale}x  (EPE={results[best_scale]['epe']:.2f}px  NMAD={results[best_scale]['nmad']:.2f}px  D1={results[best_scale]['d1']:.1f}%)")

    # PCA at best scale
    img_l_t, hp, wp = prepare_image(img_left_orig, best_scale, patch_size)
    img_r_t, _,  _  = prepare_image(img_right_orig, best_scale, patch_size)
    feat_l = extract_fn(img_l_t, hp, wp)
    feat_r = extract_fn(img_r_t, hp, wp)
    pca_visualize(feat_l, feat_r, hp, wp, name, best_scale)

    return results


# ── Global store ──────────────────────────────────────────────────────────
all_results: dict[str, dict] = {}  # name -> results dict


# %% Encoder: DINOv3 ViT-B

print("\nLoading DINOv3 ViT-B …")
_dinov3 = timm.create_model(
    "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
).eval()
_dinov3_dim = _dinov3.embed_dim

def _extract_dinov3(img_t, hp, wp):
    with torch.inference_mode():
        out = _dinov3.forward_features(img_t.unsqueeze(0))  # (1, prefix+N, D)
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)                      # last N patch tokens
    return tokens.reshape(hp, wp, _dinov3_dim).permute(2, 0, 1)

all_results["DINOv3 ViT-B"] = run_encoder("DINOv3 ViT-B", _extract_dinov3, patch_size=16)


# %% Encoder: MAE ViT-B

print("\nLoading MAE ViT-B …")
_mae = timm.create_model(
    "vit_base_patch16_224.mae", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_mae_dim = _mae.embed_dim

def _extract_mae(img_t, hp, wp):
    with torch.inference_mode():
        out = _mae.forward_features(img_t.unsqueeze(0))   # (1, 1+N, D)
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _mae_dim).permute(2, 0, 1)

all_results["MAE ViT-B"] = run_encoder("MAE ViT-B", _extract_mae, patch_size=16)


# %% Encoder: DINOv2 ViT-B  (patch_size=14 !)

print("\nLoading DINOv2 ViT-B …")
_dinov2 = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_dinov2_dim = _dinov2.embed_dim

def _extract_dinov2(img_t, hp, wp):
    with torch.inference_mode():
        out = _dinov2.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _dinov2_dim).permute(2, 0, 1)

all_results["DINOv2 ViT-B"] = run_encoder("DINOv2 ViT-B", _extract_dinov2, patch_size=14)


# %% Encoder: DINOv2 ViT-B + Register tokens  (patch_size=14 !)

print("\nLoading DINOv2 ViT-B + Registers …")
_dinov2reg = timm.create_model(
    "vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_dinov2reg_dim = _dinov2reg.embed_dim

def _extract_dinov2reg(img_t, hp, wp):
    with torch.inference_mode():
        out = _dinov2reg.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _dinov2reg_dim).permute(2, 0, 1)

all_results["DINOv2+Reg ViT-B"] = run_encoder("DINOv2+Reg ViT-B", _extract_dinov2reg, patch_size=14)


# %% Encoder: DINOv1 ViT-B

print("\nLoading DINOv1 ViT-B …")
_dinov1 = timm.create_model(
    "vit_base_patch16_224.dino", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_dinov1_dim = _dinov1.embed_dim

def _extract_dinov1(img_t, hp, wp):
    with torch.inference_mode():
        out = _dinov1.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _dinov1_dim).permute(2, 0, 1)

all_results["DINOv1 ViT-B"] = run_encoder("DINOv1 ViT-B", _extract_dinov1, patch_size=16)


# %% Encoder: CLIP ViT-B/16

print("\nLoading CLIP ViT-B/16 …")
_clip = timm.create_model(
    "vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_clip_dim = _clip.embed_dim

def _extract_clip(img_t, hp, wp):
    with torch.inference_mode():
        out = _clip.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _clip_dim).permute(2, 0, 1)

all_results["CLIP ViT-B"] = run_encoder("CLIP ViT-B", _extract_clip, patch_size=16)


# %% Encoder: Supervised ViT-B (IN21k → IN1k)

print("\nLoading Supervised ViT-B …")
_sup = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0,
    dynamic_img_size=True
).eval()
_sup_dim = _sup.embed_dim

def _extract_sup(img_t, hp, wp):
    with torch.inference_mode():
        out = _sup.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _sup_dim).permute(2, 0, 1)

all_results["Supervised ViT-B"] = run_encoder("Supervised ViT-B", _extract_sup, patch_size=16)


# %% Encoder: Random ViT-B (untrained baseline)

print("\nLoading Random ViT-B (untrained baseline) …")
_random = timm.create_model(
    "vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0
).eval()
_random_dim = _random.embed_dim

def _extract_random(img_t, hp, wp):
    with torch.inference_mode():
        out = _random.forward_features(img_t.unsqueeze(0))
    n = hp * wp
    tokens = out[:, -n:, :].squeeze(0)
    return tokens.reshape(hp, wp, _random_dim).permute(2, 0, 1)

all_results["Random ViT-B"] = run_encoder("Random ViT-B", _extract_random, patch_size=16)


# %% Encoder: CroCo v2 ViT-B  (optional — skip on any failure)

def _try_load_croco():
    import subprocess

    # Clone repo if needed
    if not CROCO_DIR.exists():
        print("  Cloning naver/croco …")
        subprocess.run(
            ["git", "clone", "https://github.com/naver/croco.git", str(CROCO_DIR)],
            check=True, capture_output=True,
        )

    if str(CROCO_DIR) not in sys.path:
        sys.path.insert(0, str(CROCO_DIR))

    # Download checkpoint if needed
    if not CROCO_CKPT.exists():
        print("  Downloading CroCo v2 checkpoint (~200 MB) …")
        import urllib.request
        url = ("https://download.europe.naverlabs.com/ComputerVision/"
               "CroCo/CroCo_V2_ViTBase_SmallDecoder.pth")
        urllib.request.urlretrieve(url, CROCO_CKPT)

    from models.croco import CroCoNet   # noqa: PLC0415

    ckpt  = torch.load(CROCO_CKPT, map_location="cpu")
    model = CroCoNet(**ckpt.get("croco_kwargs", {}))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Monkey-patch PatchEmbed to remove size assertion
    def _flex_forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    model.patch_embed.forward = types.MethodType(_flex_forward, model.patch_embed)
    return model


print("\nLoading CroCo v2 ViT-B …")
try:
    _croco = _try_load_croco()
    _croco_dim = _croco.enc_embed_dim

    def _extract_croco(img_t, hp, wp):
        x = img_t.unsqueeze(0)
        with torch.inference_mode():
            feat = _croco.patch_embed(x)
            gy, gx = torch.meshgrid(
                torch.arange(hp), torch.arange(wp), indexing="ij"
            )
            xpos = torch.stack([gy, gx], dim=-1).long().reshape(1, -1, 2)
            for blk in _croco.enc_blocks:
                feat = blk(feat, xpos=xpos)
            feat = _croco.enc_norm(feat)
        tokens = feat.squeeze(0)                 # (N, D)
        return tokens.reshape(hp, wp, _croco_dim).permute(2, 0, 1)

    all_results["CroCo v2 ViT-B"] = run_encoder(
        "CroCo v2 ViT-B", _extract_croco, patch_size=16
    )
except Exception as e:
    print(f"  CroCo skipped: {e}")


# %% Summary table

ENCODER_META = {
    "DINOv3 ViT-B":     ("DINO v3",       "RoPE"),
    "MAE ViT-B":        ("MAE",            "Learned"),
    "DINOv2 ViT-B":     ("DINO v2",        "Learned"),
    "DINOv2+Reg ViT-B": ("DINO v2+Reg",   "Learned"),
    "DINOv1 ViT-B":     ("DINO v1",        "Learned"),
    "CLIP ViT-B":       ("CLIP",           "Learned"),
    "Supervised ViT-B": ("Supervised",     "Learned"),
    "CroCo v2 ViT-B":   ("CroCo v2",      "RoPE"),
    "Random ViT-B":     ("Random",        "RoPE"),
}

scale_hdr = "  ".join(f"{s}x EPE  {s}x NMAD  {s}x D1" for s in UPSCALE_FACTORS)
header = f"{'Encoder':<22} {'Training':<14} {'Pos':<8} {scale_hdr}"
print(f"\n{header}")
print("-" * len(header))
for name, res in all_results.items():
    training, pos = ENCODER_META.get(name, ("?", "?"))
    cols = "  ".join(
        f"{res[s]['epe']:>6.2f}  {res[s]['nmad']:>6.2f}  {res[s]['d1']:>5.1f}" for s in UPSCALE_FACTORS
    )
    print(f"{name:<22} {training:<14} {pos:<8} {cols}")


# %% Comparison plot: EPE and MedErr vs scale

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
for name, res in all_results.items():
    if name == "Random ViT-B":
        continue
    epes  = [res[s]["epe"]  for s in UPSCALE_FACTORS]
    nmads = [res[s]["nmad"] for s in UPSCALE_FACTORS]
    d1s   = [res[s]["d1"]   for s in UPSCALE_FACTORS]
    ax1.plot(UPSCALE_FACTORS, epes,  marker="o", label=name)
    ax2.plot(UPSCALE_FACTORS, nmads, marker="o", label=name)
    ax3.plot(UPSCALE_FACTORS, d1s,   marker="o", label=name)

ax1.set_xlabel("Scale"); ax1.set_ylabel("EPE (pixels)")
ax1.set_title("End-Point Error vs Input Scale")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

ax2.set_xlabel("Scale"); ax2.set_ylabel("NMAD (pixels)")
ax2.set_title("NMAD vs Input Scale")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

ax3.set_xlabel("Scale"); ax3.set_ylabel("D1 (%)")
ax3.set_title("D1 (>3px outlier rate) vs Input Scale")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

out = SCRIPT_DIR / "encoder_comparison.png"
plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nComparison plot saved → {out.name}")


# %% Per-encoder disparity visualizations at best scale

DISP_PLOT_SCALE = 4   # always visualize at this scale

for name, res in all_results.items():
    r = res[DISP_PLOT_SCALE]
    wp, hp = r["grid"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(disp_gt,        cmap="gray", vmin=gt_vmin, vmax=gt_vmax)
    axes[0].set_title("Ground truth"); axes[0].axis("off")
    axes[1].imshow(r["disp_full"], cmap="gray", vmin=gt_vmin, vmax=gt_vmax)
    axes[1].set_title(f"{name} @ {DISP_PLOT_SCALE}x\nEPE={r['epe']:.2f}px  NMAD={r['nmad']:.2f}px  D1={r['d1']:.1f}%")
    axes[1].axis("off")
    err = np.abs(r["disp_full"] - disp_gt)
    err_disp = np.where(gt_valid, err, np.nan)
    _ecmap = plt.cm.hot.copy(); _ecmap.set_bad("black")
    im = axes[2].imshow(err_disp, cmap=_ecmap, vmin=0, vmax=20)
    axes[2].set_title("Absolute error"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="px")
    plt.tight_layout()
    tag  = name.lower().replace(" ", "_")
    path = SCRIPT_DIR / f"disparity_comparison_{tag}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()

print("Disparity comparison images saved.")


# %% Save results as markdown table

md_lines = [
    "# ViT Encoder Stereo Disparity Results\n",
    "Metrics: EPE = mean |pred − GT|; NMAD = 1.4826 × median(|error − median(error)|); D1 = % pixels with |error| > 3px (valid pixels only)\n",
    "| Encoder | Training | Pos Embed |"
    + "".join(f" {s}x EPE | {s}x NMAD | {s}x D1 |" for s in UPSCALE_FACTORS),
    "|---|---|---|" + "---|---|---|" * len(UPSCALE_FACTORS),
]
for name, res in all_results.items():
    training, pos = ENCODER_META.get(name, ("?", "?"))
    row = f"| {name} | {training} | {pos} |"
    row += "".join(
        f" {res[s]['epe']:.2f} | {res[s]['nmad']:.2f} | {res[s]['d1']:.1f} |" for s in UPSCALE_FACTORS
    )
    md_lines.append(row)

md_path = SCRIPT_DIR / "results_table.md"
md_path.write_text("\n".join(md_lines) + "\n")
print(f"Markdown table saved → {md_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# %% Block matching baselines
# ═══════════════════════════════════════════════════════════════════════════

from scipy.ndimage import uniform_filter   # noqa: PLC0415

# Convert stereo images to grayscale float [0, 1]
img_l_gray = np.array(img_left_orig.convert("L")).astype(np.float32) / 255.0
img_r_gray = np.array(img_right_orig.convert("L")).astype(np.float32) / 255.0


# %% Block matching — dense (stride 1, SAD)

def block_match_dense(left_gray: np.ndarray, right_gray: np.ndarray,
                      block_size: int, max_disp: int = 30):
    """
    Dense SAD block matching (stride 1).
    Returns disparity in pixels, shape (H, W).
    """
    H, W  = left_gray.shape
    l     = left_gray.astype(np.float32)
    r     = right_gray.astype(np.float32)
    cost  = np.full((max_disp, H, W), np.inf, dtype=np.float32)

    for d in range(max_disp):
        if d == 0:
            diff = np.abs(l - r)
        else:
            diff          = np.full((H, W), np.inf, dtype=np.float32)
            diff[:, d:]   = np.abs(l[:, d:] - r[:, :W - d])
        # Box-filter to get block SAD per pixel
        sad = uniform_filter(
            np.where(np.isinf(diff), 0.0, diff).astype(np.float64),
            size=block_size, mode="reflect",
        ).astype(np.float32)
        # Mark pixels where any block element was out of bounds as inf
        if d > 0:
            sad[:, :d] = np.inf
        cost[d] = sad

    return cost.argmin(axis=0).astype(np.float32)


DENSE_BLOCK_SIZES = [3, 5, 7, 9]
MAX_DISP          = 30

dense_results = {}

print(f"\n{'='*65}")
print("  Block Matching — Dense (stride 1, SAD)")
print(f"{'='*65}")

for bs in DENSE_BLOCK_SIZES:
    t0 = time.time()
    disp_full = block_match_dense(img_l_gray, img_r_gray, bs, MAX_DISP)
    epe, nmad_val, d1_val = compute_metrics(disp_full)
    elapsed = time.time() - t0
    dense_results[bs] = dict(disp_full=disp_full, epe=epe, nmad=nmad_val, d1=d1_val)
    print(f"  bs={bs}  EPE={epe:.2f}px  NMAD={nmad_val:.2f}px  D1={d1_val:.1f}%  ({elapsed:.2f}s)")


# %% Block matching result tables

print("\n--- Dense block matching ---")
print(f"{'Block size':<12} {'EPE':>8} {'NMAD':>8} {'D1':>8}")
print("-" * 40)
for bs in DENSE_BLOCK_SIZES:
    r = dense_results[bs]
    print(f"  {bs:<10} {r['epe']:>8.2f} {r['nmad']:>8.2f} {r['d1']:>7.1f}%")


# %% Block matching disparity visualizations

_bm_ecmap = plt.cm.hot.copy(); _bm_ecmap.set_bad("black")

for bs in DENSE_BLOCK_SIZES:
    r = dense_results[bs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(disp_gt,        cmap="gray", vmin=gt_vmin, vmax=gt_vmax)
    axes[0].set_title("Ground truth"); axes[0].axis("off")
    axes[1].imshow(r["disp_full"], cmap="gray", vmin=gt_vmin, vmax=gt_vmax)
    axes[1].set_title(f"Block dense bs={bs}\nEPE={r['epe']:.2f}px  NMAD={r['nmad']:.2f}px  D1={r['d1']:.1f}%")
    axes[1].axis("off")
    err = np.abs(r["disp_full"] - disp_gt)
    err_disp = np.where(gt_valid, err, np.nan)
    im  = axes[2].imshow(err_disp, cmap=_bm_ecmap, vmin=0, vmax=20)
    axes[2].set_title("Absolute error"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="px")
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / f"disparity_block_dense_{bs}.png",
                dpi=120, bbox_inches="tight"); plt.close()

print("Block matching disparity images saved.")


# %% Combined comparison plot: dense block matching vs best ViT

best_vit_epe  = min(all_results["DINOv3 ViT-B"][s]["epe"]  for s in UPSCALE_FACTORS)
best_vit_nmad = min(all_results["DINOv3 ViT-B"][s]["nmad"] for s in UPSCALE_FACTORS)
best_vit_d1   = min(all_results["DINOv3 ViT-B"][s]["d1"]   for s in UPSCALE_FACTORS)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

dense_epes  = [dense_results[bs]["epe"]  for bs in DENSE_BLOCK_SIZES]
dense_nmads = [dense_results[bs]["nmad"] for bs in DENSE_BLOCK_SIZES]
dense_d1s   = [dense_results[bs]["d1"]   for bs in DENSE_BLOCK_SIZES]

ax1.plot(DENSE_BLOCK_SIZES, dense_epes, marker="s", linewidth=2,
         label="Pixel blocks (dense SAD)", color="tab:orange")
ax1.axhline(best_vit_epe, linestyle="--", color="tab:blue", linewidth=1.5,
            label=f"Best ViT (DINOv3, EPE={best_vit_epe:.2f}px)")
ax1.set_xlabel("Block size (pixels)"); ax1.set_ylabel("EPE (pixels)")
ax1.set_title("Dense block matching vs best ViT — EPE")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.4)

ax2.plot(DENSE_BLOCK_SIZES, dense_nmads, marker="s", linewidth=2,
         label="Pixel blocks (dense SAD)", color="tab:orange")
ax2.axhline(best_vit_nmad, linestyle="--", color="tab:blue", linewidth=1.5,
            label=f"Best ViT (DINOv3, NMAD={best_vit_nmad:.2f}px)")
ax2.set_xlabel("Block size (pixels)"); ax2.set_ylabel("NMAD (pixels)")
ax2.set_title("Dense block matching vs best ViT — NMAD")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.4)

ax3.plot(DENSE_BLOCK_SIZES, dense_d1s, marker="s", linewidth=2,
         label="Pixel blocks (dense SAD)", color="tab:orange")
ax3.axhline(best_vit_d1, linestyle="--", color="tab:blue", linewidth=1.5,
            label=f"Best ViT (DINOv3, D1={best_vit_d1:.1f}%)")
ax3.set_xlabel("Block size (pixels)"); ax3.set_ylabel("D1 (%)")
ax3.set_title("Dense block matching vs best ViT — D1")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "block_matching_comparison.png",
            dpi=150, bbox_inches="tight"); plt.close()
print("Combined comparison plot saved → block_matching_comparison.png")


# %% Save block matching markdown table

bm_md = [
    "# Block Matching Baseline Results\n",
    "Metrics computed on valid pixels only (GT > 0). EPE = mean |pred − GT|; NMAD = 1.4826 × median(|error − median(error)|); D1 = % pixels with |error| > 3px\n",
    "## Dense (stride 1, SAD, max_disp=30)\n",
    "| Block size | EPE | NMAD | D1 (%) |",
    "|---|---|---|---|",
]
for bs in DENSE_BLOCK_SIZES:
    r = dense_results[bs]
    bm_md.append(f"| {bs} | {r['epe']:.2f} | {r['nmad']:.2f} | {r['d1']:.1f} |")

(SCRIPT_DIR / "results_block_matching.md").write_text("\n".join(bm_md) + "\n")
print("Block matching markdown table saved → results_block_matching.md")

print("\nDone.")
