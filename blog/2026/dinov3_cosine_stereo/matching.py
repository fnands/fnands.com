"""Scanline disparity matching and evaluation metrics."""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def prepare_image(image, scale: int, patch_size: int, orig_h: int, orig_w: int):
    """
    Resize PIL image to scale × original, snapped to patch_size multiples.
    Returns (img_tensor, hp, wp, new_h, new_w)
    """
    new_h = (orig_h * scale // patch_size) * patch_size
    new_w = (orig_w * scale // patch_size) * patch_size
    img_r = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BICUBIC)
    img_t = TF.normalize(TF.to_tensor(img_r), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img_t, new_h // patch_size, new_w // patch_size, new_h, new_w


def compute_disparity_scanline(feat_l: torch.Tensor, feat_r: torch.Tensor):
    """
    Scanline cosine similarity matching with non-negative disparity constraint.
    feat_l, feat_r: (D, H, W) tensors on CPU
    Returns (disparity, confidence) both (H, W)
    """
    D, H, W = feat_l.shape
    fl  = F.normalize(feat_l.permute(1, 2, 0), dim=-1)   # (H, W, D)
    fr  = F.normalize(feat_r.permute(1, 2, 0), dim=-1)
    sim = torch.bmm(fl, fr.transpose(1, 2))                # (H, W_l, W_r)
    mask = torch.tril(torch.ones(W, W, dtype=torch.bool))
    sim.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    best_j     = sim.argmax(dim=-1)
    confidence = sim.max(dim=-1).values
    cols       = torch.arange(W).unsqueeze(0).expand(H, -1)
    disparity  = (cols - best_j).float()
    return disparity, confidence


def disp_patches_to_pixels(disp_patches: np.ndarray, patch_size: int,
                            actual_scale: float, orig_h: int, orig_w: int):
    """
    Convert patch-unit disparity to pixel units and upsample to original resolution.
    actual_scale = new_h / orig_h
    Returns (disp_orig_px_grid, disp_full_res)
    """
    disp_px   = disp_patches * (patch_size / actual_scale)
    disp_full = upsample_nearest(disp_px, orig_h, orig_w)
    return disp_px, disp_full


def upsample_nearest(disp: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsample a 2-D disparity map to (out_h, out_w) with nearest-neighbor."""
    return F.interpolate(
        torch.from_numpy(disp)[None, None].float(),
        size=(out_h, out_w), mode="nearest",
    ).squeeze().numpy()


def compute_metrics(predicted: np.ndarray, ground_truth: np.ndarray,
                    valid_mask: np.ndarray = None) -> dict:
    """
    Compute EPE and D1 on valid pixels.
    D1 threshold: max(3px, 5% of GT)  — standard Middlebury formula.
    """
    if valid_mask is not None:
        pred = predicted[valid_mask]
        gt   = ground_truth[valid_mask]
    else:
        pred = predicted.ravel()
        gt   = ground_truth.ravel()
    err       = np.abs(pred - gt)
    epe       = float(err.mean())
    threshold = np.maximum(3.0, 0.05 * np.abs(gt))
    d1        = float((err > threshold).mean() * 100.0)
    return {"epe": epe, "d1": d1}
