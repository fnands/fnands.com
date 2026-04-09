# Stereo Matching Encoder Comparison — Instructions for Claude Code

## Goal

Compare how well different pretrained ViT encoders perform at **stereo disparity estimation via scanline cosine similarity matching** on DINOv3 patch features. We want to understand which self-supervised/supervised training objectives produce the most spatially discriminative patch features.

## Method

For each encoder:
1. Load pretrained ViT encoder (ViT-B/16 preferred for fair comparison, fall back to ViT-S/16 or ViT-L/16 if ViT-B unavailable)
2. Feed in a rectified stereo pair (`im3.png` and `im4.png`, 384×288 pixels) at multiple resolutions: **1x, 2x, 3x, 4x**
3. Extract post-LayerNorm patch features reshaped to spatial grid `(D, H_patches, W_patches)`
4. For each row in the feature grid, compute cosine similarity between all left patches and all right patches on the same row (epipolar constraint), with non-negative disparity constraint (right_col ≤ left_col)
5. Convert disparity from patch units to original pixel units: `disp_orig_px = disp_patches * PATCH_SIZE / upscale_factor`
6. Bilinearly upsample to original resolution
7. Compare against ground truth (`groundtruth.png`, values must be divided by 16) using MAE (mean absolute error)

## Encoders to Test

### 1. DINOv3 (patch-level self-supervised, discriminative, RoPE)
- **timm**: `vit_small_patch16_dinov3.lvd1689m` (ViT-S, 384-dim)
- **timm**: `vit_base_patch16_dinov3.lvd1689m` (ViT-B, 768-dim) — preferred
- Uses RoPE — handles arbitrary resolutions natively
- No CLS token; `forward_features` returns patch tokens only (check this)
- `num_classes=0`

### 2. DINOv2 (patch-level self-supervised, discriminative, learned pos embed)
- **timm**: `vit_base_patch14_dinov2.lvd142m` (ViT-B, 768-dim, **patch size 14** not 16!)
- Has register tokens in some variants: `vit_base_patch14_reg4_dinov2.lvd142m`
- Learned pos embed — needs `dynamic_img_size=True` for multiscale
- **IMPORTANT**: patch_size=14, not 16. All disparity conversion math must use the actual patch size. Resize images so dimensions are divisible by 14.
- `num_classes=0`

### 3. DINOv1 (patch-level self-supervised, discriminative, learned pos embed)
- **timm**: `vit_base_patch16_224.dino`
- Learned pos embed — needs `dynamic_img_size=True`
- Has CLS token — strip token index 0 from `forward_features` output
- `num_classes=0`

### 4. MAE (reconstructive in pixel space, self-supervised, learned pos embed)
- **timm**: `vit_base_patch16_224.mae`
- Learned pos embed — needs `dynamic_img_size=True`
- Has CLS token — strip token index 0
- `num_classes=0`

### 5. CroCo v2 (cross-view reconstructive, self-supervised, RoPE)
- **NOT in timm** — load from naver/croco repo
- Download: `wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_SmallDecoder.pth`
- Clone repo: `git clone https://github.com/naver/croco.git`
- Load via:
  ```python
  import sys
  sys.path.insert(0, 'path/to/croco')
  from models.croco import CroCoNet
  ckpt = torch.load('CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
  model = CroCoNet(**ckpt.get('croco_kwargs', {}))
  model.load_state_dict(ckpt['model'], strict=True)
  ```
- Extract encoder features only via `model._encode_image(img_tensor, do_mask=False)` — returns `(features, positions, masks)`. Take `features` which is `(B, N_patches, D)`.
- Then apply `model.enc_norm(features)` — **this is critical**, it applies the final LayerNorm.
- **RoPE**: the encoder blocks need positions passed in, which `_encode_image` handles internally.
- **IMPORTANT**: The `PatchEmbed` in CroCo asserts input size matches `img_size`. You may need to override this. The simplest approach: modify the assertion in `models/blocks.py` `PatchEmbed.forward()` to remove the size check, or monkey-patch it:
  ```python
  import types
  def flexible_forward(self, x):
      B, C, H, W = x.shape
      x = self.proj(x)
      pos = self.position_getter(B, x.size(2), x.size(3), x.device)
      if self.flatten:
          x = x.flatten(2).transpose(1, 2)
      x = self.norm(x)
      return x, pos
  model.patch_embed.forward = types.MethodType(flexible_forward, model.patch_embed)
  ```
- ViT-B encoder, 768-dim, patch_size=16

### 6. CLIP ViT-B/16 (image-level contrastive, vision-language, learned pos embed)
- **timm**: `vit_base_patch16_clip_224.openai`
- Learned pos embed — needs `dynamic_img_size=True`
- Has CLS token — strip token index 0
- `num_classes=0`

### 7. Supervised ViT (classification baseline, learned pos embed)
- **timm**: `vit_base_patch16_224.augreg_in21k_ft_in1k`
- Learned pos embed — needs `dynamic_img_size=True`
- Has CLS token — strip token index 0
- `num_classes=0`

### 8. I-JEPA (reconstructive in latent space, self-supervised, learned pos embed)
- **NOT in timm** — load from Meta's repo
- Repo: `git clone https://github.com/facebookresearch/ijepa.git`
- Download ViT-B/16 checkpoint from the repo's README (hosted on Meta's servers)
- Architecture is a standard ViT — load weights and rename if needed, similar to CroCo approach
- Learned pos embed — needs pos embed interpolation for multiscale
- Has CLS token — check and strip if present
- If this is too much hassle to get working, skip it and note it as future work

### 9. iBOT (DINO + masked image modeling, self-supervised, learned pos embed)
- **Check timm first** — may be available. Otherwise from `https://github.com/bytedance/ibot`
- Download ViT-B/16 checkpoint
- Learned pos embed — needs `dynamic_img_size=True` or manual interpolation
- Has CLS token — strip if present

## Key Implementation Details

### Feature extraction pattern for timm models
```python
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, dynamic_img_size=True)
model = model.eval().to(device)

with torch.inference_mode():
    out = model.forward_features(img_tensor)  # (B, N_tokens, D)

# Strip CLS/register tokens — check per model:
# - DINOv3: no CLS token, might have register tokens at the start
# - DINOv2: has CLS + optional register tokens at the start
# - DINOv1, MAE, CLIP, Supervised ViT: CLS token at index 0
# - Always verify: num_patches = hp * wp; tokens should be out[:, -num_patches:, :]
#   or out[:, 1:, :] etc depending on model

patch_tokens = out[:, -num_patches:, :]  # safest: take last N tokens
feat_map = patch_tokens.squeeze(0).reshape(hp, wp, feat_dim).permute(2, 0, 1)
```

### Image preparation
```python
def prepare_image(image, scale, patch_size=16):
    new_h = (H_orig * scale // patch_size) * patch_size
    new_w = (W_orig * scale // patch_size) * patch_size
    img_resized = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BICUBIC)
    img_t = TF.to_tensor(img_resized)
    img_t = TF.normalize(img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img_t, new_h // patch_size, new_w // patch_size
```

### Disparity computation
```python
def compute_disparity_scanline(feat_l, feat_r):
    D, H, W = feat_l.shape
    fl = F.normalize(feat_l.permute(1, 2, 0), dim=-1)  # (H, W, D)
    fr = F.normalize(feat_r.permute(1, 2, 0), dim=-1)
    sim = torch.bmm(fl, fr.transpose(1, 2))  # (H, W_left, W_right)
    # Non-negative disparity mask
    mask = torch.tril(torch.ones(W, W, dtype=torch.bool))
    sim.masked_fill_(~mask.unsqueeze(0), float("-inf"))
    best_j = sim.argmax(dim=-1)
    confidence = sim.max(dim=-1).values
    cols = torch.arange(W).unsqueeze(0).expand(H, -1)
    disparity = (cols - best_j).float()
    return disparity, confidence
```

### Disparity unit conversion
```python
upscale_factor = actual_image_height / H_orig
disp_orig_px = disp_patches * (PATCH_SIZE / upscale_factor)
```

### Ground truth
```python
disp_gt = np.array(Image.open("groundtruth.png")).astype(np.float32) / 16.0
```

## Output

### 1. Summary table
Print a results table like:
```
| Encoder         | Training     | Pos Embed | 1x MAE | 2x MAE | 3x MAE | 4x MAE |
|-----------------|------------- |-----------|--------|--------|--------|--------|
| DINOv3 ViT-B    | DINO v3      | RoPE      | ...    | ...    | ...    | ...    |
| DINOv2 ViT-B    | DINO v2      | Learned   | ...    | ...    | ...    | ...    |
| ...             | ...          | ...       | ...    | ...    | ...    | ...    |
```

### 2. Comparison plot
Create a matplotlib figure:
- X axis: scale (1x, 2x, 3x, 4x)
- Y axis: MAE (pixels)
- One line per encoder, with legend
- Save as `encoder_comparison.png`

### 3. Visual comparison at best scale
For each encoder, at its best-performing scale:
- Plot ground truth vs predicted disparity side by side (grayscale, same vmin/vmax)
- Save as `disparity_comparison_{encoder_name}.png`

### 4. PCA correspondence visualization
For DINOv3 at highest scale only:
- Fit PCA(n_components=3, whiten=True) on L2-normalized left image features
- Project both images, apply `sigmoid(x * 2.0)` for vibrant colors
- Save as `pca_correspondence.png`

## Priority Order

If time or compute is limited, test in this order:
1. DINOv3 ViT-B (if ViT-B unavailable, ViT-S is fine)
2. MAE ViT-B
3. CroCo v2 ViT-B
4. DINOv2 ViT-B
5. DINOv1 ViT-B
6. CLIP ViT-B
7. Supervised ViT-B
8. I-JEPA ViT-B
9. iBOT ViT-B

## Notes

- All models should use the same normalization (ImageNet mean/std)
- Be careful with **patch size** — DINOv2 uses patch_size=14, all others use 16
- Be careful with **token layout** — always verify which tokens are patches vs CLS/register by checking shapes
- For non-timm models (CroCo, I-JEPA, iBOT), weight renaming or custom extraction code may be needed
- If a model errors at a particular scale (e.g. OOM at 4x), record what you got and note the failure
- Make sure `dynamic_img_size=True` is set for all timm models with learned pos embeddings
