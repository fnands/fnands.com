# Block Matching Baselines for Stereo Disparity — Instructions for Claude Code

## Context

We have an existing encoder comparison script (`stereo_comparison.py`, attached) that tests ViT encoders at scanline cosine similarity matching for stereo disparity. We now want to add classical **block matching** baselines using raw pixel patches (no learned features).

The existing script structure should be reused — same `disp_gt`, same images (`im3.png`, `im4.png`, `groundtruth.png`), same metrics (EPE/MAE and NMAD), same plotting conventions.

## What to implement

Two block matching variants, each as a separate set of experiments:

### Variant A: "Strided block matching" (ViT-equivalent)

This matches the ViT setup exactly: divide the image into non-overlapping blocks, compute one disparity per block, upsample to full resolution.

- **Block sizes: 16, 8, 5, 4** — these correspond to the effective patch size of a ViT at 1x, 2x, ~3x, 4x scale
- **Stride = block_size** (non-overlapping)
- For each block at position `(row, col)` in the left image, search all blocks on the **same row** in the right image (scanline/epipolar constraint)
- Similarity metric: **normalized cross-correlation (NCC)** or equivalently cosine similarity on flattened pixel vectors — this is the pixel-space analogue of what the ViT does with features
- Non-negative disparity constraint: only search `right_col ≤ left_col`
- Disparity is in block units → convert to pixels: `disp_px = disp_blocks * block_size`
- Upsample to original resolution with nearest-neighbor interpolation (same as the ViT pipeline)

### Variant B: "Dense block matching" (standard stereo)

This is classical dense block matching at stride 1 — the standard baseline for stereo algorithms.

- **Block sizes: 3, 5, 7, 9** (half-window sizes of 1, 2, 3, 4)
- **Stride = 1** (every pixel gets a disparity)
- For each pixel `(y, x)` in the left image, extract a block centered on it, and slide it along the **same row** in the right image
- Similarity: **Sum of Absolute Differences (SAD)** or **NCC** — SAD is simpler and standard for block matching
- Non-negative disparity constraint: only search `right_x ≤ left_x`
- Max disparity search range: set to something reasonable, e.g. `max_disp = W_orig // 3` or just cap at 150 pixels
- Output is already at pixel resolution — no upsampling needed
- This produces **per-pixel** disparity, so it should be compared fairly (it has a structural advantage over the strided version and the ViT version)

## Implementation details

### Block matching core (strided version)
```python
def block_match_strided(img_left, img_right, block_size):
    """
    img_left, img_right: (H, W, 3) or (H, W) numpy arrays (use grayscale or RGB)
    Returns: disparity in pixel units, shape (H_blocks, W_blocks)
    """
    # Convert to grayscale float
    # Divide into non-overlapping blocks
    # For each block row, extract all blocks → (W_blocks, block_size*block_size) 
    # Compute cosine similarity matrix between left and right blocks on same row
    # Apply non-negative disparity mask (tril)
    # argmax → disparity in block units → multiply by block_size
    # Return (H_blocks, W_blocks) disparity
```

### Block matching core (dense version)
```python
def block_match_dense(img_left, img_right, block_size, max_disp=150):
    """
    img_left, img_right: (H, W) grayscale numpy arrays
    Returns: disparity in pixels, shape (H, W)
    """
    half = block_size // 2
    disparity = np.zeros((H, W))
    # For each pixel (y, x) in left image (with border padding):
    #   left_block = img_left[y-half:y+half+1, x-half:x+half+1]
    #   For d in range(0, min(max_disp, x+1)):
    #     right_block = img_right[y-half:y+half+1, (x-d)-half:(x-d)+half+1]
    #     cost = SAD or SSD
    #   disparity[y, x] = argmin(cost)
    # This naive loop is slow — vectorize with numpy or use cv2.StereoBM if available
```

### Performance note
The dense version with large images can be slow in pure Python. Options:
1. **Use OpenCV** if available: `cv2.StereoBM_create(numDisparities=..., blockSize=...)` — but this doesn't give exact control over block sizes and uses a slightly different algorithm
2. **Vectorize with numpy**: for each disparity `d`, shift the right image by `d` pixels and compute the block-wise cost in one vectorized operation
3. **Work on the original 384×288 images only** — no need to upscale since dense matching already operates at pixel level

Recommended approach for vectorized dense matching:
```python
def block_match_dense_vectorized(left_gray, right_gray, block_size, max_disp):
    """Vectorized SAD block matching."""
    H, W = left_gray.shape
    half = block_size // 2
    
    # Pad images
    left_pad = np.pad(left_gray, half, mode='reflect')
    right_pad = np.pad(right_gray, half, mode='reflect')
    
    # Compute cost volume: (max_disp, H, W)
    cost_volume = np.full((max_disp, H, W), np.inf)
    
    for d in range(max_disp):
        # Shift right image by d pixels
        # For each pixel, compute SAD over the block using uniform_filter or cumsum trick
        diff = np.abs(left_pad[:, half+d:half+d+W] - right_pad[:, half:half+W])
        # Box filter the diff to get SAD per pixel
        # Use scipy.ndimage.uniform_filter or manual cumsum
        from scipy.ndimage import uniform_filter
        sad = uniform_filter(diff.astype(np.float64), size=block_size, mode='reflect')
        # Crop to original size
        cost_volume[d] = sad[half:half+H, half:half+W] if sad.shape == (H+2*half, W+2*half) else sad[:H, :W]
    
    # Handle border: for pixel at column x, disparities > x are invalid
    for d in range(max_disp):
        cost_volume[d, :, :d] = np.inf
    
    disparity = cost_volume.argmin(axis=0).astype(np.float32)
    return disparity
```

Actually, a cleaner vectorized approach:
```python
def block_match_dense_fast(left_gray, right_gray, block_size, max_disp):
    H, W = left_gray.shape
    half = block_size // 2
    cost_volume = np.full((max_disp, H, W), np.inf, dtype=np.float32)
    
    for d in range(max_disp):
        # Absolute difference at disparity d
        if d == 0:
            diff = np.abs(left_gray.astype(np.float32) - right_gray.astype(np.float32))
        else:
            diff = np.zeros((H, W), dtype=np.float32)
            diff[:, d:] = np.abs(left_gray[:, d:].astype(np.float32) - right_gray[:, :W-d].astype(np.float32))
            diff[:, :d] = np.inf  # invalid: can't match before image start
        
        # Sum over block using uniform_filter (box filter)
        from scipy.ndimage import uniform_filter
        cost_volume[d] = uniform_filter(diff, size=block_size, mode='constant', cval=0)
    
    disparity = cost_volume.argmin(axis=0).astype(np.float32)
    return disparity
```

## Metrics

Compute the same metrics as the encoder comparison:

- **EPE / MAE**: `mean(|predicted - ground_truth|)` 
- **NMAD**: `1.4826 * median(|error - median(error)|)` where `error = predicted - ground_truth`

Ground truth: `np.array(Image.open("groundtruth.png")).astype(np.float32) / 16.0`

## Output

### 1. Results table for strided block matching
```
| Block Size | Equiv ViT Scale | EPE    | NMAD   |
|------------|-----------------|--------|--------|
| 16         | 1x              | ...    | ...    |
| 8          | 2x              | ...    | ...    |
| 5          | ~3x             | ...    | ...    |
| 4          | 4x              | ...    | ...    |
```

### 2. Results table for dense block matching
```
| Block Size | EPE    | NMAD   |
|------------|--------|--------|
| 3          | ...    | ...    |
| 5          | ...    | ...    |
| 7          | ...    | ...    |
| 9          | ...    | ...    |
```

### 3. Comparison plot
One figure with two subplots:

**Left subplot**: "Strided matching: ViT features vs pixel blocks"
- X axis: effective resolution (label as "16px/patch", "8px/patch", "5px/patch", "4px/patch")
- Y axis: EPE (pixels)
- Two lines: "Best ViT encoder (DINOv3)" and "Pixel block matching (strided)"
- Include other ViT encoders as lighter/dashed lines if it doesn't clutter too much

**Right subplot**: "Dense block matching (stride 1)"
- X axis: block size (3, 5, 7, 9)
- Y axis: EPE (pixels)  
- Single line for dense block matching
- Add a horizontal dashed line for "Best ViT result" (DINOv3 at best scale) for reference

Save as `block_matching_comparison.png`

### 4. Disparity visualization
For each method/block size, save a side-by-side with ground truth (same as the encoder script does):
- `disparity_block_strided_{blocksize}.png`
- `disparity_block_dense_{blocksize}.png`

## Integration with existing script

Either:
- Add the block matching code as additional sections at the end of the existing `stereo_comparison.py`
- Or create a separate `block_matching_baseline.py` that imports the shared utilities

The block matching results should also appear in the `all_results` dict (or equivalent) and in the final combined comparison plot alongside the ViT encoders. The combined plot should make it visually clear which methods use learned features vs raw pixels.

## Also update the encoder comparison to use NMAD

Add NMAD as a metric alongside EPE in the existing encoder comparison. The NMAD formula:
```python
def nmad(predicted, ground_truth):
    error = predicted - ground_truth
    return 1.4826 * np.median(np.abs(error - np.median(error)))
```

Update the summary table and plots to include NMAD columns.

## Notes

- Use **grayscale** images for block matching (convert RGB to gray)
- The strided version should be fast since it's just matrix operations on small block counts
- The dense version may be slow for large `max_disp` — set `max_disp` based on the GT range (max is ~14px, so `max_disp=30` or so is plenty)
- Make sure to handle image borders correctly for both variants
- All disparity values should be in **original pixel units** for comparison with GT
