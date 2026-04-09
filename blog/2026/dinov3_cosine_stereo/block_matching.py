"""Classical block matching baselines for stereo disparity."""
import numpy as np
from scipy.ndimage import uniform_filter


def block_match_strided(left_gray: np.ndarray, right_gray: np.ndarray,
                        block_size: int) -> np.ndarray:
    """
    Non-overlapping NCC block matching (stride = block_size).
    Returns disparity in pixel units, shape (H // block_size, W // block_size).
    Caller should upsample to original resolution with matching.upsample_nearest.
    """
    H, W  = left_gray.shape
    H_b   = H // block_size
    W_b   = W // block_size

    def extract_blocks(img):
        rows = img[:H_b * block_size, :W_b * block_size]
        return (rows.reshape(H_b, block_size, W_b, block_size)
                    .transpose(0, 2, 1, 3)
                    .reshape(H_b, W_b, -1))

    bl = extract_blocks(left_gray)
    br = extract_blocks(right_gray)
    bl_n = bl / (np.linalg.norm(bl, axis=-1, keepdims=True) + 1e-8)
    br_n = br / (np.linalg.norm(br, axis=-1, keepdims=True) + 1e-8)

    disp_blocks = np.zeros((H_b, W_b), dtype=np.float32)
    for row in range(H_b):
        sim      = bl_n[row] @ br_n[row].T          # (W_b, W_b)
        sim_tril = np.tril(sim)
        sim_tril[sim_tril == 0] = -np.inf
        np.fill_diagonal(sim_tril, np.diag(sim))
        best_j             = sim_tril.argmax(axis=-1)
        disp_blocks[row]   = (np.arange(W_b) - best_j) * block_size

    return disp_blocks


def block_match_dense(left_gray: np.ndarray, right_gray: np.ndarray,
                      block_size: int, max_disp: int) -> np.ndarray:
    """
    Dense SAD block matching (stride = 1).
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
            diff        = np.full((H, W), np.inf, dtype=np.float32)
            diff[:, d:] = np.abs(l[:, d:] - r[:, :W - d])
        sad = uniform_filter(
            np.where(np.isinf(diff), 0.0, diff).astype(np.float64),
            size=block_size, mode="reflect",
        ).astype(np.float32)
        if d > 0:
            sad[:, :d] = np.inf
        cost[d] = sad

    return cost.argmin(axis=0).astype(np.float32)
