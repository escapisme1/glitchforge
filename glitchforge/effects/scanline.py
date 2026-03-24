import numpy as np


def scanline_shift(arr: np.ndarray, intensity: float, seed: int, density: float = 0.3) -> np.ndarray:
    """
    Shift horizontal rows of pixels by varying amounts.

    arr: H x W x C uint8 array
    intensity: 0.0–1.0, max shift as fraction of image width
    seed: RNG seed for reproducibility
    density: fraction of rows affected (0.0–1.0)
    """
    if intensity == 0.0:
        return arr.copy()

    rng = np.random.default_rng(seed)
    H, W = arr.shape[:2]
    max_shift = int(intensity * W)
    if max_shift == 0:
        return arr.copy()

    result = arr.copy()

    # Boolean mask: which rows are affected
    affected = rng.random(H) < density

    # Power-law distribution: most rows shift little, a few shift a lot
    raw = rng.power(0.5, size=H)  # power < 1 → heavy toward 1, invert for heavy-small
    # Actually use beta distribution skewed toward 0 for most-small effect
    magnitudes = rng.beta(0.4, 1.0, size=H)
    signs = rng.choice([-1, 1], size=H)
    shifts = (signs * magnitudes * max_shift).astype(int)

    for i in range(H):
        if affected[i] and shifts[i] != 0:
            result[i] = np.roll(arr[i], shifts[i], axis=0)

    return result
