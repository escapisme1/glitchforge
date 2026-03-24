import numpy as np


def inject_noise(arr: np.ndarray, noise_type: str, density: float, seed: int) -> np.ndarray:
    """
    Inject random pixel corruption into the image.

    arr: H x W x 3 uint8 array
    noise_type: 'salt_pepper', 'color_burst', or 'scanline_drop'
    density: 0.0–1.0, fraction of pixels/rows affected
    seed: RNG seed
    """
    if density == 0.0:
        return arr.copy()

    rng = np.random.default_rng(seed)
    result = arr.copy()
    H, W = arr.shape[:2]

    if noise_type == "salt_pepper":
        mask = rng.random((H, W)) < density
        values = rng.choice([0, 255], size=(H, W)).astype(np.uint8)
        result[mask] = values[mask, np.newaxis]

    elif noise_type == "color_burst":
        mask = rng.random((H, W)) < density
        n_pixels = int(mask.sum())
        if n_pixels > 0:
            colors = rng.integers(0, 256, size=(n_pixels, 3), dtype=np.uint8)
            result[mask] = colors

    elif noise_type == "scanline_drop":
        row_mask = rng.random(H) < density
        result[row_mask] = 0

    return result
