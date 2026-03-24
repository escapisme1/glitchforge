import numpy as np


def block_shift(arr: np.ndarray, block_size: int, intensity: float, seed: int, max_displacement: int = 80) -> np.ndarray:
    """
    Displace rectangular blocks to simulate MPEG datamosh corruption.

    arr: H x W x 3 uint8 array
    block_size: size of each block in pixels
    intensity: 0.0–1.0, probability that a given block gets corrupted
    seed: RNG seed
    max_displacement: max pixels a block can be displaced from its source
    """
    if intensity == 0.0:
        return arr.copy()

    rng = np.random.default_rng(seed)
    result = arr.copy()
    H, W = arr.shape[:2]

    # Iterate over block grid
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            if rng.random() > intensity:
                continue

            # Actual block bounds (may be smaller at edges)
            by = min(block_size, H - y)
            bx = min(block_size, W - x)

            # Random source offset
            dy = int(rng.integers(-max_displacement, max_displacement + 1))
            dx = int(rng.integers(-max_displacement, max_displacement + 1))

            src_y = np.clip(y + dy, 0, H - by)
            src_x = np.clip(x + dx, 0, W - bx)

            result[y:y + by, x:x + bx] = arr[src_y:src_y + by, src_x:src_x + bx]

    return result
