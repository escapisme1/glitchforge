import numpy as np


def channel_shift(arr: np.ndarray, r_shift: tuple, g_shift: tuple, b_shift: tuple) -> np.ndarray:
    """
    Shift RGB channels independently to produce chromatic aberration.

    arr: H x W x 3 uint8 array
    r_shift, g_shift, b_shift: (dx, dy) tuples — pixel offset per channel
    """
    result = arr.copy()
    for ch_idx, (dx, dy) in enumerate([r_shift, g_shift, b_shift]):
        channel = arr[:, :, ch_idx]
        if dy != 0:
            channel = np.roll(channel, dy, axis=0)
        if dx != 0:
            channel = np.roll(channel, dx, axis=1)
        result[:, :, ch_idx] = channel
    return result


def channel_shift_from_params(arr: np.ndarray, amount: int, direction: str, seed: int = 0) -> np.ndarray:
    """
    High-level wrapper used by presets and CLI.

    amount: max pixel offset
    direction: 'horizontal', 'vertical', or 'both'
    """
    rng = np.random.default_rng(seed)

    def make_shift():
        if direction == "horizontal":
            return (int(rng.integers(-amount, amount + 1)), 0)
        elif direction == "vertical":
            return (0, int(rng.integers(-amount, amount + 1)))
        else:  # both
            return (int(rng.integers(-amount, amount + 1)), int(rng.integers(-amount, amount + 1)))

    r_shift = make_shift()
    g_shift = make_shift()
    b_shift = make_shift()
    return channel_shift(arr, r_shift, g_shift, b_shift)
