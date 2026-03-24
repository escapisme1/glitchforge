import numpy as np


def color_drift(arr: np.ndarray, direction: str, strength: float, decay: float) -> np.ndarray:
    """
    Bleed color values along a direction with exponential decay.

    arr: H x W x 3 uint8 array
    direction: 'left', 'right', 'up', 'down'
    strength: 0.0–1.0, blend weight of drifted result
    decay: 0.5–0.99, how quickly the bleed fades
    """
    if strength == 0.0:
        return arr.copy()

    original = arr.astype(np.float32)

    # Try fast scipy path first
    try:
        from scipy.ndimage import uniform_filter1d
        _scipy_drift(original, direction, strength, decay)
    except ImportError:
        pass

    drifted = _manual_drift(original, direction, decay)
    result = original * (1.0 - strength) + drifted * strength
    return np.clip(result, 0, 255).astype(np.uint8)


def _manual_drift(arr: np.ndarray, direction: str, decay: float) -> np.ndarray:
    H, W = arr.shape[:2]
    drifted = arr.copy()

    if direction == "right":
        acc = np.zeros((H, 3), dtype=np.float32)
        for x in range(W):
            acc = acc * decay + arr[:, x, :] * (1.0 - decay)
            drifted[:, x, :] = acc

    elif direction == "left":
        acc = np.zeros((H, 3), dtype=np.float32)
        for x in range(W - 1, -1, -1):
            acc = acc * decay + arr[:, x, :] * (1.0 - decay)
            drifted[:, x, :] = acc

    elif direction == "down":
        acc = np.zeros((W, 3), dtype=np.float32)
        for y in range(H):
            acc = acc * decay + arr[y, :, :] * (1.0 - decay)
            drifted[y, :, :] = acc

    elif direction == "up":
        acc = np.zeros((W, 3), dtype=np.float32)
        for y in range(H - 1, -1, -1):
            acc = acc * decay + arr[y, :, :] * (1.0 - decay)
            drifted[y, :, :] = acc

    return drifted


def _scipy_drift(arr: np.ndarray, direction: str, strength: float, decay: float) -> np.ndarray:
    # Placeholder — scipy path not used since uniform_filter1d doesn't model
    # exponential decay directly. Manual loop is the correct implementation.
    pass
