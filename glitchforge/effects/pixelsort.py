import numpy as np


def _luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _saturation(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.where(cmax == 0, 0.0, delta / cmax)
    return sat * 255.0


def _sort_row(row_pixels: np.ndarray, row_key: np.ndarray, threshold: float, reverse: bool, span) -> np.ndarray:
    """Sort contiguous masked segments within a single row."""
    N = len(row_pixels)
    mask = row_key > threshold
    result = row_pixels.copy()

    # Find contiguous runs where mask is True
    # Pad mask to detect edges at boundaries
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        seg_len = e - s
        if seg_len < 2:
            continue
        if span is not None and seg_len > span:
            e = s + span
            seg_len = span

        seg = row_pixels[s:e]
        seg_key = row_key[s:e]
        order = np.argsort(seg_key)
        if reverse:
            order = order[::-1]
        result[s:e] = seg[order]

    return result


def pixel_sort(arr: np.ndarray, axis: int, key: str, threshold: float, reverse: bool, span: int = None) -> np.ndarray:
    """
    Sort pixels within contiguous runs by a key value.

    arr: H x W x 3 uint8 array
    axis: 0 = sort along rows, 1 = sort along columns
    key: 'luminance', 'red', or 'saturation'
    threshold: 0–255, only pixels above this are included in sorted runs
    reverse: if True, sort descending
    span: max segment length to sort (None = unlimited)
    """
    if key == "luminance":
        key_map = _luminance(arr)
    elif key == "red":
        key_map = arr[:, :, 0].astype(np.float32)
    elif key == "saturation":
        key_map = _saturation(arr)
    else:
        raise ValueError(f"Unknown sort key: {key!r}")

    result = arr.copy()
    H, W = arr.shape[:2]

    if axis == 0:
        # Sort along rows (each row independently)
        for i in range(H):
            result[i] = _sort_row(arr[i], key_map[i], threshold, reverse, span)
    else:
        # Sort along columns — transpose, sort rows, transpose back
        arr_t = arr.transpose(1, 0, 2)          # W x H x 3
        key_t = key_map.T                        # W x H
        result_t = arr_t.copy()
        for j in range(W):
            result_t[j] = _sort_row(arr_t[j], key_t[j], threshold, reverse, span)
        result = result_t.transpose(1, 0, 2)    # back to H x W x 3

    return result
