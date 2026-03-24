import numpy as np
import pytest

from glitchforge.effects.channel import channel_shift, channel_shift_from_params
from glitchforge.effects.scanline import scanline_shift
from glitchforge.effects.noise import inject_noise
from glitchforge.effects.colordrift import color_drift
from glitchforge.effects.blockshift import block_shift
from glitchforge.effects.pixelsort import pixel_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_arr(h=100, w=100):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, np.newaxis]
    arr[:, :, 2] = 128
    return arr


# ---------------------------------------------------------------------------
# channel_shift
# ---------------------------------------------------------------------------

class TestChannelShift:
    def test_shape_preserved(self):
        arr = make_arr()
        out = channel_shift(arr, (5, 0), (0, 5), (-3, 2))
        assert out.shape == arr.shape
        assert out.dtype == np.uint8

    def test_zero_shift_unchanged(self):
        arr = make_arr()
        out = channel_shift(arr, (0, 0), (0, 0), (0, 0))
        np.testing.assert_array_equal(out, arr)

    def test_seed_reproducibility(self):
        arr = make_arr()
        out1 = channel_shift_from_params(arr, amount=10, direction="both", seed=42)
        out2 = channel_shift_from_params(arr, amount=10, direction="both", seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        arr = make_arr()
        out1 = channel_shift_from_params(arr, amount=20, direction="both", seed=1)
        out2 = channel_shift_from_params(arr, amount=20, direction="both", seed=99)
        assert not np.array_equal(out1, out2)

    def test_valid_uint8_output(self):
        arr = make_arr()
        out = channel_shift_from_params(arr, amount=30, direction="both", seed=0)
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# scanline_shift
# ---------------------------------------------------------------------------

class TestScanlineShift:
    def test_shape_preserved(self):
        arr = make_arr()
        out = scanline_shift(arr, intensity=0.5, seed=0)
        assert out.shape == arr.shape

    def test_zero_intensity_unchanged(self):
        arr = make_arr()
        out = scanline_shift(arr, intensity=0.0, seed=0)
        np.testing.assert_array_equal(out, arr)

    def test_seed_reproducibility(self):
        arr = make_arr()
        out1 = scanline_shift(arr, intensity=0.7, seed=7)
        out2 = scanline_shift(arr, intensity=0.7, seed=7)
        np.testing.assert_array_equal(out1, out2)

    def test_full_intensity_valid_output(self):
        arr = make_arr()
        out = scanline_shift(arr, intensity=1.0, seed=0, density=1.0)
        assert out.dtype == np.uint8
        assert out.min() >= 0
        assert out.max() <= 255

    def test_density_zero_unchanged(self):
        arr = make_arr()
        out = scanline_shift(arr, intensity=0.9, seed=0, density=0.0)
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# inject_noise
# ---------------------------------------------------------------------------

class TestInjectNoise:
    @pytest.mark.parametrize("noise_type", ["salt_pepper", "color_burst", "scanline_drop"])
    def test_shape_preserved(self, noise_type):
        arr = make_arr()
        out = inject_noise(arr, noise_type=noise_type, density=0.1, seed=0)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8

    def test_zero_density_unchanged(self):
        arr = make_arr()
        for nt in ["salt_pepper", "color_burst", "scanline_drop"]:
            out = inject_noise(arr, noise_type=nt, density=0.0, seed=0)
            np.testing.assert_array_equal(out, arr)

    def test_seed_reproducibility(self):
        arr = make_arr()
        out1 = inject_noise(arr, noise_type="color_burst", density=0.2, seed=13)
        out2 = inject_noise(arr, noise_type="color_burst", density=0.2, seed=13)
        np.testing.assert_array_equal(out1, out2)

    def test_salt_pepper_values(self):
        arr = make_arr()
        out = inject_noise(arr, noise_type="salt_pepper", density=0.5, seed=0)
        # Modified pixels should be either 0 or 255
        modified_mask = ~np.all(out == arr, axis=2)
        if modified_mask.any():
            modified = out[modified_mask]
            assert np.all((modified == 0) | (modified == 255))

    def test_valid_uint8_output(self):
        arr = make_arr()
        out = inject_noise(arr, noise_type="color_burst", density=1.0, seed=0)
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# color_drift
# ---------------------------------------------------------------------------

class TestColorDrift:
    @pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
    def test_shape_preserved(self, direction):
        arr = make_arr()
        out = color_drift(arr, direction=direction, strength=0.5, decay=0.9)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8

    def test_zero_strength_unchanged(self):
        arr = make_arr()
        out = color_drift(arr, direction="right", strength=0.0, decay=0.9)
        np.testing.assert_array_equal(out, arr)

    def test_valid_uint8_output(self):
        arr = make_arr()
        out = color_drift(arr, direction="right", strength=1.0, decay=0.95)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_drift_changes_image(self):
        arr = make_arr()
        out = color_drift(arr, direction="right", strength=0.8, decay=0.85)
        assert not np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# block_shift
# ---------------------------------------------------------------------------

class TestBlockShift:
    def test_shape_preserved(self):
        arr = make_arr()
        out = block_shift(arr, block_size=16, intensity=0.5, seed=0)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8

    def test_zero_intensity_unchanged(self):
        arr = make_arr()
        out = block_shift(arr, block_size=16, intensity=0.0, seed=0)
        np.testing.assert_array_equal(out, arr)

    def test_seed_reproducibility(self):
        arr = make_arr()
        out1 = block_shift(arr, block_size=16, intensity=0.5, seed=99)
        out2 = block_shift(arr, block_size=16, intensity=0.5, seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_full_intensity_valid_output(self):
        arr = make_arr()
        out = block_shift(arr, block_size=8, intensity=1.0, seed=0)
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# pixel_sort
# ---------------------------------------------------------------------------

class TestPixelSort:
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("key", ["luminance", "red", "saturation"])
    def test_shape_preserved(self, axis, key):
        arr = make_arr()
        out = pixel_sort(arr, axis=axis, key=key, threshold=80, reverse=False)
        assert out.shape == arr.shape
        assert out.dtype == np.uint8

    def test_zero_threshold_sorts_everything(self):
        arr = make_arr()
        out = pixel_sort(arr, axis=0, key="luminance", threshold=0, reverse=False)
        assert out.shape == arr.shape

    def test_high_threshold_unchanged(self):
        """With threshold above all pixel values, nothing should be sorted."""
        arr = make_arr()
        out = pixel_sort(arr, axis=0, key="luminance", threshold=300, reverse=False)
        np.testing.assert_array_equal(out, arr)

    def test_valid_uint8_output(self):
        arr = make_arr()
        out = pixel_sort(arr, axis=0, key="luminance", threshold=50, reverse=True)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_invalid_key_raises(self):
        arr = make_arr()
        with pytest.raises(ValueError):
            pixel_sort(arr, axis=0, key="invalid", threshold=80, reverse=False)
