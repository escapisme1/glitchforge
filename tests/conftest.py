import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixtures")
    return d


@pytest.fixture(scope="session")
def test_image_path(fixture_dir):
    """100x100 gradient PNG used as test input."""
    path = fixture_dir / "test_100x100.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)          # R gradient
    arr[:, :, 1] = np.linspace(0, 255, 100, dtype=np.uint8)[np.newaxis, :].T  # G gradient
    arr[:, :, 2] = 128
    Image.fromarray(arr).save(str(path))
    return str(path)


@pytest.fixture(scope="session")
def test_arr():
    """Reusable 100x100 numpy array."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(0, 255, 100, dtype=np.uint8)[np.newaxis, :].T
    arr[:, :, 2] = 128
    return arr
