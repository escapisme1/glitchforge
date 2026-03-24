import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from glitchforge.cli import main


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_fixtures")
    path = d / "input.png"
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(50, 200, 80, dtype=np.uint8)
    arr[:, :, 1] = 100
    arr[:, :, 2] = np.linspace(200, 50, 80, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Preset smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["--p1", "--p2", "--p3", "--p4"])
def test_preset_runs(sample_image, tmp_path, preset):
    out = str(tmp_path / "out.png")
    main([sample_image, out, preset, "--seed", "42"])
    assert os.path.exists(out)
    img = Image.open(out)
    assert img.size == (80, 80)


def test_preset_with_strength(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--p1", "--strength", "0.3", "--seed", "1"])
    assert os.path.exists(out)


def test_preset_with_override(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--p2", "--block-size", "16", "--seed", "1337"])
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Manual parameters
# ---------------------------------------------------------------------------

def test_manual_scanline(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--scanline", "0.5", "--seed", "0"])
    assert os.path.exists(out)


def test_manual_channel(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--channel", "10", "--seed", "0"])
    assert os.path.exists(out)


def test_manual_noise(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--noise", "0.1", "--noise-type", "salt_pepper", "--seed", "0"])
    assert os.path.exists(out)


def test_manual_drift(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--drift", "0.5", "--drift-dir", "right", "--seed", "0"])
    assert os.path.exists(out)


def test_manual_blockshift(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--blockshift", "0.4", "--seed", "0"])
    assert os.path.exists(out)


def test_manual_pixelsort(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    main([sample_image, out, "--pixelsort", "0.3", "--sort-axis", "rows", "--seed", "0"])
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_produces_same_output(sample_image, tmp_path):
    out1 = str(tmp_path / "out1.png")
    out2 = str(tmp_path / "out2.png")
    main([sample_image, out1, "--p1", "--seed", "42"])
    main([sample_image, out2, "--p1", "--seed", "42"])
    arr1 = np.array(Image.open(out1))
    arr2 = np.array(Image.open(out2))
    np.testing.assert_array_equal(arr1, arr2)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_input_exits(tmp_path):
    out = str(tmp_path / "out.png")
    with pytest.raises(SystemExit) as exc_info:
        main(["/nonexistent/file.png", out, "--p1"])
    assert exc_info.value.code != 0


def test_no_effects_exits(sample_image, tmp_path):
    out = str(tmp_path / "out.png")
    with pytest.raises(SystemExit) as exc_info:
        main([sample_image, out])
    assert exc_info.value.code != 0


def test_help_flag():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_jpeg_output(sample_image, tmp_path):
    out = str(tmp_path / "out.jpg")
    main([sample_image, out, "--p4", "--seed", "0", "--quality", "80"])
    assert os.path.exists(out)
    img = Image.open(out)
    assert img.format == "JPEG"
