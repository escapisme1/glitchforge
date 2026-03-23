# glitchforge

> CLI image glitch and datamosh tool. Melted pixels, scan-line corruption, color bleed, block displacement.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/deps-Pillow%20%2B%20NumPy-orange)

---

## What it does

glitchforge applies image-space corruption effects to photos: channel separation, scan-line displacement, pixel sorting, block-level datamosh simulation, color drift, and digital noise. Fast, lightweight, runs on CPU.

## Examples

| Original | `--p1` MELTDOWN | `--p2` BLOCKROT | `--p3` FLOWSORT | `--p4` SCANBURN |
|---|---|---|---|---|
| *(your photo)* | Vertical color bleed | Block datamosh | Pixel sort smear | CRT scan-line dropout |

*(add example output images here)*

---

## Install

```bash
pip install glitchforge
```

Or from source:

```bash
git clone https://github.com/yourusername/glitchforge
cd glitchforge
pip install -e .
```

**Requirements:** Python 3.9+, Pillow, NumPy. That's it.

---

## Quick Start

```bash
# Use a preset
glitchforge input.jpg output.png --p1

# Reduce strength
glitchforge input.jpg output.png --p3 --strength 0.4

# Manual parameters
glitchforge input.jpg output.png --scanline 0.7 --channel 15

# Reproducible output (same seed = same result)
glitchforge input.jpg output.png --p2 --seed 42
```

---

## Presets

| Flag | Name | Effect |
|---|---|---|
| `--p1` | MELTDOWN | Channel separation + scan-line displacement + color bleed. Neon colors dissolving sideways. |
| `--p2` | BLOCKROT | Block-level displacement + color bars + scanline dropout. Mimics MPEG datamosh corruption. |
| `--p3` | FLOWSORT | Pixel sorting by luminance and saturation. Colors flow in diagonal waves. |
| `--p4` | SCANBURN | Heavy scan-line smear + color drift + chromatic aberration. CRT signal collapse. |

---

## All Parameters

```
glitchforge [OPTIONS] INPUT OUTPUT
```

### Presets
```
--p1                 Preset: MELTDOWN
--p2                 Preset: BLOCKROT
--p3                 Preset: FLOWSORT
--p4                 Preset: SCANBURN
```

### Global
```
--strength FLOAT     Global multiplier for all effect intensities. 0.0–1.0. Default: 1.0
--seed INT           RNG seed for reproducible output. Default: random
--format STR         Force output format: png, jpeg, webp. Default: auto (from extension)
--quality INT        JPEG/WebP quality. 1–100. Default: 92
```

### Scan-line
```
--scanline FLOAT     Horizontal row displacement intensity. 0.0–1.0
--scanline-density FLOAT   Fraction of rows affected. 0.0–1.0. Default: 0.3
```

### Channel Separation
```
--channel INT        RGB channel shift amount in pixels. 0–50
--channel-dir STR    Direction: horizontal, vertical, both. Default: both
```

### Pixel Sort
```
--pixelsort FLOAT    Luminance threshold for pixel sorting. 0.0–1.0 (0=disabled)
--sort-axis STR      Sort axis: rows, cols, both. Default: rows
--sort-key STR       Sort key: luminance, red, saturation. Default: luminance
```

### Block Shift
```
--blockshift FLOAT   Block corruption probability. 0.0–1.0
--block-size INT     Block size in pixels. Default: 32
```

### Color Drift
```
--drift FLOAT        Color bleed strength. 0.0–1.0
--drift-dir STR      Direction: left, right, up, down. Default: right
```

### Noise
```
--noise FLOAT        Noise injection density. 0.0–1.0
--noise-type STR     Type: salt_pepper, color_burst, scanline_drop. Default: color_burst
```

---

## Combining Presets with Overrides

Presets set a baseline. You can override any individual parameter on top of a preset:

```bash
# BLOCKROT preset but with smaller blocks and custom seed
glitchforge photo.jpg out.png --p2 --block-size 16 --seed 1337

# FLOWSORT with reduced strength and column sorting instead of rows
glitchforge photo.jpg out.png --p3 --strength 0.6 --sort-axis cols
```

---

## Performance

- Target: under 2 seconds for a 12 megapixel image on a mid-range CPU
- All effects use NumPy vectorization — no GPU required
- Pixel sort is the slowest effect on large images; reduce `--sort-axis` to `rows` if needed

---

## How the Effects Work

**Scan-line displacement** shifts horizontal rows of pixels by random amounts — some rows barely move, a few shift dramatically. Produces the horizontal smearing and melted look.

**Channel separation** shifts the R, G, and B color channels independently in different directions. Creates chromatic aberration and the rainbow edge fringing common in CRT and tape artifacts.

**Pixel sorting** reorders pixels within rows or columns by luminance or saturation. Pixels above a threshold get sorted, producing flowing color bands. The threshold controls how much of the image is affected.

**Block shift** divides the image into rectangular blocks and displaces them to random source locations — simulating how MPEG video corruption displaces motion-estimation blocks when the stream is damaged.

**Color drift** makes color values bleed along a direction with exponential decay — simulating a CRT or tape signal being pulled sideways.

**Noise** injects random pixel corruption: dead pixels, saturated color bursts, or entire dropped scan-lines set to black.

---

## Note on "True" Datamoshing

Real datamoshing requires encoding video as H.264 and corrupting the bitstream (removing I-frames, mangling motion vectors). This tool simulates the *visual appearance* of those artifacts without requiring ffmpeg or video encoding. If you need genuine bitstream-level corruption, look into ffmpeg-based datamosh pipelines.

---

## License

MIT — do whatever you want, attribution appreciated.

---

## Contributing

Pull requests welcome. Keep dependencies minimal — Pillow and NumPy only (no hard scipy/numba deps). New effects should live in `glitchforge/effects/` and follow the `(arr: np.ndarray, ...) -> np.ndarray` interface. Add tests.
