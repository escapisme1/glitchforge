import argparse
import copy
import random
import sys
from typing import Optional

from PIL import Image

from glitchforge.core import run_pipeline, apply_strength
from glitchforge.presets import PRESETS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glitchforge",
        description="CLI image glitch and datamosh tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  glitchforge photo.jpg out.png --p1
  glitchforge photo.jpg out.png --p3 --strength 0.5
  glitchforge photo.jpg out.png --scanline 0.7 --channel 15 --seed 42
  glitchforge photo.jpg out.png --p2 --block-size 16 --seed 1337
""",
    )

    p.add_argument("input", metavar="INPUT", help="Input image path")
    p.add_argument("output", metavar="OUTPUT", help="Output image path")

    # Presets
    pg = p.add_argument_group("presets")
    pg.add_argument("--p1", action="store_true", help="Preset: MELTDOWN (channel + scanline + color bleed)")
    pg.add_argument("--p2", action="store_true", help="Preset: BLOCKROT (datamosh block corruption)")
    pg.add_argument("--p3", action="store_true", help="Preset: FLOWSORT (pixel sort smear)")
    pg.add_argument("--p4", action="store_true", help="Preset: SCANBURN (CRT scan-line dropout)")

    # Global
    gg = p.add_argument_group("global")
    gg.add_argument("--strength", type=float, default=1.0, metavar="FLOAT",
                    help="Global strength multiplier 0.0–1.0 (default: 1.0)")
    gg.add_argument("--seed", type=int, default=None, metavar="INT",
                    help="RNG seed for reproducible output (default: random)")
    gg.add_argument("--format", dest="fmt", type=str, default=None, metavar="STR",
                    help="Force output format: png, jpeg, webp")
    gg.add_argument("--quality", type=int, default=92, metavar="INT",
                    help="JPEG/WebP output quality 1–100 (default: 92)")

    # Scan-line
    sl = p.add_argument_group("scan-line")
    sl.add_argument("--scanline", type=float, default=None, metavar="FLOAT",
                    help="Horizontal row displacement intensity 0.0–1.0")
    sl.add_argument("--scanline-density", type=float, default=0.3, metavar="FLOAT",
                    help="Fraction of rows affected 0.0–1.0 (default: 0.3)")

    # Channel
    ch = p.add_argument_group("channel separation")
    ch.add_argument("--channel", type=int, default=None, metavar="INT",
                    help="RGB channel shift amount in pixels")
    ch.add_argument("--channel-dir", type=str, default="both", metavar="STR",
                    help="Channel shift direction: horizontal, vertical, both (default: both)")

    # Pixel sort
    ps = p.add_argument_group("pixel sort")
    ps.add_argument("--pixelsort", type=float, default=None, metavar="FLOAT",
                    help="Luminance threshold for pixel sorting 0.0–1.0 (0=disabled)")
    ps.add_argument("--sort-axis", type=str, default="rows", metavar="STR",
                    help="Sort axis: rows, cols, both (default: rows)")
    ps.add_argument("--sort-key", type=str, default="luminance", metavar="STR",
                    help="Sort key: luminance, red, saturation (default: luminance)")

    # Block shift
    bs = p.add_argument_group("block shift")
    bs.add_argument("--blockshift", type=float, default=None, metavar="FLOAT",
                    help="Block corruption probability 0.0–1.0")
    bs.add_argument("--block-size", type=int, default=32, metavar="INT",
                    help="Block size in pixels (default: 32)")

    # Color drift
    dr = p.add_argument_group("color drift")
    dr.add_argument("--drift", type=float, default=None, metavar="FLOAT",
                    help="Color bleed strength 0.0–1.0")
    dr.add_argument("--drift-dir", type=str, default="right", metavar="STR",
                    help="Drift direction: left, right, up, down (default: right)")

    # Noise
    ns = p.add_argument_group("noise")
    ns.add_argument("--noise", type=float, default=None, metavar="FLOAT",
                    help="Noise injection density 0.0–1.0")
    ns.add_argument("--noise-type", type=str, default="color_burst", metavar="STR",
                    help="Noise type: salt_pepper, color_burst, scanline_drop (default: color_burst)")

    return p


def _preset_name(args) -> Optional[str]:
    for name in ("p1", "p2", "p3", "p4"):
        if getattr(args, name, False):
            return name
    return None


def _build_config(args, seed: int) -> dict:
    """Merge preset config with manual CLI overrides."""
    preset_key = _preset_name(args)

    if preset_key:
        config = copy.deepcopy(PRESETS[preset_key])
    else:
        config = {"pipeline": []}

    # Apply manual overrides / additions
    _apply_manual_params(config, args, seed)

    return config


def _apply_manual_params(config: dict, args, seed: int) -> None:
    """Mutate config in-place with any explicitly provided CLI flags."""

    # Scan-line
    if args.scanline is not None:
        if "scanline" not in config["pipeline"]:
            config["pipeline"].append("scanline")
        config.setdefault("scanline", {})
        config["scanline"]["intensity"] = args.scanline
        config["scanline"]["density"] = args.scanline_density

    # Channel
    if args.channel is not None:
        if "channel" not in config["pipeline"]:
            config["pipeline"].append("channel")
        config.setdefault("channel", {})
        config["channel"]["amount"] = args.channel
        config["channel"]["direction"] = args.channel_dir

    # Pixel sort
    if args.pixelsort is not None and args.pixelsort > 0:
        threshold = args.pixelsort * 255.0
        sort_axes = ["rows", "cols"] if args.sort_axis == "both" else [args.sort_axis]
        for axis_str in sort_axes:
            axis_int = 0 if axis_str == "rows" else 1
            config["pipeline"].append("pixelsort")
            n = config["pipeline"].count("pixelsort") - 1
            key = "pixelsort" if n == 0 else f"pixelsort_{n}"
            # If first occurrence has no indexed key yet, leave as bare 'pixelsort'
            config[key] = {
                "axis": axis_int,
                "key": args.sort_key,
                "threshold": threshold,
                "reverse": False,
            }

    # Block shift
    if args.blockshift is not None:
        if "blockshift" not in config["pipeline"]:
            config["pipeline"].append("blockshift")
        config.setdefault("blockshift", {})
        config["blockshift"]["intensity"] = args.blockshift
        config["blockshift"]["block_size"] = args.block_size

    # Color drift
    if args.drift is not None:
        if "colordrift" not in config["pipeline"]:
            config["pipeline"].append("colordrift")
        config.setdefault("colordrift", {})
        config["colordrift"].setdefault("decay", 0.9)
        config["colordrift"]["strength"] = args.drift
        config["colordrift"]["direction"] = args.drift_dir

    # Noise
    if args.noise is not None:
        if "noise" not in config["pipeline"]:
            config["pipeline"].append("noise")
        config.setdefault("noise", {})
        config["noise"]["density"] = args.noise
        config["noise"]["noise_type"] = args.noise_type


def _save_image(image: Image.Image, path: str, fmt: Optional[str], quality: int) -> None:
    save_kwargs: dict = {}

    if fmt:
        fmt_upper = fmt.upper()
        if fmt_upper == "JPEG":
            fmt_upper = "JPEG"
        save_kwargs["format"] = fmt_upper
    else:
        fmt_upper = None

    if fmt_upper in ("JPEG", None) and path.lower().endswith((".jpg", ".jpeg")):
        save_kwargs["quality"] = quality
    elif fmt_upper == "WEBP" or path.lower().endswith(".webp"):
        save_kwargs["quality"] = quality

    # Convert to RGB if saving as JPEG (no alpha support)
    target_fmt = fmt_upper or path.rsplit(".", 1)[-1].upper()
    if target_fmt in ("JPEG", "JPG") and image.mode != "RGB":
        image = image.convert("RGB")

    image.save(path, **save_kwargs)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)

    # Validate input
    try:
        image = Image.open(args.input)
    except FileNotFoundError:
        print(f"glitchforge: error: input file not found: {args.input!r}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"glitchforge: error: cannot open input image: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate at least one effect is requested
    preset_key = _preset_name(args)
    manual_effects = any([
        args.scanline is not None,
        args.channel is not None,
        args.pixelsort is not None,
        args.blockshift is not None,
        args.drift is not None,
        args.noise is not None,
    ])
    if not preset_key and not manual_effects:
        print("glitchforge: error: specify a preset (--p1/--p2/--p3/--p4) or at least one effect flag", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    config = _build_config(args, seed)

    # Apply global strength multiplier
    if args.strength != 1.0:
        config = apply_strength(config, args.strength)

    result = run_pipeline(image, config, seed=seed)

    try:
        _save_image(result, args.output, args.fmt, args.quality)
    except Exception as e:
        print(f"glitchforge: error: cannot save output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"glitchforge: saved {args.output} (seed={seed})")


if __name__ == "__main__":
    main()
