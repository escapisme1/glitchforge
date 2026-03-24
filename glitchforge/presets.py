from glitchforge.effects.channel import channel_shift_from_params
from glitchforge.effects.scanline import scanline_shift
from glitchforge.effects.noise import inject_noise
from glitchforge.effects.colordrift import color_drift
from glitchforge.effects.blockshift import block_shift
from glitchforge.effects.pixelsort import pixel_sort


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

P1_MELTDOWN = {
    "pipeline": ["channel", "scanline", "colordrift"],
    "channel": {"amount": 12, "direction": "both"},
    "scanline": {"intensity": 0.6, "density": 0.4},
    "colordrift": {"direction": "right", "strength": 0.7, "decay": 0.92},
}

P2_BLOCKROT = {
    "pipeline": ["blockshift", "channel", "noise"],
    "blockshift": {"block_size": 32, "intensity": 0.35, "max_displacement": 80},
    "channel": {"amount": 6, "direction": "horizontal"},
    "noise": {"noise_type": "scanline_drop", "density": 0.05},
}

P3_FLOWSORT = {
    "pipeline": ["channel", "pixelsort", "pixelsort"],
    "channel": {"amount": 4, "direction": "horizontal"},
    "pixelsort_0": {"axis": 0, "key": "luminance", "threshold": 80, "reverse": False},
    "pixelsort_1": {"axis": 1, "key": "saturation", "threshold": 120, "reverse": True},
}

P4_SCANBURN = {
    "pipeline": ["scanline", "colordrift", "channel", "noise"],
    "scanline": {"intensity": 0.85, "density": 0.65},
    "colordrift": {"direction": "right", "strength": 0.5, "decay": 0.88},
    "channel": {"amount": 8, "direction": "horizontal"},
    "noise": {"noise_type": "salt_pepper", "density": 0.02},
}

PRESETS = {
    "p1": P1_MELTDOWN,
    "p2": P2_BLOCKROT,
    "p3": P3_FLOWSORT,
    "p4": P4_SCANBURN,
}

# ---------------------------------------------------------------------------
# Effect registry — maps pipeline names to callable functions
# ---------------------------------------------------------------------------

EFFECT_REGISTRY = {
    "scanline": scanline_shift,
    "channel": channel_shift_from_params,
    "pixelsort": pixel_sort,
    "blockshift": block_shift,
    "colordrift": color_drift,
    "noise": inject_noise,
}
