import copy
from typing import Dict
import numpy as np
from PIL import Image

from glitchforge.presets import EFFECT_REGISTRY


def run_pipeline(image: Image.Image, config: dict, seed: int = 0) -> Image.Image:
    """
    Run a glitch effect pipeline on a PIL Image.

    config must contain a 'pipeline' key with a list of effect names.
    Effect parameters are looked up from config[effect_name].
    When an effect appears multiple times (e.g. 'pixelsort' twice in FLOWSORT),
    config keys 'pixelsort_0', 'pixelsort_1', ... are used for each call.

    A 'seed' key in the config is passed to seeded effects; otherwise the seed
    parameter is used, incremented per effect to avoid identical patterns.
    """
    arr = np.array(image.convert("RGB"), dtype=np.uint8)

    # Count how many times each effect name has been seen (for _0, _1 aliasing)
    call_counts: Dict[str, int] = {}

    for step_idx, effect_name in enumerate(config["pipeline"]):
        effect_fn = EFFECT_REGISTRY[effect_name]

        # Resolve config key: check indexed variant first, then bare name
        call_idx = call_counts.get(effect_name, 0)
        indexed_key = f"{effect_name}_{call_idx}"
        if indexed_key in config:
            effect_config = copy.deepcopy(config[indexed_key])
        elif effect_name in config:
            effect_config = copy.deepcopy(config[effect_name])
        else:
            effect_config = {}

        call_counts[effect_name] = call_idx + 1

        # Inject seed for effects that accept one (derived per-step for uniqueness)
        import inspect
        sig = inspect.signature(effect_fn)
        if "seed" in sig.parameters and "seed" not in effect_config:
            effect_config["seed"] = seed + step_idx

        arr = effect_fn(arr, **effect_config)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def apply_strength(config: dict, strength: float) -> dict:
    """
    Scale all numeric intensity/amount/density/strength values in a config dict
    by a global strength multiplier (0.0–1.0).
    """
    if strength == 1.0:
        return config

    SCALE_KEYS = {"intensity", "amount", "density", "strength"}
    result = copy.deepcopy(config)

    for effect_key, effect_val in result.items():
        if effect_key == "pipeline":
            continue
        if isinstance(effect_val, dict):
            for param, val in effect_val.items():
                if param in SCALE_KEYS and isinstance(val, (int, float)):
                    effect_val[param] = val * strength

    return result
