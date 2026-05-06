"""
iGluSnFR Analysis Toolbox – Hierarchical Config Manager
========================================================

Config filename : iglusnfr_config.json
Resolution order (later wins):
    1. Built-in DEFAULT_CONFIG
    2. iglusnfr_config.json in the input root
    3. iglusnfr_config.json in each successive subdirectory toward the image
    4. (Optional) GUI overrides when the "Override config with GUI" toggle is ON

Per-subfolder granularity is used for processing: images in the same
subdirectory share one resolved config.  A flat input folder (no
subdirectories) is supported – the root config applies to all images.
"""

import copy
import json
import os
import re
from pathlib import Path

CONFIG_FILENAME = "iglusnfr_config.json"

# =========================================================================
# Default config – single source of truth for every parameter
# =========================================================================
DEFAULT_CONFIG = {
    "_comment": (
        "iGluSnFR Analysis Config. "
        "Place this file in the input folder or any subfolder. "
        "Closer configs override parent settings.  "
        "Only include the keys you want to change – missing keys use defaults."
    ),
    "processing": {
        "bkg_percentile": 10,
        "max_distance": 6,
        "min_area": 20,
        "max_area": 400,
        "snr_threshold": 3.0,
        "iou_threshold": 0.4,
        "snr_start_frame": None,
        "snr_end_frame": None,
        "n_jobs": 15,
        "parallel_backend": "threading",
        "ilastik_models": {
            "default": None,
            "spontaneous": "/home/ashirini/projects/vevea/Vevea/mouse10ms_10percentileBG_12_images_BKG1024v2_sponOnly.ilp",
            "evoked": "/home/ashirini/projects/vevea/Vevea/mouse10ms_10percentileBG_22_images_BKG1024v2.ilp",
        },
    },
    "viewing": {
        "frame_rate": 100.0,
        "sd_multiplier": 4.0,
        "rolling_window": 10,
        "baseline_start_frame": 0,
        "baseline_end_frame": 49,
    },
    "ui": {
        "font_size": 11,
        "font_size_large": 13,
        "font_size_title": 14,
        "min_panel_width": 350,
    },
    "analysis": {
        "frame_rate": 100.0,
        "groups": ["WT", "APOE"],
        "sd_multiplier": 4.0,
        "rolling_window": 10,
        "baseline_start_frame": 0,
        "baseline_end_frame": 49,
        "order": 1,
        "stim_start_frame": 50,
        "response_window": 5,
        "ppr_pulse1_frame": 50,
        "ppr_pulse2_frame": 60,
        "ppr_response_window": 4,
        "outlier_config": {
            "baseline_median": {"enabled": True, "threshold": 3.0},
            "baseline_sd":     {"enabled": True, "threshold": 3.0},
            "max_signal":      {"enabled": False, "threshold": 3.0},
            "total_firings":   {"enabled": True, "threshold": 3.0},
            "area":            {"enabled": True, "threshold": 3.0},
            "solidity":        {"enabled": True, "threshold": 3.0},
            "circularity":     {"enabled": True, "threshold": 3.0},
        },
    },
}

# =========================================================================
# Internal helpers
# =========================================================================

def _deep_merge(base, override):
    """Recursively merge *override* into a copy of *base*.

    Only dict values are merged recursively; everything else is replaced.
    Keys present in *base* but absent in *override* are kept as-is.
    """
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key.startswith("_"):
            continue  # skip meta-keys like _comment
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _load_json(path):
    """Load a JSON file, returning an empty dict on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _config_chain(image_dir, input_root):
    """Return a list of config file paths from *input_root* down to
    *image_dir* (inclusive), in order from root → leaf.

    Only directories that actually contain CONFIG_FILENAME are included.
    """
    image_dir = os.path.realpath(image_dir)
    input_root = os.path.realpath(input_root)

    dirs = []
    d = image_dir
    while True:
        dirs.append(d)
        if os.path.normpath(d) == os.path.normpath(input_root):
            break
        parent = os.path.dirname(d)
        if parent == d:
            # reached filesystem root without hitting input_root
            break
        d = parent

    dirs.reverse()  # root first

    configs = []
    for d in dirs:
        cfg_path = os.path.join(d, CONFIG_FILENAME)
        if os.path.isfile(cfg_path):
            configs.append(cfg_path)
    return configs


# =========================================================================
# Public API
# =========================================================================

def resolve_config(image_dir, input_root, gui_overrides=None,
                   use_gui_overrides=False, section=None):
    """Resolve the effective config for images in *image_dir*.

    Parameters
    ----------
    image_dir : str
        Directory containing the image(s).  For a flat input folder this
        equals *input_root*.
    input_root : str
        Top-level input folder selected by the user.
    gui_overrides : dict or None
        Parameter dict collected from the GUI (flat, not sectioned).
    use_gui_overrides : bool
        If True, *gui_overrides* are merged on top of everything else.
    section : str or None
        If given (e.g. "processing"), return only that section of the
        config as a flat dict.  Otherwise return the full nested dict.

    Returns
    -------
    dict
        Resolved configuration.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    for cfg_path in _config_chain(image_dir, input_root):
        user_cfg = _load_json(cfg_path)
        if user_cfg:
            config = _deep_merge(config, user_cfg)

    # Apply GUI overrides into the requested section
    if use_gui_overrides and gui_overrides and section:
        sec = config.get(section, {})
        for k, v in gui_overrides.items():
            if k in sec:
                sec[k] = v
        config[section] = sec

    if section:
        return dict(config.get(section, {}))
    return config


def resolve_config_for_processing(image_path, input_root,
                                  gui_overrides=None,
                                  use_gui_overrides=False):
    """Convenience wrapper: resolve the *processing* section for one image.

    Returns a flat dict suitable for passing directly to ``ProcessImages``.
    """
    image_dir = os.path.dirname(os.path.realpath(image_path))
    return resolve_config(
        image_dir, input_root,
        gui_overrides=gui_overrides,
        use_gui_overrides=use_gui_overrides,
        section="processing",
    )


def resolve_config_for_analysis(output_folder, input_root=None,
                                gui_overrides=None,
                                use_gui_overrides=False):
    """Resolve the *analysis* section.

    For analysis the relevant directory is the output folder
    (which mirrors the input structure).  If *input_root* is not
    available we just use *output_folder* as the root.
    """
    if input_root is None:
        input_root = output_folder
    return resolve_config(
        output_folder, input_root,
        gui_overrides=gui_overrides,
        use_gui_overrides=use_gui_overrides,
        section="analysis",
    )


# =========================================================================
# ilastik model resolution
# =========================================================================

# Keyword patterns used to classify a path as spontaneous or evoked.
# Matched case-insensitively against every component of the relative path
# from input_root to the image.
_MODEL_TYPE_PATTERNS = [
    # (compiled regex, model-type key)
    (re.compile(r"\b(?:spon|spont|spontaneous)\b", re.IGNORECASE), "spontaneous"),
    (re.compile(r"\bevoked\b", re.IGNORECASE), "evoked"),
]


def _classify_path(image_path, input_root):
    """Return the model-type key ('spontaneous', 'evoked', or 'default')
    by scanning the relative path from *input_root* to *image_path* for
    known keywords.
    """
    try:
        rel = os.path.relpath(os.path.realpath(image_path),
                              os.path.realpath(input_root))
    except ValueError:
        rel = os.path.basename(image_path)

    for pattern, model_key in _MODEL_TYPE_PATTERNS:
        if pattern.search(rel):
            return model_key
    return "default"


def resolve_model_path(image_path, input_root, gui_models=None,
                       gui_overrides=None, use_gui_overrides=False):
    """Pick the correct ilastik model for *image_path*.

    Resolution order
    ----------------
    1. Per-subfolder config ``processing.ilastik_model`` (singular string)
       — highest priority explicit override.
    2. Auto-detect experiment type from the image's relative path
       (spontaneous / evoked) and look up the matching entry in the
       resolved ``processing.ilastik_models`` dict (which itself may come
       from root or subfolder configs).
    3. GUI-supplied ``gui_models`` dict fills in any ``None`` slots before
       the lookup in step 2.
    4. Fall back to the ``"default"`` entry.

    Parameters
    ----------
    image_path : str
        Full path to the image file being processed.
    input_root : str
        Top-level input folder selected by the user.
    gui_models : dict or None
        ``{"spontaneous": "/path/...", "evoked": "/path/...",
        "default": "/path/..."}`` collected from the GUI model slots.
        ``None`` values are ignored.
    gui_overrides : dict or None
        Flat processing-parameter overrides from the GUI.
    use_gui_overrides : bool
        Whether the GUI override toggle is on.

    Returns
    -------
    str
        Absolute path to the ``.ilp`` model file.

    Raises
    ------
    FileNotFoundError
        If no valid model path could be resolved.
    """
    image_dir = os.path.dirname(os.path.realpath(image_path))
    cfg = resolve_config(image_dir, input_root,
                         gui_overrides=gui_overrides,
                         use_gui_overrides=use_gui_overrides,
                         section="processing")

    # --- Priority 1: explicit per-subfolder override (singular key) ---
    explicit = cfg.get("ilastik_model")
    if explicit and os.path.isfile(explicit):
        return explicit

    # --- Build effective models dict (config + GUI fill-in) ---
    models = dict(cfg.get("ilastik_models") or {})
    if gui_models:
        for key, path in gui_models.items():
            # GUI fills in only where the config left None / empty
            if path and (not models.get(key)):
                models[key] = path

    # --- Priority 2: auto-detect experiment type from path ---
    model_key = _classify_path(image_path, input_root)
    candidate = models.get(model_key)
    if candidate and os.path.isfile(candidate):
        return candidate

    # --- Priority 3: fall back to default ---
    fallback = models.get("default")
    if fallback and os.path.isfile(fallback):
        return fallback

    # Build a helpful error message
    tried = {model_key: candidate, "default": fallback}
    raise FileNotFoundError(
        f"No valid ilastik model found for '{os.path.basename(image_path)}' "
        f"(detected type='{model_key}'). Tried: {tried}. "
        f"Set model paths in the GUI or in iglusnfr_config.json."
    )


def save_resolved_config(config, output_path):
    """Save the *resolved* config to disk for reproducibility.

    Parameters
    ----------
    config : dict
        Full or section config dict.
    output_path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serialisable = _make_serialisable(config)
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)


def generate_template_config(dest_path):
    """Write a template ``iglusnfr_config.json`` with all defaults.

    The file includes a helpful ``_comment`` key explaining usage.

    Parameters
    ----------
    dest_path : str
        Full path to write the template file.  Parent dirs are created
        automatically.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    template = copy.deepcopy(DEFAULT_CONFIG)
    template["_comment"] = (
        "iGluSnFR Analysis Config template. "
        "Place this file in the input folder or any subfolder to customise "
        "parameters for images in that directory. "
        "Closer configs (deeper subfolders) override parent settings. "
        "You only need to include the keys you want to change – "
        "missing keys fall back to built-in defaults. "
        "Sections: 'processing' (image pipeline), 'viewing' (napari viewer), "
        "'analysis' (trace analysis & plots). "
        "ilastik_models: set model paths per experiment type "
        "(spontaneous / evoked / default).  To override for a specific "
        "subfolder, add 'ilastik_model' (singular) with the path."
    )
    serialisable = _make_serialisable(template)
    with open(dest_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    return dest_path


def check_analysis_consistency(input_root):
    """Check whether *analysis* config parameters are identical across
    all subfolders that contain a config file.

    This is important because the analysis compares groups across
    different subfolders – if parameters differ, the comparison may be
    invalid.

    Parameters
    ----------
    input_root : str
        Top-level input folder.

    Returns
    -------
    list of str
        Warning messages (empty list means all consistent).
    """
    warnings = []
    analysis_configs = {}  # subfolder -> resolved analysis config

    # Collect all subdirectories (including root) that contain images or configs
    root = os.path.realpath(input_root)
    subdirs = set()
    subdirs.add(root)
    for dirpath, dirnames, filenames in os.walk(root):
        subdirs.add(os.path.realpath(dirpath))

    # Resolve analysis config for each subdir
    for d in sorted(subdirs):
        cfg = resolve_config(d, root, section="analysis")
        rel = os.path.relpath(d, root) if d != root else "."
        analysis_configs[rel] = cfg

    if len(analysis_configs) <= 1:
        return warnings

    # Compare all configs to the root config
    root_cfg = analysis_configs.get(".", {})
    # Keys that must be consistent for valid cross-group comparison
    critical_keys = [
        "sd_multiplier", "rolling_window", "baseline_start_frame",
        "baseline_end_frame", "order", "frame_rate",
        "stim_start_frame", "response_window",
        "ppr_pulse1_frame", "ppr_pulse2_frame", "ppr_response_window",
    ]

    mismatches = {}
    for subdir, cfg in analysis_configs.items():
        if subdir == ".":
            continue
        for key in critical_keys:
            val_root = root_cfg.get(key)
            val_sub = cfg.get(key)
            if val_root != val_sub:
                mismatches.setdefault(key, []).append(
                    (subdir, val_sub, val_root)
                )

    for key, entries in mismatches.items():
        for subdir, val_sub, val_root in entries:
            warnings.append(
                f"Parameter '{key}' differs: "
                f"root={val_root}, {subdir}={val_sub}. "
                f"Cross-group comparisons may be invalid."
            )

    return warnings


def format_config_log(config, section=None):
    """Return a human-readable string summarising config values.

    Parameters
    ----------
    config : dict
        Full config or a flat section dict.
    section : str or None
        If given, only format that section.

    Returns
    -------
    str
    """
    if section and section in config:
        flat = config[section]
    elif section:
        flat = config
    else:
        flat = config

    lines = []
    for k, v in sorted(flat.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            lines.append(f"  [{k}]")
            for k2, v2 in sorted(v.items()):
                if not k2.startswith("_"):
                    lines.append(f"    {k2} = {v2}")
        else:
            lines.append(f"  {k} = {v}")
    return "\n".join(lines)


# =========================================================================
# Serialisation helper
# =========================================================================

def _make_serialisable(obj):
    """Recursively convert numpy types to native Python for JSON."""
    import numbers
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    elif isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        return float(obj)
    return obj
