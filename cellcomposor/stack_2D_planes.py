import os
import re
import numpy as np
import tifffile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def find_folders(root):
    """Return all subfolders that actually contain files."""
    out = set()
    for r, d, f in os.walk(root):
        if f:
            out.add(os.path.abspath(r))
    return sorted(out)


def tz_sort_key(name):
    """Sort slice folder names by (t, z) from *_t*_z* or *_z*."""
    m = re.search(r"_t(\d+)_z(\d+)$", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"_z(\d+)$", name)
    return (0, int(m.group(1)) if m else -1)


def discover_flat_volumes(base_folder):
    """
    Layout: base_folder / <volume_name> / <z_slice_folder> / <z_slice_folder>_final_mask.tif
    Used when there is no numeric 'animal id' path segment (see build_nested_structure).
    """
    base_folder = os.path.abspath(base_folder)
    volumes = {}
    if not os.path.isdir(base_folder):
        return volumes
    for name in sorted(os.listdir(base_folder)):
        vpath = os.path.join(base_folder, name)
        if not os.path.isdir(vpath):
            continue
        zfolders = []
        for sub in os.listdir(vpath):
            sp = os.path.join(vpath, sub)
            if not os.path.isdir(sp):
                continue
            mask = os.path.join(sp, f"{sub}_final_mask.tif")
            if os.path.exists(mask):
                zfolders.append(sub)
        if zfolders:
            volumes[name] = sorted(zfolders, key=tz_sort_key)
    return volumes


def discover_deep_flat_volumes(base_folder):
    """
    Like discover_flat_volumes, but allows an arbitrary-depth volume path under
    base_folder: .../<volume_rel>/<zfolder>/<zfolder>_final_mask.tif

    When z-folders sit directly under *base_folder* (no intermediate volume
    directory), the volume key is ``""`` so callers can handle this root-level
    case separately.
    """
    base_folder = os.path.abspath(base_folder)
    groups = defaultdict(list)
    if not os.path.isdir(base_folder):
        return {}
    for r, _dirs, files in os.walk(base_folder):
        for fn in files:
            if not fn.endswith("_final_mask.tif"):
                continue
            stem = fn[: -len("_final_mask.tif")]
            if os.path.basename(r) != stem:
                continue
            vol_rel = os.path.relpath(os.path.dirname(r), base_folder)
            if vol_rel == ".":
                vol_rel = ""
            groups[vol_rel].append(stem)
    return {k: sorted(v, key=tz_sort_key) for k, v in groups.items() if v}


def build_nested_structure(paths, base_folder):
    """
    Returns a dict keyed by the relative animal folder, which includes the middle dirs.

    Example key: "Control Mice/6 Months/148"
    Value: { "148 Slide 1 Section A2": ["148 Slide 1 Section A2_t0_z0", ...] }
    """
    base_folder = os.path.abspath(base_folder)
    grouped = defaultdict(lambda: defaultdict(list))

    for p in paths:
        rel = os.path.relpath(p, base_folder)
        parts = rel.split(os.sep)

        # locate the animal id folder (numeric)
        idx = None
        for i, part in enumerate(parts):
            if part.isdigit():
                idx = i
                break
        if idx is None or idx + 2 >= len(parts):
            continue

        prefix = os.path.join(*parts[:idx]) if idx > 0 else ""
        animal_id = parts[idx]
        animal_rel = os.path.join(prefix, animal_id) if prefix else animal_id

        section_folder = parts[idx + 1]
        zfolder = parts[idx + 2]
        grouped[animal_rel][section_folder].append(zfolder)

    for animal_rel in grouped:
        for section in grouped[animal_rel]:
            grouped[animal_rel][section] = sorted(grouped[animal_rel][section], key=tz_sort_key)

    return dict(grouped)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def should_skip(out_path):
    """Return True if the expected output already exists."""
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def stack_one_section(base_folder, out_base, animal_rel, section, zfolders, out_dtype=np.uint16):
    """
    Worker that stacks all z folders for a single section and writes the output.
    Returns a message string describing the result.
    """
    out_folder = os.path.join(out_base, animal_rel, section)
    ensure_dir(out_folder)
    out_path = os.path.join(out_folder, f"{section}_2D_stacked.tif")

    if should_skip(out_path):
        return f"Skip existing: {out_path}"

    volume = []
    missing = 0
    for zf in zfolders:
        abs_folder = os.path.join(base_folder, animal_rel, section, zf)
        tif_path = os.path.join(abs_folder, f"{zf}_final_mask.tif")
        if os.path.exists(tif_path):
            img = tifffile.imread(tif_path)
            volume.append(img)
        else:
            missing += 1

    if not volume:
        return f"No slices found for {animal_rel} | {section}  missing {missing}"

    vol = np.stack(volume, axis=0)
    tifffile.imwrite(out_path, vol.astype(out_dtype))
    return f"Saved {out_path} with shape {vol.shape}  missing {missing}"


def stack_one_flat_volume(base_folder, out_base, volume_name, zfolders, out_dtype=np.uint16):
    """Stack z-slices for layout base_folder/<volume_name>/<zfolder>/mask.tif.

    When *volume_name* is ``""`` the z-folders sit directly under *base_folder*
    (per-image output mode).  The output stem is derived from ``OUTPUT_DIR``.
    """
    if volume_name:
        mask_root = os.path.join(base_folder, volume_name)
        out_folder = os.path.join(out_base, volume_name)
        stem = os.path.basename(volume_name)
    else:
        mask_root = base_folder
        out_folder = out_base
        from pipeline_config import OUTPUT_DIR
        stem = OUTPUT_DIR.name

    ensure_dir(out_folder)
    out_path = os.path.join(out_folder, f"{stem}_2D_stacked.tif")

    if should_skip(out_path):
        return f"Skip existing: {out_path}"

    volume = []
    missing = 0
    for zf in zfolders:
        abs_folder = os.path.join(mask_root, zf)
        tif_path = os.path.join(abs_folder, f"{zf}_final_mask.tif")
        if os.path.exists(tif_path):
            img = tifffile.imread(tif_path)
            volume.append(img)
        else:
            missing += 1

    if not volume:
        return f"No slices found for flat volume {stem}  missing {missing}"

    vol = np.stack(volume, axis=0)
    tifffile.imwrite(out_path, vol.astype(out_dtype))
    return f"Saved {out_path} with shape {vol.shape}  missing {missing}"


def stack_and_save_multiproc(nested, base_folder, out_base, out_dtype=np.uint16, max_workers=16):
    """
    Prepare tasks for each section and run them in parallel.
    Skips sections that already have an output file.
    """
    tasks = []
    for animal_rel, sections in nested.items():
        for section, zfolders in sections.items():
            tasks.append((animal_rel, section, zfolders))

    if not tasks:
        print("No work to do.")
        return

    worker = partial(stack_one_section, base_folder, out_base, out_dtype=out_dtype)

    print(f"Submitting {len(tasks)} sections with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, animal_rel, section, zfolders)
                   for animal_rel, section, zfolders in tasks]
        for fut in as_completed(futures):
            try:
                msg = fut.result()
                print(msg)
            except Exception as e:
                print(f"Error in worker: {e}")


# main
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from pipeline_config import SEGMENTATION_2D_DIR, SEGMENTATION_2D_STACKED_DIR

    input_base = os.fspath(SEGMENTATION_2D_DIR)
    output_base = os.fspath(SEGMENTATION_2D_STACKED_DIR)

    folders = find_folders(input_base)
    nested = build_nested_structure(folders, input_base)
    if nested:
        stack_and_save_multiproc(nested, input_base, output_base, out_dtype=np.uint16, max_workers=16)
    else:
        flat = discover_flat_volumes(input_base)
        if not flat:
            flat = discover_deep_flat_volumes(input_base)
        if flat:
            print(f"Flat volume layout: {len(flat)} volume(s)")
            for vol_name, zfolders in flat.items():
                msg = stack_one_flat_volume(input_base, output_base, vol_name, zfolders)
                print(msg)
        else:
            print("No volumes found (neither nested animal/section nor flat volume/z-slice layout).")
