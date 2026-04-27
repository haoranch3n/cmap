"""
Analyze Results Module - CSV Combination and Metadata Extraction

Reads all _full_SNRlabeled_ CSV files from the output folder,
concatenates them with image name and metadata columns extracted
from the filename, and saves a combined CSV.

Metadata extracted from filenames:
  - group:                WT, APOE, etc.
  - experiment_type:      spontaneous (spon) or evoked (evok)
  - action_potentials:    e.g. 10AP -> 10
  - stimulus_frequency:   e.g. 10Hz -> 10

Based on: iGluSnFR_analysis_toolbox/synapse_analysis.py
"""

import os
import re
import pandas as pd
from pathlib import Path
from glob import glob

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


# =========================================================================
# Metadata extraction
# =========================================================================

def extract_metadata_from_filename(filename, custom_groups=None):
    """
    Extract experimental metadata encoded in an image filename.

    Parameters
    ----------
    filename : str
        Image or CSV filename (without directory path).
    custom_groups : list[str] or None
        User-supplied group names to search for (case-insensitive).
        Defaults to ``["WT", "APOE"]`` when *None*.

    Returns
    -------
    dict
        Keys: group, experiment_type, evoked_subtype, action_potentials,
              stimulus_frequency_hz
    """
    name_upper = filename.upper()

    # --- Group ---
    groups_to_check = custom_groups if custom_groups else ["WT", "APOE"]
    group = "unknown"
    for g in groups_to_check:
        if g.upper() in name_upper:
            group = g
            break

    # --- Paired-Pulse Ratio (PPR) ---
    is_ppr = bool(re.search(r"[_\-]PPR[_\-.]", name_upper) or
                  "PPR" == name_upper.split("_")[0])

    # --- Action potentials (e.g. _100AP_, _10APs_) ---
    # Negative look-behind/ahead prevent matching APOE as AP
    ap_match = re.search(r"(?<![A-Z])(\d+)APS?(?![A-Z])", name_upper)
    action_potentials = int(ap_match.group(1)) if ap_match else None

    # --- Stimulus frequency (e.g. _10Hz_, _20HZ_) ---
    hz_match = re.search(r"(?<![A-Z])(\d+)HZ(?![A-Z])", name_upper)
    stimulus_frequency_hz = int(hz_match.group(1)) if hz_match else None

    # --- Experiment type & evoked sub-type ---
    experiment_type = "unknown"
    evoked_subtype = None      # "train", "paired_pulse", "single_stim"

    if "SPON" in name_upper:
        experiment_type = "spontaneous"
    elif is_ppr:
        experiment_type = "evoked"
        evoked_subtype = "paired_pulse"
        # PPR always has 2 APs; if filename doesn't specify, set it
        if action_potentials is None:
            action_potentials = 2
    elif "EVOK" in name_upper or ap_match or hz_match:
        experiment_type = "evoked"
        if action_potentials is not None and action_potentials > 2 and hz_match:
            evoked_subtype = "train"
        elif action_potentials == 1 or (ap_match and not hz_match):
            evoked_subtype = "single_stim"
        elif hz_match:
            evoked_subtype = "train"
        else:
            evoked_subtype = "train"   # default for generic evoked

    return {
        "group": group,
        "experiment_type": experiment_type,
        "evoked_subtype": evoked_subtype,
        "action_potentials": action_potentials,
        "stimulus_frequency_hz": stimulus_frequency_hz,
    }


# =========================================================================
# CSV discovery
# =========================================================================

def find_snr_labeled_csvs(folder_path):
    """
    Find all CSV files containing ``_full_SNRlabeled_`` in their name.

    Searches first in ``csv_outputs/per_image_csv/`` (new layout), then
    falls back to the root of the output folder (legacy layout).

    Parameters
    ----------
    folder_path : str or Path
        Path to the output folder.

    Returns
    -------
    list[Path]
        Sorted list of matching CSV file paths.
    """
    folder = Path(folder_path)

    # Prefer new directory layout
    csv_dir = folder / "csv_outputs" / "per_image_csv"
    if csv_dir.exists():
        csv_files = list(csv_dir.glob("**/*_full_SNRlabeled_*.csv"))
    else:
        csv_files = []

    # Fallback: also search the root output folder (legacy)
    if not csv_files:
        csv_files = list(folder.glob("**/*_full_SNRlabeled_*.csv"))
        # Exclude anything already under csv_outputs to avoid duplicates
        csv_files = [f for f in csv_files if "csv_outputs" not in f.parts]

    return sorted(csv_files)


# =========================================================================
# Combination logic
# =========================================================================

def combine_csv_files(csv_files, frame_rate=None, custom_groups=None):
    """
    Read and concatenate multiple per-image CSV files.

    Uses **polars** for fast parallel CSV reads when available, falling
    back to pandas otherwise.

    Adds columns for:
      - ``image_name`` (reconstructed from filename)
      - ``source_csv``
      - Metadata extracted from the filename
      - ``frame_rate_hz`` and ``time_seconds`` (if *frame_rate* is provided)

    Parameters
    ----------
    csv_files : list[Path]
        CSV file paths to combine.
    frame_rate : float or None
        Imaging frame rate in Hz.  If given, a ``time_seconds`` column is
        computed from the ``slice`` column.
    custom_groups : list[str] or None
        User-supplied group names for metadata extraction.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all rows and extra metadata columns.
    """
    if _HAS_POLARS:
        return _combine_polars(csv_files, frame_rate, custom_groups)
    return _combine_pandas(csv_files, frame_rate, custom_groups)


def _combine_polars(csv_files, frame_rate, custom_groups):
    """Polars-accelerated CSV combination (5-10x faster on large files)."""
    dfs = []
    for csv_file in csv_files:
        csv_file = Path(csv_file)
        # Read everything as strings first to avoid per-file type conflicts,
        # then cast numeric columns back after concatenation.
        df = pl.read_csv(str(csv_file), infer_schema_length=0)
        # Lowercase column names
        df = df.rename({c: c.lower() for c in df.columns})

        csv_stem = csv_file.stem
        image_name = re.split(r"_full_SNRlabeled_", csv_stem)[0]
        metadata = extract_metadata_from_filename(
            image_name, custom_groups=custom_groups
        )

        ap = metadata["action_potentials"]
        hz = metadata["stimulus_frequency_hz"]

        evoked_sub = metadata.get("evoked_subtype")

        # Use explicit String dtype for lit(None) to avoid Null vs String
        # type conflicts during pl.concat across spontaneous/evoked CSVs.
        def _str_lit(val, name):
            if val is None:
                return pl.lit(None, dtype=pl.Utf8).alias(name)
            return pl.lit(str(val)).alias(name)

        new_cols = [
            pl.lit(image_name).alias("image_name"),
            pl.lit(csv_file.name).alias("source_csv"),
            pl.lit(metadata["group"]).alias("group"),
            pl.lit(metadata["experiment_type"]).alias("experiment_type"),
            _str_lit(evoked_sub, "evoked_subtype"),
            _str_lit(ap, "action_potentials"),
            _str_lit(hz, "stimulus_frequency_hz"),
        ]

        if frame_rate is not None:
            new_cols.append(pl.lit(str(frame_rate)).alias("frame_rate_hz"))
            # time_seconds computed after concat (needs numeric slice)

        df = df.with_columns(new_cols)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # All columns are Utf8/String, so concat is safe
    combined = pl.concat(dfs, how="diagonal")

    # Drop junk columns (unnamed/empty/extra columns from malformed CSVs)
    keep_cols = [c for c in combined.columns
                 if not c.lower().startswith("unnamed")
                 and c.strip() != ""
                 and not c.lower().startswith("column ")]
    combined = combined.select(keep_cols)

    # ---- Cast types -------------------------------------------------------
    _STR_COLS = {"image_name", "source_csv", "group", "experiment_type",
                 "evoked_subtype", "snr_status"}

    numeric_casts = []
    for col_name in combined.columns:
        if col_name in _STR_COLS:
            continue
        numeric_casts.append(
            pl.col(col_name).cast(pl.Float64, strict=False)
        )
    if numeric_casts:
        combined = combined.with_columns(numeric_casts)

    # Integer columns
    for int_col in ("label", "slice", "action_potentials",
                    "stimulus_frequency_hz"):
        if int_col in combined.columns:
            try:
                combined = combined.with_columns(
                    pl.col(int_col).cast(pl.Int64, strict=False)
                )
            except Exception:
                pass

    # Compute time_seconds now that slice is numeric
    if frame_rate is not None and "slice" in combined.columns:
        combined = combined.with_columns(
            (pl.col("slice").cast(pl.Float64, strict=False) / frame_rate
             ).alias("time_seconds")
        )

    pdf = combined.to_pandas()
    return _clean_columns(pdf)


def _clean_columns(df_pd):
    """Drop junk/unnamed columns from a pandas DataFrame.

    Keeps only the known pipeline columns plus any metadata columns we add.
    """
    _KNOWN = {
        "label", "centroid-0", "centroid-1", "mean_intensity", "area",
        "axis_major_length", "axis_minor_length", "slice",
        "mean_intensity_subtracted", "snr", "snr_status",
        # Morphology metrics
        "eccentricity", "solidity", "perimeter", "circularity",
        # Metadata added by this module
        "image_name", "source_csv", "group", "experiment_type",
        "evoked_subtype", "action_potentials", "stimulus_frequency_hz",
        "frame_rate_hz", "time_seconds",
    }
    keep = [c for c in df_pd.columns if c in _KNOWN]
    return df_pd[keep]


def _combine_pandas(csv_files, frame_rate, custom_groups):
    """Pandas fallback for CSV combination."""
    all_dfs = []
    for csv_file in csv_files:
        csv_file = Path(csv_file)
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()

        # Drop unnamed / junk columns immediately (cheap per-file)
        df = df[[c for c in df.columns
                 if not c.startswith("unnamed") and c.strip() != ""]]

        csv_stem = csv_file.stem
        image_name = re.split(r"_full_SNRlabeled_", csv_stem)[0]

        df["image_name"] = image_name
        df["source_csv"] = csv_file.name

        metadata = extract_metadata_from_filename(
            image_name, custom_groups=custom_groups
        )
        for key, value in metadata.items():
            df[key] = value

        if frame_rate is not None:
            df["frame_rate_hz"] = frame_rate
            if "slice" in df.columns:
                df["time_seconds"] = df["slice"] / frame_rate

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # Final cleanup: keep only known columns
    return _clean_columns(combined)


# =========================================================================
# Public API
# =========================================================================

def run_analysis(output_folder, frame_rate=None, save=True, custom_groups=None):
    """
    Main entry point: discover CSVs, combine, extract metadata, save.

    Parameters
    ----------
    output_folder : str or Path
        Root output directory produced by the processing pipeline.
    frame_rate : float or None
        Imaging frame rate in Hz (user-supplied).
    save : bool
        Whether to write the combined CSV to disk.

    Returns
    -------
    (pd.DataFrame or None, list[Path])
        The combined DataFrame and the list of source CSV files found.
    """
    csv_files = find_snr_labeled_csvs(output_folder)

    if not csv_files:
        print("No _full_SNRlabeled_ CSV files found.")
        return None, []

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f.name}")

    combined_df = combine_csv_files(csv_files, frame_rate=frame_rate,
                                     custom_groups=custom_groups)

    if save and len(combined_df) > 0:
        combined_dir = os.path.join(str(output_folder), "csv_outputs", "combined_csv")
        os.makedirs(combined_dir, exist_ok=True)
        output_path = os.path.join(combined_dir, "combined_SNRlabeled_traces.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined CSV ({len(combined_df)} rows) to: {output_path}")

    # Print quick summary
    if combined_df is not None and len(combined_df) > 0:
        print(f"\n--- Summary ---")
        print(f"Total rows:   {len(combined_df)}")
        print(f"Images:       {combined_df['image_name'].nunique()}")
        if "label" in combined_df.columns:
            print(f"Unique ROIs:  {combined_df.groupby('image_name')['label'].nunique().sum()}")
        groups = combined_df["group"].unique()
        print(f"Groups:       {', '.join(groups)}")
        exp_types = combined_df["experiment_type"].unique()
        print(f"Exp. types:   {', '.join(exp_types)}")

    return combined_df, csv_files


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <output_folder> [frame_rate]")
        sys.exit(1)

    folder = sys.argv[1]
    fr = float(sys.argv[2]) if len(sys.argv) > 2 else None

    run_analysis(folder, frame_rate=fr, save=True)
