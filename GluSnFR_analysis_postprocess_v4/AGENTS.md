# Agent Context — iGluSnFR Analysis Toolbox v3

This file provides comprehensive context for AI coding agents working on this codebase.

## Quick Orientation

- **What**: Napari-based GUI for calcium imaging analysis (iGluSnFR glutamate sensor data)
- **Who**: Neuroscience researchers comparing groups (e.g., WT vs APOE) across spontaneous, evoked, and paired-pulse experiments
- **Where**: Linux servers with NVIDIA GPUs (CUDA 12.x)
- **How**: Two conda envs — viewer (Python 3.10, napari) spawns processing (Python 3.9, ilastik, CuPy) as subprocess

## File Map with Entry Points

```
viewer.py           ← MAIN ENTRY POINT: `python viewer.py` launches napari
  ProcessingWidget    line ~664  (Tab 1: processing pipeline)
  DatasetLoaderWidget line ~1085 (Tab 2: dataset browser)
  ROIPlotterWidget    line ~1363 (Tab 2: trace plots + outlier explorer)
  AnalyzeResultsWidget line ~1780 (Tab 3: batch analysis)
  NapariViewer        line ~2400 (main class, layer management)

processing_utils.py ← GPU pipeline, runs in iglusnfr_processing env
  ProcessImages()     line ~528  (main pipeline function)

trace_analysis.py   ← Firing detection + summaries + plots
  detect_firings()    line ~66
  compute_outlier_scores() line ~541
  run_full_analysis() line ~1100 (orchestrator)

analyze_results.py  ← CSV combination + metadata extraction
  combine_csv_files() line ~200
  extract_metadata_from_filename() line ~100

outlier_explorer.py ← Interactive histogram QDialog
  OutlierExplorer     line ~60

config_manager.py   ← Hierarchical config resolution
  DEFAULT_CONFIG      line ~27  (ALL default parameters)
  resolve_config()    line ~146
```

## Common Tasks and Where to Edit

| Task | File(s) | Location |
|------|---------|----------|
| Add a new processing parameter | `config_manager.py` (DEFAULT_CONFIG.processing), `processing_utils.py` (ProcessImages args), `viewer.py` (ProcessingWidget UI) |
| Add a new analysis parameter | `config_manager.py` (DEFAULT_CONFIG.analysis), `viewer.py` (AnalyzeResultsWidget UI + _get_params/_set_params) |
| Add a new plot type | `trace_analysis.py` (new plot function + call from generate_all_plots) |
| Add a new CSV output | `trace_analysis.py` (in run_full_analysis, after summary generation) |
| Change metadata extraction | `analyze_results.py` (extract_metadata_from_filename regex patterns) |
| Add a new outlier metric | `trace_analysis.py` (OUTLIER_METRICS list + DEFAULT_OUTLIER_CONFIG), `config_manager.py` (DEFAULT_CONFIG.analysis.outlier_config), `viewer.py` (AnalyzeResultsWidget outlier UI) |
| Modify napari layer behavior | `viewer.py` (NapariViewer class: load_new_dataset, _apply_outlier_flags) |
| Change UI layout | `viewer.py` (respective Widget._setup_ui methods) |

## Known Gotchas

1. **Never `import cupy` in viewer.py** — it's only available in the processing env
2. **Never use `layer.remove()` + `add_points()` in sequence** — causes segfaults. Use in-place `layer.data = ...` updates
3. **f-string + dict embedding**: When building Python scripts as strings (ProcessingWorker), use `repr()` not `json.dumps()` to avoid brace conflicts
4. **Polars type errors**: CSV columns may have mixed types. Always read as String first, then cast
5. **napari 0.6.x**: Use `border_color` not `edge_color` for points layers
6. **ilastik requires Python 3.9** — don't upgrade the processing env
7. **Config files override GUI** unless the override checkbox is checked
8. **Settings file**: `~/.iglusnfr_settings.json` persists folder paths and parameter values between sessions

## Testing Checklist

- [ ] `python -m py_compile *.py` — all files compile
- [ ] `./run_viewer.sh` — viewer launches without errors
- [ ] Processing tab: select folders, run on test image
- [ ] View Results: load dataset, shift-click ROI, see trace
- [ ] Outlier Explorer: open, drag sliders, apply
- [ ] Analyze Results: select folder, run analysis, check output CSVs and plots

## Recent Changes (v2 → v3)

- Renamed env names: `iglusnfr_processing_validation` → `iglusnfr_processing`, `iglusnfr_viewer_validation` → `iglusnfr_viewer`
- Exposed hardcoded `n_jobs=15` and `parallel_backend="threading"` in config
- Added `ui` section to config (font sizes, panel width)
- Reordered Processing tab: Config File moved to #2 (after Input), Run Controls merged
- Reordered Analyze Results tab: Config #2, Metadata Preview #3, Groups #4, Baseline merged into Acquisition, Outlier Detection moved to #9
- Replaced z-score outlier UI in View Results with interactive Outlier Explorer (histogram + range sliders)
- Added polars, pyarrow to viewer environment
- Added NVIDIA driver version check to setup.sh
- Updated all version strings v2 → v3
