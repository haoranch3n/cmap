# CMAP Cell Exemption Napari Plugin

Separate Napari plugin for reviewing reconstructed CMAP sample volumes. It does
not modify or replace the existing `cmap/napari-plugin` package.

The plugin supports two annotation workflows:

- Label existing 3D cell crops from `cell_coordinates.csv` as `good` or `bad`.
- Draw 2D boxes on the active Z slice for missing cell masks.

## Expected Input Layout

Select the CMAP output root that contains reconstructed sample folders:

```text
output_root/
  batch_name/
    sample_name/
      cell_box_bg_sigma_488560_shape_combined.tif
      cell_box_bg_sigma_488560_shape/
        cell_coordinates.csv
```

The combined TIFF is expected to be `(Z, 5, Y, X)`:

```text
0: 642 original intensity
1: 488 original intensity
2: 560 original intensity
3: QC-pass label mask, where voxel value is cell_id
4: mask boundary
```

The coordinate CSV must contain:

```text
cell_id,z0,z1,y0,y1,x0,x1
```

## Launch (recommended): run.sh

This mirrors `napari-plugin/run.sh`. It loads the `napari` environment module
(which provides a working conda + napari + Qt + OpenGL stack), installs the
plugin if needed, and opens Napari with the widget docked:

```bash
bash /research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/cell-exemption-napari/run.sh
```

Optionally auto-discover samples under an output root and/or preselect the
annotation folder:

```bash
bash run.sh /path/to/output_no_deconv
bash run.sh /path/to/output_no_deconv --annotation-dir /path/to/annotations
```

Environment check without opening the GUI:

```bash
bash run.sh --check
```

Using `run.sh` (with `module load napari`) is the supported way to get the GUI,
because conda base Python often ships an OpenGL/`libstdc++` stack that is too old
for the system Mesa driver (see Troubleshooting below).

## Install As A Napari Plugin

To register it in an existing Napari install, use the same Python environment
that starts Napari:

```bash
module load napari   # or otherwise activate the env that provides napari
cd /research/dept/dnb/core_operations/ImageAnalysis/Core/Haoran/cmap/cell-exemption-napari
pip install -e .
```

Then start Napari and open:

```text
Plugins -> CMAP Cell Exemption -> CMAP Cell Exemption
```

If Napari is not already provided by the environment, use:

```bash
pip install -e ".[bundled]"
```

## Alternate Launcher (source tree, no module)

`run_exemption.py` runs directly from the source folder with whatever `python`
is active. If that is conda base, it auto-preloads a newer system `libstdc++`
to work around the OpenGL driver issue:

```bash
python run_exemption.py
```

Prefer `run.sh` when the `napari` module is available.

## Annotation CSV

Choose an annotation output folder in the widget. The plugin writes a timestamped
CSV:

```text
cell_exemption_annotations_YYYYMMDD_HHMMSS.csv
```

Prior files matching `cell_exemption_annotations*.csv` in the same folder are
loaded oldest-first so work can resume. Newer rows override older rows with the
same `(image_path, annotation_type, cell_id)` key.

Columns start with `image_name,cell_id`:

```text
image_name,cell_id,annotation_type,label,batch,sample,image_path,z0,z1,y0,y1,x0,x1,z_index,center_x,center_y,width,height,reviewer,notes,created_at,updated_at
```

Existing-cell rows use numeric `cell_id` values from `cell_coordinates.csv` and
`annotation_type=existing_cell`.

Missing-cell rows use synthetic IDs such as `missing_0001`, scoped to the image,
with `annotation_type=missing_cell`, `label=missing`, and `z_index` set to the
active Z slice where the rectangle was drawn.

Each missing-cell box is also mirrored into a read-only `CMAP Missing Box
Z-Extent` layer as a faint cyan outline that stays visible on every Z slice, so
the reviewer can see where a box was drawn after scrolling away from its slice.
The guide is display-only: the drawn box remains the single source of truth, and
the CSV still holds one row per box with its single `z_index`.

## Controls

- `Open Output Root...`: discover reconstructed CMAP samples.
- `Annotation Output...`: choose where CSV annotations are saved.
- `Previous` / `Next`: navigate samples with autosave.
- `Cell ID`: select an existing segmented cell.
- `Good`, `Bad`, `Clear`: set or remove the selected existing-cell label.
- `Missing Box Mode`: sets the missing-cell Shapes layer to rectangle drawing.
- `Delete Selected Missing Box`: remove selected missing-cell rectangle.
- `Save CSV`: save all loaded annotations for the current session.

Keyboard shortcuts:

```text
G: mark selected existing cell good
B: mark selected existing cell bad
U: clear selected existing-cell label
M: toggle missing-cell box mode
D/Delete/Backspace: delete selected missing-cell box
N/P: next/previous sample
Ctrl+S: save CSV
```

## Troubleshooting OpenGL (MESA-LOADER / swrast / QOpenGLWidget)

On HPC nodes with a conda base environment, Napari may fail to open a window
with errors such as:

```text
libGL error: MESA-LOADER: failed to open swrast: .../swrast_dri.so: cannot open shared object file
libGL error: failed to load driver: swrast
QOpenGLWidget: Failed to create context
```

This is caused by conda shipping an older `libstdc++` than the system Mesa
OpenGL driver requires (the system `swrast` driver needs a newer `GLIBCXX`
symbol than conda base provides).

`run_exemption.py` detects this automatically and re-execs itself with the
system `libstdc++` preloaded, so usually you can just run:

```bash
python run_exemption.py
```

To do it manually (for example when launching `napari` directly), preload the
system library:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 napari
```

To disable the automatic fix in the launcher, set:

```bash
CMAP_EXEMPTION_NO_GL_FIX=1 python run_exemption.py
```

A permanent fix is to update conda's C++ runtime:

```bash
conda install -c conda-forge "libstdcxx-ng>=12"
```

This OpenGL stack must run in a graphical session (X / VNC), not on a plain
login shell without a display.

## Development

Unit tests use synthetic directories and do not open real full-volume TIFF data:

```bash
pip install -e ".[dev]"
pytest
```

Full-volume TIFF review is interactive GUI work. Do not run batch volume IO or
long processing on the HPC login node.
