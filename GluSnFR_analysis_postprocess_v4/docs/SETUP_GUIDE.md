# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Linux | Ubuntu 22.04+ | Tested on 22.04 and 24.04 |
| NVIDIA GPU | Compute capability 6.0+ | Required for processing pipeline |
| NVIDIA Driver | >= 525 | For CUDA 12.x. Use >= 570 for Blackwell (B100/B200) |
| Conda | Miniconda or Anaconda | [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| Display | X11 or Wayland | For napari GUI |

## Step 1: Check NVIDIA Driver

```bash
nvidia-smi
```

You should see your GPU name and driver version. The driver version must be >= 525 for CUDA 12.x.

### Upgrading the Driver

For Blackwell GPUs or to get the latest driver:

```bash
# Ubuntu — add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# List available drivers
ubuntu-drivers devices

# Install specific version (example: 580)
sudo apt install nvidia-driver-580

# Reboot
sudo reboot
```

After reboot, verify with `nvidia-smi`.

### Blackwell GPU Notes

- Blackwell GPUs (B100, B200, GB200) require driver >= 570
- CUDA 12.8+ is recommended for full Blackwell support
- CuPy 12.x and 13.x both support Blackwell through CUDA 12.x builds
- No code changes are needed — the toolbox is architecture-agnostic

## Step 2: Install Conda

If conda is not installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart shell
```

If your system uses `module`:

```bash
module load conda
```

## Step 3: Run Setup

```bash
cd /path/to/GluSnFR_analysis_postprocess_v4
chmod +x setup.sh run_viewer.sh
./setup.sh
```

This creates two conda environments:

| Environment | Python | Key Packages |
|-------------|--------|-------------|
| `iglusnfr_viewer` | 3.10 | napari, plotly, polars, matplotlib, scipy |
| `iglusnfr_processing` | 3.9 | ilastik, cupy, numba, cucim, dask |

The setup script will:
1. Check your NVIDIA driver and warn if too old
2. Create the viewer environment (~3-5 min)
3. Create the processing environment (~5-15 min)
4. Verify both environments

## Step 4: Prepare an ilastik Model

The processing pipeline requires a pre-trained ilastik pixel classification project (`.ilp` file). If you don't have one:

1. Install ilastik: `conda run -n iglusnfr_processing python -m ilastik` or download from [ilastik.org](https://www.ilastik.org/)
2. Train a pixel classification model on representative images
3. Save the project as a `.ilp` file

Model naming convention:
- Models with `spon` or `spononly` in the filename are treated as **zyx** (3D spatial)
- All other models are treated as **tyx** (time + 2D spatial)

## Step 5: Launch the Viewer

```bash
./run_viewer.sh
```

Or directly:

```bash
~/.conda/envs/iglusnfr_viewer/bin/python viewer.py
```

## Troubleshooting

### "No suitable viewer conda environment found"

Run `./setup.sh` first.

### Qt/display errors

```bash
# If using SSH, enable X forwarding
ssh -X user@server

# Or use VNC/NoMachine for better performance
```

### CUDA out of memory

- Reduce `n_jobs` in the config file (default: 15)
- Process fewer images at a time
- Check GPU memory with `nvidia-smi`

### ilastik import errors

The processing environment requires Python 3.9 (ilastik constraint). Do not upgrade it.

### "module load conda" fails

Your system may not use the module system. The setup script falls back to `conda shell.bash hook` automatically.
