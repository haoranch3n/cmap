"""
iGluSnFR Analysis Toolbox v3 - Napari Viewer

A comprehensive napari-based viewer for processing and visualizing iGluSnFR calcium imaging data.

Features:
- Browse and select input folders with ome.tif files
- Configure and run processing pipeline
- Review accepted/rejected ROIs
- Visualize time-series traces from CSV files
- Monitor processing progress with CPU and memory utilization
"""

import numpy as np
import napari
import tifffile
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox,
    QSizePolicy, QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout, QComboBox,
    QFileDialog, QMessageBox, QProgressBar, QTextEdit, QTabWidget,
    QFrame, QSplitter, QApplication, QScrollArea, QLineEdit
)
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtCore import QUrl, QThread, Signal, QTimer, Qt
from qtpy.QtGui import QFont
import plotly.graph_objects as go
import tempfile
import os
import json
import pandas as pd
from scipy.signal import argrelextrema
from pathlib import Path
from glob import glob
import subprocess
import psutil
import time
import threading

from config_manager import (
    DEFAULT_CONFIG,
    CONFIG_FILENAME,
    resolve_config,
    resolve_config_for_processing,
    generate_template_config,
    save_resolved_config,
    check_analysis_consistency,
    format_config_log,
)

# =============================================================================
# Settings Persistence
# =============================================================================

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".iglusnfr_settings.json")

def load_settings():
    """Load settings from file."""
    defaults = {
        'input_folder': '',
        'output_folder': '',
        'model_path': '',
        'results_folder': '',
        'bkg_percentile': 10,
        'max_distance': 6,
        'min_area': 20,
        'max_area': 400,
        'snr_threshold': 3.0,
    }
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                defaults.update(saved)
    except Exception as e:
        print(f"Could not load settings: {e}")
    return defaults

def save_settings(settings):
    """Save settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Could not save settings: {e}")

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Toolbox configuration settings."""
    # UI (from config_manager single source of truth)
    _ui = DEFAULT_CONFIG.get("ui", {})
    FONT_SIZE = _ui.get("font_size", 11)
    FONT_SIZE_LARGE = _ui.get("font_size_large", 13)
    FONT_SIZE_TITLE = _ui.get("font_size_title", 14)
    
    # Processing defaults (from config_manager single source of truth)
    _proc = DEFAULT_CONFIG["processing"]
    DEFAULT_BKG_PERCENTILE = _proc["bkg_percentile"]
    DEFAULT_MAX_DISTANCE = _proc["max_distance"]
    DEFAULT_SNR_THRESHOLD = _proc["snr_threshold"]
    DEFAULT_N_JOBS = _proc["n_jobs"]
    
    # Paths - everything is relative to this script's directory
    TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Utility Functions
# =============================================================================

def get_stylesheet():
    """Return stylesheet with larger fonts."""
    return f"""
        QWidget {{
            font-size: {Config.FONT_SIZE}px;
        }}
        QGroupBox {{
            font-size: {Config.FONT_SIZE_LARGE}px;
            font-weight: bold;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            padding: 5px;
        }}
        QPushButton {{
            font-size: {Config.FONT_SIZE}px;
            padding: 6px 12px;
        }}
        QLabel {{
            font-size: {Config.FONT_SIZE}px;
        }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            font-size: {Config.FONT_SIZE}px;
            padding: 4px;
        }}
        QTextEdit {{
            font-family: monospace;
            font-size: 10px;
        }}
    """


def detect_synapse_firings(signal, frames, sd_multiplier=4.0, rolling_window=10,
                           baseline_start_frame=0, baseline_end_frame=49,
                           order=1):
    """Detect synapse firings using local maxima above threshold.

    Baseline SD is estimated from the frame window
    [baseline_start_frame, baseline_end_frame] (inclusive), clamped to
    the actual trace length.
    """
    signal = np.asarray(signal, dtype=float)
    frames = np.asarray(frames)
    n = len(signal)

    # Map frame numbers to array indices
    bl_idx_start = int(np.searchsorted(frames, baseline_start_frame, side="left"))
    bl_idx_end = int(np.searchsorted(frames, baseline_end_frame, side="right"))
    bl_idx_start = max(0, min(bl_idx_start, n - 1))
    bl_idx_end = max(bl_idx_start + 1, min(bl_idx_end, n))

    baseline_signal = signal[bl_idx_start:bl_idx_end]
    if len(baseline_signal) == 0:
        baseline_signal = signal
    baseline_sd = float(np.std(baseline_signal))
    baseline_mean = float(np.mean(baseline_signal))

    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=rolling_window, center=True, min_periods=1).mean().values
    threshold = rolling_mean + (sd_multiplier * baseline_sd)

    local_max_indices = argrelextrema(signal, np.greater, order=order)[0]
    accepted_indices = [idx for idx in local_max_indices if signal[idx] > threshold[idx]]

    return {
        'rolling_mean': rolling_mean,
        'threshold': threshold,
        'baseline_sd': baseline_sd,
        'baseline_mean': baseline_mean,
        'local_max_indices': local_max_indices,
        'accepted_indices': accepted_indices,
        'local_max_frames': frames[local_max_indices] if len(local_max_indices) > 0 else np.array([]),
        'local_max_values': signal[local_max_indices] if len(local_max_indices) > 0 else np.array([]),
        'accepted_frames': frames[accepted_indices] if len(accepted_indices) > 0 else np.array([]),
        'accepted_values': signal[accepted_indices] if len(accepted_indices) > 0 else np.array([])
    }


def get_env_python_path(env_name):
    """Get the python executable path for a conda environment."""
    home = os.path.expanduser("~")
    env_python = os.path.join(home, ".conda", "envs", env_name, "bin", "python")
    if os.path.exists(env_python):
        return env_python
    return None


# =============================================================================
# Resource Monitor
# =============================================================================

class ResourceMonitor(QWidget):
    """Widget for monitoring CPU and memory usage."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.process = None
        self.child_pids = set()
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_stats)
        self.timer.start(500)  # Update every 500ms
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_label = QLabel("CPU:")
        cpu_label.setFixedWidth(50)
        cpu_layout.addWidget(cpu_label)
        
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100 * os.cpu_count())
        self.cpu_bar.setTextVisible(True)
        self.cpu_bar.setFormat("%v%")
        cpu_layout.addWidget(self.cpu_bar)
        layout.addLayout(cpu_layout)
        
        # Memory usage
        mem_layout = QHBoxLayout()
        mem_label = QLabel("RAM:")
        mem_label.setFixedWidth(50)
        mem_layout.addWidget(mem_label)
        
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 100)
        self.mem_bar.setTextVisible(True)
        self.mem_bar.setFormat("%v%")
        mem_layout.addWidget(self.mem_bar)
        layout.addLayout(mem_layout)
        
        # Stats label
        self.stats_label = QLabel("Idle")
        self.stats_label.setStyleSheet(f"font-size: 10px; color: #666;")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def set_process(self, pid):
        """Set the main process to monitor."""
        try:
            self.process = psutil.Process(pid)
        except:
            self.process = None
    
    def _update_stats(self):
        """Update resource statistics."""
        try:
            if self.process and self.process.is_running():
                # Get main process and children
                try:
                    children = self.process.children(recursive=True)
                except:
                    children = []
                
                # Calculate total CPU and memory
                total_cpu = 0
                total_mem = 0
                n_threads = 0
                
                # Main process
                try:
                    total_cpu += self.process.cpu_percent()
                    mem_info = self.process.memory_info()
                    total_mem += mem_info.rss
                    n_threads += self.process.num_threads()
                except:
                    pass
                
                # Child processes
                for child in children:
                    try:
                        total_cpu += child.cpu_percent()
                        mem_info = child.memory_info()
                        total_mem += mem_info.rss
                        n_threads += child.num_threads()
                    except:
                        pass
                
                # Update bars
                self.cpu_bar.setValue(int(total_cpu))
                
                # Memory as percentage of total system memory
                total_system_mem = psutil.virtual_memory().total
                mem_percent = (total_mem / total_system_mem) * 100
                self.mem_bar.setValue(int(mem_percent))
                
                # Format memory in GB
                mem_gb = total_mem / (1024**3)
                n_procs = 1 + len(children)
                self.stats_label.setText(
                    f"Procs: {n_procs} | Threads: {n_threads} | Mem: {mem_gb:.1f} GB"
                )
            else:
                # Show system stats when idle
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                self.cpu_bar.setValue(int(cpu))
                self.mem_bar.setValue(int(mem))
                self.stats_label.setText("Idle - System stats")
        except Exception as e:
            self.stats_label.setText(f"Monitor error")


# =============================================================================
# Processing Worker
# =============================================================================

class ProcessingWorker(QThread):
    """Worker thread for running image processing pipeline."""
    
    progress = Signal(str)
    cpu_update = Signal(float)
    pid_update = Signal(int)
    finished = Signal(bool, str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.process = None
        self.should_stop = False
    
    def run(self):
        """Run the processing pipeline."""
        try:
            script_content = self._build_processing_script()
            script_path = os.path.join(self.params['output_dir'], '_temp_process.py')
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            self.progress.emit("Starting processing...")
            
            # Use processing environment (ilastik or validation env)
            proc_env = self.params.get('processing_env', 'ilastik')
            
            # Get python path for the environment
            env_python = get_env_python_path(proc_env)
            if not env_python:
                self.finished.emit(False, f"Could not find python for environment: {proc_env}")
                return
            
            self.progress.emit(f"Using environment: {proc_env}")
            
            cmd = f'''export MPLBACKEND=Agg && \
                      "{env_python}" "{script_path}"'''
            
            self.process = subprocess.Popen(
                cmd, shell=True, executable='/bin/bash',
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            
            # Emit PID for monitoring
            self.pid_update.emit(self.process.pid)
            
            # Track success/failure counts from the script's own summary output
            import re
            n_successful = None
            n_failed = None
            n_total = None
            
            while self.process.poll() is None:
                if self.should_stop:
                    self.process.terminate()
                    self.progress.emit("Processing stopped by user")
                    self.finished.emit(False, "Stopped by user")
                    return
                
                line = self.process.stdout.readline()
                if line:
                    stripped = line.strip()
                    self.progress.emit(stripped)
                    # Parse summary lines: "Successful: 5/5" and "Failed: 0/5"
                    m = re.match(r'Successful:\s*(\d+)/(\d+)', stripped)
                    if m:
                        n_successful = int(m.group(1))
                        n_total = int(m.group(2))
                    m = re.match(r'Failed:\s*(\d+)/(\d+)', stripped)
                    if m:
                        n_failed = int(m.group(1))
            
            # Read any remaining output
            remaining = self.process.stdout.read()
            if remaining:
                for line in remaining.strip().split('\n'):
                    stripped = line.strip()
                    self.progress.emit(stripped)
                    m = re.match(r'Successful:\s*(\d+)/(\d+)', stripped)
                    if m:
                        n_successful = int(m.group(1))
                        n_total = int(m.group(2))
                    m = re.match(r'Failed:\s*(\d+)/(\d+)', stripped)
                    if m:
                        n_failed = int(m.group(1))
            
            # Determine success from the script's own summary (more reliable than
            # exit code, which can be non-zero due to cleanup handlers in GPU libs)
            if n_failed is not None and n_successful is not None:
                if n_failed == 0:
                    self.finished.emit(True, f"Processing completed: {n_successful}/{n_total} successful, {n_failed} failed")
                else:
                    self.finished.emit(False, f"Processing completed: {n_successful}/{n_total} successful, {n_failed} failed")
            elif self.process.returncode != 0:
                self.finished.emit(False, "Processing failed - check log for details")
            else:
                self.finished.emit(True, "Processing completed")
            
            try:
                os.remove(script_path)
            except:
                pass
                
        except Exception as e:
            import traceback
            self.progress.emit(f"Error: {str(e)}")
            self.progress.emit(traceback.format_exc())
            self.finished.emit(False, str(e))
    
    def _build_processing_script(self):
        """Build the processing script content.

        The script now uses ``config_manager.resolve_config`` to obtain
        per-subfolder parameters, logs them for each image, and saves
        a copy of the resolved config (``iglusnfr_config_used.json``)
        in each output subfolder.
        """
        p = self.params
        
        # All code is local — use this script's directory
        toolbox_dir = Config.TOOLBOX_DIR
        
        # Serialise dicts so the subprocess can use them
        # Use repr() to get Python literals that are safe inside the f-string
        gui_overrides_repr = repr(p.get('gui_processing_overrides', {}))
        gui_models_repr = repr(p.get('gui_models', {}))
        use_gui = p.get('use_gui_overrides', False)
        
        script = f'''
import os
import sys
import json
import glob
import datetime
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, "{toolbox_dir}")

from processing_utils import ProcessImages
import processing_utils
from config_manager import (resolve_config, resolve_model_path,
                            save_resolved_config, format_config_log)
processing_utils.readSegmented = {p['skip_segmentation']}

input_dir = "{p['input_dir']}"
output_dir = "{p['output_dir']}"
norm_dir = os.path.join(output_dir, "BKG_subtracted_normalized")
seg_dir = os.path.join(output_dir, "Segmented")
log_dir = os.path.join(output_dir, "logs")

# GUI model paths (spontaneous / evoked / default)
gui_models = {gui_models_repr}

# GUI overrides (used only when toggle is on)
gui_overrides = {gui_overrides_repr}
use_gui_overrides = {use_gui}

csv_per_image_dir = os.path.join(output_dir, "csv_outputs", "per_image_csv")
csv_combined_dir = os.path.join(output_dir, "csv_outputs", "combined_csv")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(norm_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(csv_per_image_dir, exist_ok=True)
os.makedirs(csv_combined_dir, exist_ok=True)

class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.log_file = None

    def set_log_file(self, path):
        self.close_log_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.log_file = open(path, 'w')

    def close_log_file(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        if self.log_file is not None:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        if self.log_file is not None:
            self.log_file.flush()

tee = TeeWriter(sys.stdout)
sys.stdout = tee
sys.stderr = TeeWriter(sys.stderr)
sys.stderr.log_file = None  # stderr will share log file via reference below

imagelist = {p['image_list']}

print("="*60)
print(f"Processing {{len(imagelist)}} images")
print(f"Skip segmentation: {p['skip_segmentation']}")
print(f"Config override from GUI: {{use_gui_overrides}}")
print(f"Log directory: {{log_dir}}")
print("="*60)

# Pre-resolve configs per subfolder to avoid repeated I/O
_cfg_cache = {{}}

def _get_processing_cfg(image_path):
    image_dir = os.path.dirname(os.path.realpath(image_path))
    if image_dir not in _cfg_cache:
        full_cfg = resolve_config(
            image_dir, input_dir,
            gui_overrides=gui_overrides if use_gui_overrides else None,
            use_gui_overrides=use_gui_overrides,
            section="processing")
        _cfg_cache[image_dir] = full_cfg
    return _cfg_cache[image_dir]

errors = []
successful = []

for i, imfile in enumerate(imagelist):
    basename = os.path.basename(imfile)

    # Resolve config for this image's subfolder
    cfg = _get_processing_cfg(imfile)

    MaxDistance     = cfg.get("max_distance", {p['max_distance']})
    BKGpercentile  = cfg.get("bkg_percentile", {p['bkg_percentile']})
    min_area       = cfg.get("min_area", {p['min_area']})
    max_area       = cfg.get("max_area", {p['max_area']})
    snr_threshold  = cfg.get("snr_threshold", {p['snr_threshold']})
    snr_start_frame = cfg.get("snr_start_frame", None)
    snr_end_frame   = cfg.get("snr_end_frame", None)

    # Resolve ilastik model for this image (auto-detect spon vs evoked)
    try:
        model = resolve_model_path(
            imfile, input_dir, gui_models=gui_models,
            gui_overrides=gui_overrides if use_gui_overrides else None,
            use_gui_overrides=use_gui_overrides)
        print(f"  Model: {{os.path.basename(model)}}")
    except FileNotFoundError as model_err:
        print(f"ERROR: {{basename}} - {{model_err}}")
        errors.append(f"{{basename}}: {{str(model_err)}}")
        continue

    # Compute relative path from input_dir to mirror folder structure in logs/
    rel_path = os.path.relpath(imfile, input_dir)
    log_subdir = os.path.join(log_dir, os.path.dirname(rel_path))
    log_filename = os.path.splitext(basename)[0]
    if '.ome' in basename:
        log_filename = basename.split('.ome')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_subdir, f"{{log_filename}}_{{timestamp}}.log")

    # Point both stdout and stderr tee writers to this image's log file
    tee.set_log_file(log_path)
    sys.stderr.log_file = tee.log_file

    print(f"\\n[{{i+1}}/{{len(imagelist)}}] Processing: {{basename}}")
    print(f"Log file: {{log_path}}")
    print("--- Resolved processing parameters ---")
    print(format_config_log(cfg, section=None))
    print("--------------------------------------")

    # Save resolved config in the output subfolder (mirroring input structure)
    out_sub = os.path.join(output_dir, os.path.dirname(rel_path))
    os.makedirs(out_sub, exist_ok=True)
    used_cfg_path = os.path.join(out_sub, "iglusnfr_config_used.json")
    if not os.path.exists(used_cfg_path):
        save_resolved_config(cfg, used_cfg_path)

    try:
        ProcessImages(
            filepath=imfile,
            input_dir=input_dir,
            norm_dir=norm_dir,
            seg_dir=seg_dir,
            output_dir=output_dir,
            model=model,
            MaxDistance=MaxDistance,
            BKGpercentile=BKGpercentile,
            min_area=min_area,
            max_area=max_area,
            snr_threshold=snr_threshold,
            snr_start_frame=snr_start_frame,
            snr_end_frame=snr_end_frame
        )
        
        # Verify outputs were created (search recursively since output mirrors input subfolder structure)
        base = basename.split('.ome')[0] + '.ome' if '.ome' in basename else basename.rsplit('.', 1)[0]
        expected_csv = glob.glob(os.path.join(csv_per_image_dir, "**", f"{{base}}*_full_SNRlabeled_*.csv"), recursive=True)
        expected_tif = glob.glob(os.path.join(seg_dir, "**", f"{{base}}*_objects_accepted_*.tif"), recursive=True)
        
        if expected_csv and expected_tif:
            print(f"SUCCESS: {{basename}} - outputs verified")
            successful.append(basename)
        else:
            print(f"ERROR: {{basename}} - expected outputs not found")
            errors.append(f"{{basename}}: outputs not created")
            
    except Exception as e:
        print(f"ERROR: {{basename}} - {{e}}")
        import traceback
        traceback.print_exc()
        errors.append(f"{{basename}}: {{str(e)}}")

# Close per-image log and write the summary to a session-level log
tee.close_log_file()
session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_log_path = os.path.join(log_dir, f"session_summary_{{session_timestamp}}.log")
tee.set_log_file(summary_log_path)
sys.stderr.log_file = tee.log_file

print("\\n" + "="*60)
print("PROCESSING SUMMARY")
print("="*60)
print(f"Successful: {{len(successful)}}/{{len(imagelist)}}")
print(f"Failed: {{len(errors)}}/{{len(imagelist)}}")

if errors:
    print("\\nErrors:")
    for err in errors:
        print(f"  - {{err}}")
    print("\\nPROCESSING FAILED")
    tee.close_log_file()
    sys.exit(1)
else:
    print("\\nALL PROCESSING COMPLETED SUCCESSFULLY")
    tee.close_log_file()
    sys.exit(0)
'''
        return script
    
    def stop(self):
        """Stop the processing."""
        self.should_stop = True
        if self.process:
            self.process.terminate()


# =============================================================================
# Processing Widget
# =============================================================================

class ProcessingWidget(QWidget):
    """Widget for configuring and running processing pipeline."""
    
    def __init__(self, viewer_instance):
        super().__init__()
        self.viewer_instance = viewer_instance
        self.input_folder = None
        self.output_folder = None
        self.image_list = []
        self.worker = None
        self.settings = load_settings()
        
        self._setup_ui()
        self._load_saved_settings()
    
    def _setup_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Title (fixed, non-resizable)
        title = QLabel("Processing Pipeline")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setFixedHeight(25)
        layout.addWidget(title)

        # ── 1. Input Data ──────────────────────────────────────────────
        input_group = QGroupBox("1. Input Data")
        input_group.setMaximumHeight(80)
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(5, 5, 5, 5)
        input_layout.setSpacing(2)

        input_browse = QHBoxLayout()
        self.input_label = QLabel("No folder selected")
        self.input_label.setWordWrap(True)
        self.input_label.setStyleSheet("color: #666; font-size: 10px;")
        input_browse.addWidget(self.input_label)

        input_btn = QPushButton("Browse...")
        input_btn.clicked.connect(self._browse_input)
        input_browse.addWidget(input_btn)
        input_layout.addLayout(input_browse)

        self.image_count_label = QLabel("")
        self.image_count_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        input_layout.addWidget(self.image_count_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # ── 2. Config File ─────────────────────────────────────────────
        cfg_group = QGroupBox("2. Config File")
        cfg_layout = QVBoxLayout()
        cfg_layout.setContentsMargins(5, 5, 5, 5)
        cfg_layout.setSpacing(4)

        cfg_help = QLabel(
            "Place iglusnfr_config.json in the input folder "
            "(or subfolders) to customise parameters per dataset.")
        cfg_help.setWordWrap(True)
        cfg_help.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        cfg_layout.addWidget(cfg_help)

        self.gui_override_checkbox = QCheckBox(
            "Override config file with GUI values")
        self.gui_override_checkbox.setToolTip(
            "When checked, the parameter values set below in the GUI\n"
            "will take precedence over any iglusnfr_config.json files\n"
            "found in the input folder.  Useful for quick experiments\n"
            "before committing changes to the config file.")
        self.gui_override_checkbox.setChecked(False)
        cfg_layout.addWidget(self.gui_override_checkbox)

        cfg_btn_row = QHBoxLayout()
        gen_cfg_btn = QPushButton("Generate Template Config")
        gen_cfg_btn.setToolTip(
            f"Write a {CONFIG_FILENAME} template with all default\n"
            "values into the input folder.  You can then edit it to\n"
            "customise processing parameters per subfolder.")
        gen_cfg_btn.clicked.connect(self._generate_template_config)
        cfg_btn_row.addWidget(gen_cfg_btn)
        cfg_layout.addLayout(cfg_btn_row)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        # ── 3. Output Directory ────────────────────────────────────────
        output_group = QGroupBox("3. Output Directory")
        output_group.setMaximumHeight(60)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(5, 5, 5, 5)

        output_browse = QHBoxLayout()
        self.output_label = QLabel("No folder selected")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: #666; font-size: 10px;")
        output_browse.addWidget(self.output_label)

        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output)
        output_browse.addWidget(output_btn)
        output_layout.addLayout(output_browse)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # ── 4. ilastik Models ─────────────────────────────────────────
        model_group = QGroupBox("4. ilastik Models")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(5, 5, 5, 5)
        model_layout.setSpacing(3)

        model_help = QLabel(
            "Auto-selects the correct model per image based on folder "
            "name (spon/spont → Spontaneous, evoked → Evoked). "
            "Default is used when no keyword matches.")
        model_help.setWordWrap(True)
        model_help.setStyleSheet(
            "color: #666; font-size: 9px; font-style: italic;")
        model_layout.addWidget(model_help)

        # Store model paths: key -> path string
        self.model_paths = {
            "spontaneous": None,
            "evoked": None,
            "default": None,
        }
        self._model_labels = {}

        for model_key, display_name in [
            ("spontaneous", "Spontaneous:"),
            ("evoked", "Evoked:"),
            ("default", "Default (optional):"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl_name = QLabel(display_name)
            lbl_name.setFixedWidth(110)
            lbl_name.setStyleSheet("font-size: 10px; font-weight: bold;")
            row.addWidget(lbl_name)

            lbl_path = QLabel("(none)")
            lbl_path.setWordWrap(True)
            lbl_path.setStyleSheet("color: #666; font-size: 10px;")
            lbl_path.setMinimumWidth(80)
            row.addWidget(lbl_path, stretch=1)
            self._model_labels[model_key] = lbl_path

            btn = QPushButton("Browse...")
            btn.setMaximumWidth(65)
            btn.clicked.connect(
                lambda checked, k=model_key: self._browse_model(k))
            row.addWidget(btn)

            clr = QPushButton("Clear")
            clr.setMaximumWidth(42)
            clr.setStyleSheet("font-size: 9px;")
            clr.clicked.connect(
                lambda checked, k=model_key: self._clear_model(k))
            row.addWidget(clr)

            model_layout.addLayout(row)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # ── 5. Processing Parameters ──────────────────────────────────
        params_group = QGroupBox("5. Processing Parameters")
        params_group.setMinimumHeight(140)
        params_layout = QFormLayout()
        params_layout.setContentsMargins(5, 5, 5, 5)
        params_layout.setSpacing(4)

        self.bkg_percentile_spin = QSpinBox()
        self.bkg_percentile_spin.setRange(1, 50)
        self.bkg_percentile_spin.setValue(Config.DEFAULT_BKG_PERCENTILE)
        self.bkg_percentile_spin.setToolTip("Percentile for background estimation")
        params_layout.addRow("Background Percentile:", self.bkg_percentile_spin)

        self.max_distance_spin = QSpinBox()
        self.max_distance_spin.setRange(1, 20)
        self.max_distance_spin.setValue(Config.DEFAULT_MAX_DISTANCE)
        self.max_distance_spin.setToolTip("Maximum distance for spatial clustering (pixels)")
        params_layout.addRow("Max Cluster Distance:", self.max_distance_spin)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100)
        self.min_area_spin.setValue(20)
        self.min_area_spin.setToolTip("Minimum object area (pixels)")
        params_layout.addRow("Min Area:", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(50, 1000)
        self.max_area_spin.setValue(400)
        self.max_area_spin.setToolTip("Maximum object area (pixels)")
        params_layout.addRow("Max Area:", self.max_area_spin)

        self.snr_threshold_spin = QDoubleSpinBox()
        self.snr_threshold_spin.setRange(1.0, 10.0)
        self.snr_threshold_spin.setValue(3.0)
        self.snr_threshold_spin.setSingleStep(0.5)
        self.snr_threshold_spin.setToolTip("SNR threshold for accepting clusters")
        params_layout.addRow("SNR Threshold:", self.snr_threshold_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ── 6. Run Controls ───────────────────────────────────────────
        run_group = QGroupBox("6. Run")
        run_layout = QVBoxLayout()
        run_layout.setContentsMargins(5, 5, 5, 5)
        run_layout.setSpacing(4)

        self.skip_seg_checkbox = QCheckBox("Skip segmentation (use existing)")
        run_layout.addWidget(self.skip_seg_checkbox)

        # Processing environment selection
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("Processing Env:"))
        self.proc_env_combo = QComboBox()
        conda_envs = os.path.join(os.path.expanduser("~"), ".conda", "envs")
        for env_name in ["iglusnfr_processing", "ilastik"]:
            env_path = os.path.join(conda_envs, env_name, "bin", "python")
            if os.path.exists(env_path):
                self.proc_env_combo.addItem(env_name, env_name)
        if self.proc_env_combo.count() == 0:
            self.proc_env_combo.addItem("(no env found)", "")
        env_layout.addWidget(self.proc_env_combo, stretch=1)
        run_layout.addLayout(env_layout)

        # Run / Stop buttons in same row
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Processing")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_processing)
        self.run_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white;")
        button_layout.addWidget(self.stop_btn)
        run_layout.addLayout(button_layout)

        run_group.setLayout(run_layout)
        layout.addWidget(run_group)

        # ── 7. Log ────────────────────────────────────────────────────
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.log_text)

        # ── 8. Navigate to View Results ───────────────────────────────
        self.view_results_btn = QPushButton("Go to View Results Tab")
        self.view_results_btn.clicked.connect(self._go_to_view_results)
        self.view_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; color: white; font-weight: bold;
                padding: 8px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
        """)
        layout.addWidget(self.view_results_btn)

        inner.setLayout(layout)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
    
    def _load_saved_settings(self):
        """Load and apply saved settings."""
        # Input folder
        if self.settings.get('input_folder') and os.path.isdir(self.settings['input_folder']):
            self.input_folder = self.settings['input_folder']
            self.input_label.setText(self.input_folder)
            self._scan_input_folder()
        
        # Output folder
        if self.settings.get('output_folder') and os.path.isdir(self.settings['output_folder']):
            self.output_folder = self.settings['output_folder']
            self.output_label.setText(self.output_folder)
        
        # Model paths (3 slots)
        for key in ("spontaneous", "evoked", "default"):
            settings_key = f"model_path_{key}"
            path = self.settings.get(settings_key, '')
            if path and os.path.isfile(path):
                self.model_paths[key] = path
                self._model_labels[key].setText(os.path.basename(path))
        # Backward-compat: old single model_path → try to classify
        if (not any(self.model_paths.values())
                and self.settings.get('model_path')
                and os.path.isfile(self.settings['model_path'])):
            old = self.settings['model_path']
            self.model_paths["default"] = old
            self._model_labels["default"].setText(os.path.basename(old))
        
        # Parameters
        if 'bkg_percentile' in self.settings:
            self.bkg_percentile_spin.setValue(self.settings['bkg_percentile'])
        if 'max_distance' in self.settings:
            self.max_distance_spin.setValue(self.settings['max_distance'])
        if 'min_area' in self.settings:
            self.min_area_spin.setValue(self.settings['min_area'])
        if 'max_area' in self.settings:
            self.max_area_spin.setValue(self.settings['max_area'])
        if 'snr_threshold' in self.settings:
            self.snr_threshold_spin.setValue(self.settings['snr_threshold'])
        self._update_run_button()
    
    def _save_settings(self):
        """Save current settings."""
        self.settings['input_folder'] = self.input_folder or ''
        self.settings['output_folder'] = self.output_folder or ''
        for key in ("spontaneous", "evoked", "default"):
            self.settings[f'model_path_{key}'] = self.model_paths.get(key) or ''
        self.settings['bkg_percentile'] = self.bkg_percentile_spin.value()
        self.settings['max_distance'] = self.max_distance_spin.value()
        self.settings['min_area'] = self.min_area_spin.value()
        self.settings['max_area'] = self.max_area_spin.value()
        self.settings['snr_threshold'] = self.snr_threshold_spin.value()
        save_settings(self.settings)
    
    def _browse_input(self):
        start_dir = self.settings.get('input_folder', '')
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", start_dir)
        if folder:
            self.input_folder = folder
            self.input_label.setText(folder)
            self._scan_input_folder()
            self._save_settings()
    
    def _scan_input_folder(self):
        if not self.input_folder:
            return
        self.image_list = sorted(glob(os.path.join(self.input_folder, '**/*ome.tif'), recursive=True))
        n = len(self.image_list)
        self.image_count_label.setText(f"Found {n} image(s)" if n > 0 else "No ome.tif files found")
        self._update_run_button()
    
    def _browse_output(self):
        start_dir = self.settings.get('output_folder', '')
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if folder:
            self.output_folder = folder
            self.output_label.setText(folder)
            self._update_run_button()
            self._save_settings()
    
    def _browse_model(self, model_key):
        """Open a file dialog to pick an ilastik model for *model_key*."""
        # Start from the directory of any already-set model
        start_dir = ''
        for k in (model_key, "spontaneous", "evoked", "default"):
            p = self.model_paths.get(k)
            if p:
                start_dir = os.path.dirname(p)
                break
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select ilastik Model ({model_key})",
            start_dir, "ilastik Project (*.ilp)")
        if file_path:
            self.model_paths[model_key] = file_path
            self._model_labels[model_key].setText(os.path.basename(file_path))
            self._update_run_button()
            self._save_settings()

    def _clear_model(self, model_key):
        """Clear the model path for *model_key*."""
        self.model_paths[model_key] = None
        self._model_labels[model_key].setText("(none)")
        self._update_run_button()
        self._save_settings()
    
    def _update_run_button(self):
        has_model = any(self.model_paths.values())
        can_run = all([self.input_folder, self.output_folder,
                       has_model, self.image_list])
        self.run_btn.setEnabled(can_run)
    
    def _run_processing(self):
        if self.worker and self.worker.isRunning():
            return
        
        # Build gui_models dict (only non-None entries)
        gui_models = {k: v for k, v in self.model_paths.items() if v}

        params = {
            'input_dir': self.input_folder,
            'output_dir': self.output_folder,
            'gui_models': gui_models,
            'image_list': self.image_list,
            'bkg_percentile': self.bkg_percentile_spin.value(),
            'max_distance': self.max_distance_spin.value(),
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value(),
            'snr_threshold': self.snr_threshold_spin.value(),
            'skip_segmentation': self.skip_seg_checkbox.isChecked(),
            'processing_env': self.proc_env_combo.currentData(),
            'use_gui_overrides': self.gui_override_checkbox.isChecked(),
            'gui_processing_overrides': self._get_gui_processing_overrides(),
        }
        
        self._log("Starting processing...")
        
        self.worker = ProcessingWorker(params)
        self.worker.progress.connect(self._log)
        self.worker.pid_update.connect(self._on_pid_update)
        self.worker.finished.connect(self._on_finished)
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()
    
    def _stop_processing(self):
        if self.worker:
            self.worker.stop()
            self._log("Stopping...")
    
    def _on_pid_update(self, pid):
        pass  # PID received, could be used for monitoring
    
    def _on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self._log(f"SUCCESS: {message}")
            QMessageBox.information(self, "Complete", message)
        else:
            self._log(f"FAILED: {message}")
    
    def _go_to_view_results(self):
        if hasattr(self.viewer_instance, 'tab_widget'):
            self.viewer_instance.tab_widget.setCurrentIndex(1)
    
    def _log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _generate_template_config(self):
        """Write a template iglusnfr_config.json into the input folder."""
        if not self.input_folder:
            QMessageBox.warning(
                self, "No input folder",
                "Please select an input folder first.")
            return
        dest = os.path.join(self.input_folder, CONFIG_FILENAME)
        if os.path.isfile(dest):
            reply = QMessageBox.question(
                self, "File exists",
                f"{CONFIG_FILENAME} already exists in the input folder.\n"
                "Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        try:
            generate_template_config(dest)
            self._log(f"Template config written to: {dest}")
            QMessageBox.information(
                self, "Config generated",
                f"Template written to:\n{dest}\n\n"
                "Edit this file to customise parameters.\n"
                "Copy it into subfolders for per-subfolder overrides.")
        except Exception as e:
            self._log(f"ERROR generating template: {e}")

    def _get_gui_processing_overrides(self):
        """Collect the current GUI processing parameter values."""
        return {
            "bkg_percentile": self.bkg_percentile_spin.value(),
            "max_distance": self.max_distance_spin.value(),
            "min_area": self.min_area_spin.value(),
            "max_area": self.max_area_spin.value(),
            "snr_threshold": self.snr_threshold_spin.value(),
        }


# =============================================================================
# Dataset Loader Widget
# =============================================================================

class DatasetLoaderWidget(QWidget):
    """Widget for browsing and loading processed datasets."""
    
    def __init__(self, viewer_instance):
        super().__init__()
        self.viewer_instance = viewer_instance
        self.datasets = {}
        self.current_folder = None
        self.csv_data = {}
        self.settings = load_settings()
        
        self._setup_ui()
        self._load_saved_folder()
    
    def _setup_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title
        title = QLabel("View Results")
        title.setStyleSheet(f"font-size: {Config.FONT_SIZE_TITLE}px; font-weight: bold;")
        layout.addWidget(title)
        
        # Folder selection
        folder_group = QGroupBox("Output Folder")
        folder_layout = QVBoxLayout()
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #666;")
        folder_layout.addWidget(self.folder_label)
        
        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        btn_layout.addWidget(browse_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        btn_layout.addWidget(refresh_btn)
        folder_layout.addLayout(btn_layout)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()
        
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEnabled(False)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_selected)
        ds_layout.addWidget(self.dataset_combo, stretch=1)
        dataset_layout.addLayout(ds_layout)
        
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI Type:"))
        self.label_type_combo = QComboBox()
        self.label_type_combo.addItem("Accepted ROIs", "accepted")
        self.label_type_combo.addItem("Rejected ROIs", "rejected")
        self.label_type_combo.setEnabled(False)
        roi_layout.addWidget(self.label_type_combo, stretch=1)
        dataset_layout.addLayout(roi_layout)
        
        self.info_label = QLabel("Select a folder to see datasets")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        dataset_layout.addWidget(self.info_label)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Load button
        self.load_btn = QPushButton("LOAD SELECTED DATASET")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_dataset)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; font-weight: bold;
                font-size: 12px; padding: 12px;
            }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.load_btn.setMinimumHeight(45)
        layout.addWidget(self.load_btn)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        inner.setLayout(layout)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
    
    def _load_saved_folder(self):
        """Load and scan previously used folder."""
        if self.settings.get('results_folder') and os.path.isdir(self.settings['results_folder']):
            self._scan_folder(self.settings['results_folder'])
    
    def _browse_folder(self):
        start_dir = self.settings.get('results_folder', '')
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if folder:
            self._scan_folder(folder)
            # Save the folder for next time
            self.settings['results_folder'] = folder
            save_settings(self.settings)
    
    def _refresh(self):
        if self.current_folder:
            self._scan_folder(self.current_folder)
    
    def _scan_folder(self, folder_path):
        self.current_folder = folder_path
        self.datasets.clear()
        self.dataset_combo.clear()
        self.csv_data.clear()
        
        folder = Path(folder_path)
        self.folder_label.setText(str(folder))
        
        bkg_sub_folder = folder / "BKG_subtracted"
        seg_folder = folder / "Segmented"
        
        if not bkg_sub_folder.exists() or not seg_folder.exists():
            self.info_label.setText("Required subfolders not found (BKG_subtracted, Segmented)")
            self._disable_controls()
            return
        
        # Find subtracted images recursively (handles subfolders)
        sub_files = list(bkg_sub_folder.glob("**/*BGsubtracted_NoDownsample*.tif"))
        
        for sub_file in sub_files:
            sub_name = sub_file.name
            if '.ome' in sub_name:
                base_name = sub_name.split('.ome')[0] + '.ome'
                display_name = sub_name.split('.ome')[0]
                
                # Get relative path from BKG_subtracted folder to preserve subfolder structure
                rel_path = sub_file.parent.relative_to(bkg_sub_folder)
                
                # Search in corresponding subfolder within Segmented
                seg_subfolder = seg_folder / rel_path
                accepted_files = list(seg_subfolder.glob(f"{base_name}_objects_accepted_*.tif*")) if seg_subfolder.exists() else []
                rejected_files = list(seg_subfolder.glob(f"{base_name}_objects_rejected_*.tif*")) if seg_subfolder.exists() else []
                seg_files = list(seg_subfolder.glob(f"{base_name}percentile*.tif")) if seg_subfolder.exists() else []
                
                # Search for CSVs in csv_outputs/per_image_csv subfolder (new layout)
                csv_subfolder = folder / 'csv_outputs' / 'per_image_csv' / rel_path
                # Fallback to old location (directly under output folder)
                if not csv_subfolder.exists():
                    csv_subfolder = folder / rel_path
                snr_csv = list(csv_subfolder.glob(f"{base_name}*full_SNRlabeled_*.csv")) if csv_subfolder.exists() else []
                events_csv = list(csv_subfolder.glob(f"{base_name}*evetsOnly_withClusterIds_*.csv")) if csv_subfolder.exists() else []
                
                # Find raw input image (look in input folder with matching subfolder structure)
                # Try the saved input folder first, then fall back to sibling of output folder
                input_root = self.settings.get('input_folder', '')
                raw_files = []
                for candidate_root in [input_root, str(folder.parent)]:
                    if not candidate_root:
                        continue
                    raw_candidate = Path(candidate_root) / rel_path / f"{base_name}.tif"
                    if raw_candidate.exists():
                        raw_files = [raw_candidate]
                        break
                
                if accepted_files or rejected_files:
                    # Use full relative path in display name to distinguish subfolders
                    full_display_name = str(rel_path / display_name) if str(rel_path) != '.' else display_name
                    self.datasets[full_display_name] = {
                        'raw_image': str(raw_files[0]) if raw_files else None,
                        'subtracted_image': str(sub_file),
                        'segmented_image': str(seg_files[0]) if seg_files else None,
                        'accepted_label': str(accepted_files[0]) if accepted_files else None,
                        'rejected_label': str(rejected_files[0]) if rejected_files else None,
                        'snr_csv': str(snr_csv[0]) if snr_csv else None,
                        'events_csv': str(events_csv[0]) if events_csv else None,
                    }
                    self.dataset_combo.addItem(full_display_name)
        
        if self.datasets:
            self.dataset_combo.setEnabled(True)
            self.label_type_combo.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.info_label.setText(f"Found {len(self.datasets)} dataset(s)")
        else:
            self.info_label.setText("No datasets found")
            self._disable_controls()
    
    def _disable_controls(self):
        self.dataset_combo.setEnabled(False)
        self.label_type_combo.setEnabled(False)
        self.load_btn.setEnabled(False)
    
    def _on_dataset_selected(self, name):
        if name and name in self.datasets:
            ds = self.datasets[name]
            parts = []
            parts.append("Has SNR CSV" if ds['snr_csv'] else "No SNR CSV")
            parts.append("Has Events CSV" if ds['events_csv'] else "No Events CSV")
            self.info_label.setText(" | ".join(parts))
    
    def _load_dataset(self):
        selected = self.dataset_combo.currentText()
        if not selected or selected not in self.datasets:
            return
        
        ds = self.datasets[selected]
        label_type = self.label_type_combo.currentData()
        label_file = ds['accepted_label'] if label_type == 'accepted' else ds['rejected_label']
        
        if not label_file:
            QMessageBox.warning(self, "Not Found", f"No {label_type} labels found")
            return
        
        self.status_label.setText("Loading...")
        self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
        try:
            # Load subtracted image as main image
            subtracted_image = tifffile.imread(ds['subtracted_image'])
            label_image = tifffile.imread(label_file)
            
            # Load raw input image if available
            raw_image = None
            if ds.get('raw_image') and os.path.exists(ds['raw_image']):
                raw_image = tifffile.imread(ds['raw_image'])
            
            segmented_image = None
            if ds.get('segmented_image') and os.path.exists(ds['segmented_image']):
                segmented_image = tifffile.imread(ds['segmented_image'])
            
            csv_data = {}
            if ds['snr_csv'] and os.path.exists(ds['snr_csv']):
                csv_data['snr'] = pd.read_csv(ds['snr_csv'])
                csv_data['snr'].columns = csv_data['snr'].columns.str.lower()
            
            if ds['events_csv'] and os.path.exists(ds['events_csv']):
                csv_data['events'] = pd.read_csv(ds['events_csv'])
                csv_data['events'].columns = csv_data['events'].columns.str.lower()
            
            self.csv_data[selected] = csv_data
            
            n_rois = len(np.unique(label_image)) - 1
            label_type_display = "Accepted" if label_type == "accepted" else "Rejected"
            
            self.viewer_instance.load_new_dataset(
                image_stack=subtracted_image,
                label_image=label_image,
                dataset_name=f"{selected} ({label_type_display})",
                csv_data=csv_data,
                segmented_image=segmented_image,
                raw_image=raw_image
            )
            
            self.status_label.setText(f"Loaded: {n_rois} ROIs")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:40]}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            QMessageBox.critical(self, "Error", str(e))


# =============================================================================
# ROI Plotter Widget
# =============================================================================

class ROIPlotterWidget(QWidget):
    """Widget for displaying ROI time-series plots."""
    
    def __init__(self, napari_viewer=None, viewer_instance=None):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.viewer_instance = viewer_instance
        self.current_time_point = 0
        self.csv_data = None
        self.all_traces = {}
        self.all_firing_data = {}
        self.all_events_data = {}
        self.channel_names = ['mean_intensity']
        self._last_hover_frame = -1
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Info
        self.info_label = QLabel("Shift+Click on ROI to select")
        self.info_label.setStyleSheet(f"font-size: {Config.FONT_SIZE}px; font-weight: bold;")
        layout.addWidget(self.info_label)

        # ROI search box
        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)
        search_label = QLabel("Go to ROI:")
        search_label.setStyleSheet("font-size: 10px;")
        search_layout.addWidget(search_label)
        self.roi_search_edit = QLineEdit()
        self.roi_search_edit.setPlaceholderText("Enter ROI ID...")
        self.roi_search_edit.setMaximumWidth(100)
        self.roi_search_edit.returnPressed.connect(self._on_search_roi)
        search_layout.addWidget(self.roi_search_edit)
        self.roi_search_btn = QPushButton("Select")
        self.roi_search_btn.setMaximumWidth(55)
        self.roi_search_btn.clicked.connect(self._on_search_roi)
        search_layout.addWidget(self.roi_search_btn)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # Controls
        control_layout = QHBoxLayout()
        self.keep_traces_checkbox = QCheckBox("Keep previous")
        control_layout.addWidget(self.keep_traces_checkbox)
        control_layout.addStretch()
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0")
        self.frame_label.setStyleSheet("color: #c00; font-weight: bold;")
        control_layout.addWidget(self.frame_label)
        
        layout.addLayout(control_layout)
        
        # Help text
        help_label = QLabel("Hover over plot to navigate frames")
        help_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        layout.addWidget(help_label)
        
        # Firing detection
        firing_group = QGroupBox("Firing Detection")
        firing_layout = QFormLayout()
        
        self.enable_firing_checkbox = QCheckBox("Enable")
        self.enable_firing_checkbox.toggled.connect(self._on_firing_toggle)
        firing_layout.addRow("", self.enable_firing_checkbox)
        
        self.sd_multiplier_spin = QDoubleSpinBox()
        self.sd_multiplier_spin.setRange(1.0, 10.0)
        self.sd_multiplier_spin.setValue(4.0)
        self.sd_multiplier_spin.valueChanged.connect(self._on_firing_params_changed)
        firing_layout.addRow("SD Multiplier:", self.sd_multiplier_spin)
        
        self.rolling_window_spin = QSpinBox()
        self.rolling_window_spin.setRange(3, 50)
        self.rolling_window_spin.setValue(10)
        self.rolling_window_spin.valueChanged.connect(self._on_firing_params_changed)
        firing_layout.addRow("Rolling Window:", self.rolling_window_spin)

        self.vr_bl_start_spin = QSpinBox()
        self.vr_bl_start_spin.setRange(0, 10000)
        self.vr_bl_start_spin.setValue(0)
        self.vr_bl_start_spin.setToolTip("Baseline start frame (inclusive)")
        self.vr_bl_start_spin.valueChanged.connect(self._on_firing_params_changed)
        firing_layout.addRow("Baseline Start:", self.vr_bl_start_spin)

        self.vr_bl_end_spin = QSpinBox()
        self.vr_bl_end_spin.setRange(0, 10000)
        self.vr_bl_end_spin.setValue(49)
        self.vr_bl_end_spin.setToolTip("Baseline end frame (inclusive)")
        self.vr_bl_end_spin.valueChanged.connect(self._on_firing_params_changed)
        firing_layout.addRow("Baseline End:", self.vr_bl_end_spin)

        firing_group.setLayout(firing_layout)
        firing_group.setMaximumHeight(160)
        layout.addWidget(firing_group)

        # Outlier Explorer button (opens interactive histogram dialog)
        self.outlier_explorer_btn = QPushButton("Open Outlier Explorer")
        self.outlier_explorer_btn.setToolTip(
            "Interactive histograms — drag range sliders to flag outlier ROIs")
        self.outlier_explorer_btn.setMaximumHeight(28)
        self.outlier_explorer_btn.setStyleSheet(
            "font-size: 11px; font-weight: bold; padding: 3px 8px;")
        self.outlier_explorer_btn.clicked.connect(self._on_open_outlier_explorer)
        layout.addWidget(self.outlier_explorer_btn)

        # Plot
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.web_view, stretch=1)
        
        # Connect title change for hover navigation (JS sets document.title)
        self.web_view.titleChanged.connect(self._on_title_changed)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_plot)
        btn_layout.addWidget(clear_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_data)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        self.setMinimumWidth(400)
    
    def _on_open_outlier_explorer(self):
        """Launch the interactive Outlier Explorer histogram dialog."""
        if self.viewer_instance:
            self.viewer_instance._open_outlier_explorer()

    def set_csv_data(self, csv_data):
        self.csv_data = csv_data

    def set_roi_ids(self, roi_ids):
        """Store available ROI IDs (for search validation)."""
        self._available_roi_ids = set(int(i) for i in roi_ids)

    def _on_search_roi(self):
        """Handle ROI search: select the ROI and zoom to it."""
        text = self.roi_search_edit.text().strip()
        if not text:
            return
        try:
            roi_id = int(text)
        except ValueError:
            self.info_label.setText(f"Invalid ID: {text}")
            return

        if hasattr(self, '_available_roi_ids') and roi_id not in self._available_roi_ids:
            self.info_label.setText(f"ROI {roi_id} not found")
            return

        # Delegate to the main viewer's selection handler
        vi = self.viewer_instance
        if vi is not None:
            vi._on_roi_selected(roi_id)
            # Center the napari camera on this ROI
            if vi.label_image is not None:
                mask = vi.label_image == roi_id
                if np.any(mask):
                    rows, cols = np.where(mask)
                    cy, cx = rows.mean(), cols.mean()
                    vi.viewer.camera.center = (0, cy, cx)
                    vi.viewer.camera.zoom = max(
                        vi.viewer.camera.zoom, 4.0)
    
    def _on_firing_toggle(self, checked):
        if self.all_traces:
            self._replot_all()
    
    def _on_firing_params_changed(self):
        if self.enable_firing_checkbox.isChecked() and self.all_traces:
            self._replot_all()
    
    def _replot_all(self):
        if not self.all_traces:
            return
        
        last_roi_id = max(self.all_traces.keys())
        
        self.all_firing_data.clear()
        for roi_id, traces in self.all_traces.items():
            if 'mean_intensity' in traces:
                signal = traces['mean_intensity']
                frames = np.arange(len(signal))
                firing_data = detect_synapse_firings(
                    signal=signal, frames=frames,
                    sd_multiplier=self.sd_multiplier_spin.value(),
                    rolling_window=self.rolling_window_spin.value(),
                    baseline_start_frame=self.vr_bl_start_spin.value(),
                    baseline_end_frame=self.vr_bl_end_spin.value(),
                )
                self.all_firing_data[roi_id] = firing_data
        
        self._do_plot(last_roi_id)
    
    def update_plot_from_csv(self, roi_id):
        if not self.csv_data:
            return
        
        time_traces = {}
        events_data = None
        
        if 'snr' in self.csv_data:
            df = self.csv_data['snr']
            roi_data = df[df['label'] == roi_id].sort_values('slice')
            if len(roi_data) > 0:
                # Use subtracted intensity if available, otherwise fall back to mean_intensity
                if 'mean_intensity_subtracted' in roi_data.columns:
                    time_traces['mean_intensity'] = roi_data['mean_intensity_subtracted'].values
                else:
                    time_traces['mean_intensity'] = roi_data['mean_intensity'].values
        
        if 'events' in self.csv_data:
            df_events = self.csv_data['events']
            events_data = df_events[df_events['cluster_id'] == roi_id].sort_values('slice')
        
        if time_traces:
            self.update_plot(roi_id, time_traces, ['mean_intensity'], events_data)
    
    def update_plot(self, roi_id, time_traces, channel_names=None, events_data=None):
        if channel_names is None:
            channel_names = list(time_traces.keys())
        self.channel_names = channel_names
        
        if not self.keep_traces_checkbox.isChecked():
            self.all_traces.clear()
            self.all_firing_data.clear()
            self.all_events_data.clear()
        
        self.all_traces[roi_id] = time_traces
        
        if events_data is not None and len(events_data) > 0:
            self.all_events_data[roi_id] = events_data
        
        if self.enable_firing_checkbox.isChecked() and 'mean_intensity' in time_traces:
            signal = time_traces['mean_intensity']
            frames = np.arange(len(signal))
            firing_data = detect_synapse_firings(
                signal=signal, frames=frames,
                sd_multiplier=self.sd_multiplier_spin.value(),
                rolling_window=self.rolling_window_spin.value(),
                baseline_start_frame=self.vr_bl_start_spin.value(),
                baseline_end_frame=self.vr_bl_end_spin.value(),
            )
            self.all_firing_data[roi_id] = firing_data
        
        self._do_plot(roi_id)
    
    def _do_plot(self, roi_id):
        fig = go.Figure()
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        
        for idx, (stored_id, traces) in enumerate(sorted(self.all_traces.items())):
            color = colors[idx % len(colors)]
            
            if 'mean_intensity' in traces:
                trace_data = traces['mean_intensity']
                name = f"ROI {stored_id}" if len(self.all_traces) > 1 else "Intensity"
                fig.add_trace(go.Scatter(y=trace_data, mode='lines', name=name,
                                        line=dict(color=color, width=2)))
            
            if stored_id in self.all_events_data:
                events = self.all_events_data[stored_id]
                if len(events) > 0 and 'slice' in events.columns:
                    # Use mean_intensity_subtracted for events if available
                    y_col = 'mean_intensity_subtracted' if 'mean_intensity_subtracted' in events.columns else 'mean_intensity'
                    fig.add_trace(go.Scatter(
                        x=events['slice'].values,
                        y=events[y_col].values if y_col in events.columns else None,
                        mode='markers', name=f"Events ({len(events)})",
                        marker=dict(color='orange', size=8, symbol='diamond')
                    ))
            
            if self.enable_firing_checkbox.isChecked() and stored_id in self.all_firing_data:
                fd = self.all_firing_data[stored_id]
                fig.add_trace(go.Scatter(y=fd['threshold'], mode='lines',
                                        name=f"Threshold", line=dict(color='red', width=1, dash='dot')))
                if len(fd['accepted_frames']) > 0:
                    fig.add_trace(go.Scatter(x=fd['accepted_frames'], y=fd['accepted_values'],
                                            mode='markers', name=f"Firings",
                                            marker=dict(color='#2ca02c', size=10, symbol='star')))
        
        # Add vertical line as a shape (easier to update via JavaScript)
        title = f'ROI {roi_id}' if len(self.all_traces) == 1 else f'ROIs: {", ".join(map(str, sorted(self.all_traces.keys())))}'
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis_title='Frame Number',
            yaxis_title='Intensity (Subtracted)',
            hovermode='x unified', template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50),
            shapes=[
                dict(
                    type='line',
                    x0=self.current_time_point, x1=self.current_time_point,
                    y0=0, y1=1,
                    xref='x', yref='paper',
                    line=dict(color='rgba(255,0,0,0.7)', width=2, dash='dash')
                )
            ]
        )
        
        html_content = fig.to_html(config={'displayModeBar': True, 'responsive': True}, include_plotlyjs=True)
        
        # Inject hover handler - uses document.title to communicate with Python
        hover_script = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
                if (plotDiv && plotDiv.on) {
                    var lastFrame = -1;
                    
                    plotDiv.on('plotly_hover', function(data) {
                        if (data.points && data.points.length > 0) {
                            var frame = Math.round(data.points[0].x);
                            if (frame !== lastFrame) {
                                lastFrame = frame;
                                document.title = 'FRAME:' + frame;
                            }
                        }
                    });
                    
                    plotDiv.on('plotly_click', function(data) {
                        if (data.points && data.points.length > 0) {
                            var frame = Math.round(data.points[0].x);
                            lastFrame = frame;
                            document.title = 'FRAME:' + frame;
                        }
                    });
                }
            }, 100);
        });
        </script>
        """
        
        # Insert script before </body>
        html = html_content.replace('</body>', hover_script + '</body>')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            self.temp_file = f.name
        
        self.web_view.load(QUrl.fromLocalFile(self.temp_file))
        self.info_label.setText(f"Showing ROI {roi_id} (hover on plot to navigate)")
        self.export_btn.setEnabled(True)
    
    def clear_plot(self):
        self.all_traces.clear()
        self.all_firing_data.clear()
        self.all_events_data.clear()
        self.web_view.setHtml("<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;'><h2>No ROI Selected</h2></body></html>")
        self.info_label.setText("Shift+Click on ROI")
        self.export_btn.setEnabled(False)
    
    def update_vline(self, frame):
        """Update the vertical line position using JavaScript."""
        self.current_time_point = frame
        
        # Don't update if this frame came from plotly hover (avoid feedback loop)
        if frame == self._last_hover_frame:
            return
        
        if self.all_traces:
            # Use JavaScript to update the vertical line - minimal code for speed
            self.web_view.page().runJavaScript(
                f"Plotly.relayout(document.querySelector('.plotly-graph-div'),{{'shapes[0].x0':{frame},'shapes[0].x1':{frame}}})"
            )
    
    def _on_title_changed(self, title):
        """Handle hover events from JavaScript via document.title."""
        if title.startswith('FRAME:'):
            try:
                frame = int(title.split(':')[1])
                self._last_hover_frame = frame
                self.frame_label.setText(f"Frame: {frame}")
                
                # Update napari viewer to this frame
                if self.viewer_instance and hasattr(self.viewer_instance, 'viewer'):
                    current_step = list(self.viewer_instance.viewer.dims.current_step)
                    if len(current_step) > 0 and current_step[0] != frame:
                        current_step[0] = frame
                        self.viewer_instance.viewer.dims.current_step = tuple(current_step)
            except (ValueError, IndexError):
                pass
    
    def _export_data(self):
        if not self.all_traces:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Export", "traces.csv", "CSV (*.csv)")
        if file_path:
            rows = []
            for roi_id, traces in sorted(self.all_traces.items()):
                if 'mean_intensity' in traces:
                    for t, val in enumerate(traces['mean_intensity']):
                        rows.append({'roi_id': roi_id, 'frame': t, 'intensity': val})
            pd.DataFrame(rows).to_csv(file_path, index=False)
            self.info_label.setText(f"Exported to {os.path.basename(file_path)}")


# =============================================================================
# Analyze Results Widget
# =============================================================================

class AnalyzeResultsWidget(QWidget):
    """Widget for combining CSVs, running firing detection, and generating
    summary statistics and publication-ready plots."""

    def __init__(self, viewer_instance):
        super().__init__()
        self.viewer_instance = viewer_instance
        self.output_folder = None
        self.csv_files = []
        self.settings = load_settings()
        self._setup_ui()
        self._load_saved_folder()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _setup_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        title = QLabel("Analyze Results")
        title.setStyleSheet(f"font-size: {Config.FONT_SIZE_TITLE}px; font-weight: bold;")
        layout.addWidget(title)

        # ── 1. Output folder ──────────────────────────────────────────
        folder_group = QGroupBox("1. Output Folder")
        folder_layout = QVBoxLayout()
        folder_layout.setContentsMargins(5, 5, 5, 5)
        folder_layout.setSpacing(3)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #666; font-size: 10px;")
        folder_layout.addWidget(self.folder_label)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        btn_row.addWidget(browse_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(refresh_btn)
        folder_layout.addLayout(btn_row)

        self.csv_count_label = QLabel("")
        self.csv_count_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        folder_layout.addWidget(self.csv_count_label)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # ── 2. Config File ────────────────────────────────────────────
        cfg_group = QGroupBox("2. Config File")
        cfg_layout = QVBoxLayout()
        cfg_layout.setContentsMargins(5, 5, 5, 5)
        cfg_layout.setSpacing(4)

        self.analysis_gui_override_checkbox = QCheckBox(
            "Override config file with GUI values")
        self.analysis_gui_override_checkbox.setToolTip(
            "When checked, the parameter values set below in the GUI\n"
            "will take precedence over any iglusnfr_config.json files\n"
            "found in the output folder hierarchy.")
        self.analysis_gui_override_checkbox.setChecked(True)
        cfg_layout.addWidget(self.analysis_gui_override_checkbox)

        cfg_btn_row = QHBoxLayout()
        save_cfg_btn = QPushButton("Save Config...")
        save_cfg_btn.clicked.connect(self._save_config)
        cfg_btn_row.addWidget(save_cfg_btn)
        load_cfg_btn = QPushButton("Load Config...")
        load_cfg_btn.clicked.connect(self._load_config)
        cfg_btn_row.addWidget(load_cfg_btn)
        cfg_layout.addLayout(cfg_btn_row)

        gen_cfg_btn = QPushButton("Generate Template Config")
        gen_cfg_btn.setToolTip(
            f"Write a {CONFIG_FILENAME} template with all default\n"
            "values to a folder of your choice.")
        gen_cfg_btn.clicked.connect(self._generate_template_config)
        cfg_layout.addWidget(gen_cfg_btn)

        # Warning label for config consistency
        self.config_warning_label = QLabel("")
        self.config_warning_label.setWordWrap(True)
        self.config_warning_label.setStyleSheet(
            "color: #FF6600; font-size: 10px; font-style: italic;")
        self.config_warning_label.setVisible(False)
        cfg_layout.addWidget(self.config_warning_label)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        # ── 3. Detected Metadata ──────────────────────────────────────
        preview_group = QGroupBox("3. Detected Metadata")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(5, 5, 5, 5)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        self.preview_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # ── 4. Groups ─────────────────────────────────────────────────
        groups_group = QGroupBox("4. Groups (comma-separated, case-insensitive)")
        groups_layout = QVBoxLayout()
        groups_layout.setContentsMargins(5, 5, 5, 5)
        self.groups_edit = QLineEdit("WT, APOE")
        self.groups_edit.setToolTip(
            "Comma-separated list of group names to look for in filenames.\n"
            "Case insensitive. Example: WT, APOE, KO, HET")
        groups_layout.addWidget(self.groups_edit)
        groups_group.setLayout(groups_layout)
        layout.addWidget(groups_group)

        # ── 5. Acquisition & Baseline ─────────────────────────────────
        acq_group = QGroupBox("5. Acquisition & Baseline")
        acq_layout = QFormLayout()
        acq_layout.setContentsMargins(5, 5, 5, 5)
        acq_layout.setSpacing(4)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000.0)
        self.frame_rate_spin.setValue(100.0)
        self.frame_rate_spin.setSingleStep(1.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setToolTip("Frame rate at which images were acquired (Hz)")
        acq_layout.addRow("Frame Rate (Hz):", self.frame_rate_spin)

        self.bl_start_spin = QSpinBox()
        self.bl_start_spin.setRange(0, 10000)
        self.bl_start_spin.setValue(0)
        self.bl_start_spin.setToolTip(
            "First frame of the baseline window (inclusive).\n"
            "Default 0 = beginning of trace.")
        acq_layout.addRow("Baseline Start Frame:", self.bl_start_spin)

        self.bl_end_spin = QSpinBox()
        self.bl_end_spin.setRange(0, 10000)
        self.bl_end_spin.setValue(49)
        self.bl_end_spin.setToolTip(
            "Last frame of the baseline window (inclusive).\n"
            "Default 49 = pre-stimulus period for standard\n"
            "evoked protocols (stimulation starts at frame 50).")
        acq_layout.addRow("Baseline End Frame:", self.bl_end_spin)

        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        # ── 6. Firing Detection ───────────────────────────────────────
        fire_group = QGroupBox("6. Firing Detection")
        fire_layout = QFormLayout()
        fire_layout.setContentsMargins(5, 5, 5, 5)
        fire_layout.setSpacing(4)

        self.sd_mult_spin = QDoubleSpinBox()
        self.sd_mult_spin.setRange(1.0, 20.0)
        self.sd_mult_spin.setValue(4.0)
        self.sd_mult_spin.setSingleStep(0.5)
        self.sd_mult_spin.setToolTip(
            "Local maximum must exceed:\n"
            "rolling_mean + SD_multiplier * baseline_SD")
        fire_layout.addRow("SD Multiplier:", self.sd_mult_spin)

        self.roll_win_spin = QSpinBox()
        self.roll_win_spin.setRange(3, 100)
        self.roll_win_spin.setValue(10)
        self.roll_win_spin.setToolTip("Window size for centred rolling mean")
        fire_layout.addRow("Rolling Window:", self.roll_win_spin)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(1)
        self.order_spin.setToolTip(
            "Points on each side for local-maximum comparison\n"
            "(scipy argrelextrema order)")
        fire_layout.addRow("Local Max Order:", self.order_spin)

        fire_group.setLayout(fire_layout)
        layout.addWidget(fire_group)

        # ── 7. Evoked Experiment ──────────────────────────────────────
        evok_group = QGroupBox("7. Evoked Experiment")
        evok_layout = QFormLayout()
        evok_layout.setContentsMargins(5, 5, 5, 5)
        evok_layout.setSpacing(4)

        self.stim_start_spin = QSpinBox()
        self.stim_start_spin.setRange(0, 10000)
        self.stim_start_spin.setValue(50)
        self.stim_start_spin.setToolTip(
            "Frame number of the first stimulus (0-indexed).\n"
            "Typically 50 (49 baseline frames, first AP at frame 50).")
        evok_layout.addRow("Stim Start Frame:", self.stim_start_spin)

        self.resp_win_spin = QSpinBox()
        self.resp_win_spin.setRange(1, 50)
        self.resp_win_spin.setValue(5)
        self.resp_win_spin.setToolTip(
            "Number of frames after each stimulus to look for a response")
        evok_layout.addRow("Response Window:", self.resp_win_spin)

        evok_group.setLayout(evok_layout)
        layout.addWidget(evok_group)

        # ── 8. Paired-Pulse (PPR) ────────────────────────────────────
        ppr_group = QGroupBox("8. Paired-Pulse (PPR)")
        ppr_layout = QFormLayout()
        ppr_layout.setContentsMargins(5, 5, 5, 5)
        ppr_layout.setSpacing(4)

        self.ppr_pulse1_spin = QSpinBox()
        self.ppr_pulse1_spin.setRange(0, 10000)
        self.ppr_pulse1_spin.setValue(50)
        self.ppr_pulse1_spin.setToolTip(
            "Frame of the 1st pulse in PPR experiments.\n"
            "Default 50.")
        ppr_layout.addRow("1st Pulse Frame:", self.ppr_pulse1_spin)

        self.ppr_pulse2_spin = QSpinBox()
        self.ppr_pulse2_spin.setRange(0, 10000)
        self.ppr_pulse2_spin.setValue(60)
        self.ppr_pulse2_spin.setToolTip(
            "Frame of the 2nd pulse in PPR experiments.\n"
            "Default 60 (20 Hz paired-pulse).")
        ppr_layout.addRow("2nd Pulse Frame:", self.ppr_pulse2_spin)

        self.ppr_resp_win_spin = QSpinBox()
        self.ppr_resp_win_spin.setRange(1, 50)
        self.ppr_resp_win_spin.setValue(4)
        self.ppr_resp_win_spin.setToolTip(
            "Number of frames AFTER each pulse to search for a\n"
            "response peak (e.g. 4 → frames pulse+1 to pulse+4).\n"
            "Peaks outside this window are classified as asynchronous.")
        ppr_layout.addRow("PPR Response Window:", self.ppr_resp_win_spin)

        ppr_group.setLayout(ppr_layout)
        layout.addWidget(ppr_group)

        # ── 9. Run ────────────────────────────────────────────────────
        self.run_btn = QPushButton("RUN FULL ANALYSIS")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; color: white; font-weight: bold;
                font-size: 12px; padding: 12px;
            }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.run_btn.setMinimumHeight(45)
        layout.addWidget(self.run_btn)

        # ── 10. Status / log ─────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.log_text)

        inner.setLayout(layout)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _get_groups(self):
        """Parse the groups text field into a list."""
        raw = self.groups_edit.text()
        return [g.strip() for g in raw.split(",") if g.strip()]

    def _get_params(self):
        """Collect all analysis parameters into a dict."""
        return {
            "sd_multiplier":        self.sd_mult_spin.value(),
            "rolling_window":       self.roll_win_spin.value(),
            "baseline_start_frame": self.bl_start_spin.value(),
            "baseline_end_frame":   self.bl_end_spin.value(),
            "order":                self.order_spin.value(),
            "frame_rate":           self.frame_rate_spin.value(),
            "stim_start_frame":     self.stim_start_spin.value(),
            "response_window":      self.resp_win_spin.value(),
            "ppr_pulse1_frame":     self.ppr_pulse1_spin.value(),
            "ppr_pulse2_frame":     self.ppr_pulse2_spin.value(),
            "ppr_response_window":  self.ppr_resp_win_spin.value(),
            "groups":               self._get_groups(),
        }

    def _set_params(self, params):
        """Load parameters from a dict into the UI widgets."""
        if "sd_multiplier" in params:
            self.sd_mult_spin.setValue(float(params["sd_multiplier"]))
        if "rolling_window" in params:
            self.roll_win_spin.setValue(int(params["rolling_window"]))
        if "baseline_start_frame" in params:
            self.bl_start_spin.setValue(int(params["baseline_start_frame"]))
        if "baseline_end_frame" in params:
            self.bl_end_spin.setValue(int(params["baseline_end_frame"]))
        if "order" in params:
            self.order_spin.setValue(int(params["order"]))
        if "frame_rate" in params:
            self.frame_rate_spin.setValue(float(params["frame_rate"]))
        if "stim_start_frame" in params:
            self.stim_start_spin.setValue(int(params["stim_start_frame"]))
        if "response_window" in params:
            self.resp_win_spin.setValue(int(params["response_window"]))
        if "ppr_pulse1_frame" in params:
            self.ppr_pulse1_spin.setValue(int(params["ppr_pulse1_frame"]))
        if "ppr_pulse2_frame" in params:
            self.ppr_pulse2_spin.setValue(int(params["ppr_pulse2_frame"]))
        if "ppr_response_window" in params:
            self.ppr_resp_win_spin.setValue(int(params["ppr_response_window"]))
        if "groups" in params:
            grps = params["groups"]
            if isinstance(grps, list):
                self.groups_edit.setText(", ".join(grps))
            else:
                self.groups_edit.setText(str(grps))

    # -----------------------------------------------------------------
    # Config file save / load
    # -----------------------------------------------------------------
    def _save_config(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Config", "", "JSON Files (*.json)")
        if not path:
            return
        import json as _json
        params = self._get_params()
        # Make JSON-serialisable
        for k, v in params.items():
            if isinstance(v, (np.integer,)):
                params[k] = int(v)
            elif isinstance(v, (np.floating,)):
                params[k] = float(v)
        with open(path, "w") as f:
            _json.dump(params, f, indent=2)
        self._log(f"Config saved to {path}")

    def _load_config(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Config", "", "JSON Files (*.json)")
        if not path:
            return
        import json as _json
        try:
            with open(path) as f:
                params = _json.load(f)
            self._set_params(params)
            self._log(f"Config loaded from {path}")
        except Exception as e:
            self._log(f"ERROR loading config: {e}")

    def _generate_template_config(self):
        """Write a template iglusnfr_config.json to a user-chosen folder."""
        start_dir = self.output_folder or ''
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder for template config", start_dir)
        if not folder:
            return
        dest = os.path.join(folder, CONFIG_FILENAME)
        if os.path.isfile(dest):
            reply = QMessageBox.question(
                self, "File exists",
                f"{CONFIG_FILENAME} already exists in that folder.\n"
                "Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        try:
            generate_template_config(dest)
            self._log(f"Template config written to: {dest}")
        except Exception as e:
            self._log(f"ERROR generating template: {e}")

    # -----------------------------------------------------------------
    # Folder handling
    # -----------------------------------------------------------------
    def _load_saved_folder(self):
        folder = self.settings.get('results_folder', '')
        if folder and os.path.isdir(folder):
            self._scan_folder(folder)

    def _browse_folder(self):
        start_dir = self.settings.get('results_folder', '')
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir)
        if folder:
            self._scan_folder(folder)
            self.settings['results_folder'] = folder
            save_settings(self.settings)

    def _refresh(self):
        if self.output_folder:
            self._scan_folder(self.output_folder)

    def _scan_folder(self, folder_path):
        from analyze_results import (find_snr_labeled_csvs,
                                      extract_metadata_from_filename)
        import re as _re

        self.output_folder = folder_path
        self.folder_label.setText(folder_path)

        self.csv_files = find_snr_labeled_csvs(folder_path)
        n = len(self.csv_files)
        self.csv_count_label.setText(
            f"Found {n} _full_SNRlabeled_ CSV file(s)" if n
            else "No matching CSV files found")
        self.run_btn.setEnabled(n > 0)

        # Auto-load saved config if present
        config_path = os.path.join(
            folder_path, "csv_outputs", "analysis", "csvs",
            "analysis_params.json")
        if os.path.isfile(config_path):
            try:
                import json as _json
                with open(config_path) as _f:
                    saved = _json.load(_f)
                self._set_params(saved)
                self._log(f"Loaded saved config from: {config_path}")
            except Exception:
                pass

        # Check config consistency across subfolders and warn
        try:
            warnings = check_analysis_consistency(folder_path)
            if warnings:
                warn_text = ("WARNING: Analysis parameters differ across "
                             "subfolders. Cross-group comparisons may be "
                             "invalid.\n" + "\n".join(warnings[:5]))
                if len(warnings) > 5:
                    warn_text += f"\n... and {len(warnings) - 5} more"
                self.config_warning_label.setText(warn_text)
                self.config_warning_label.setVisible(True)
                self._log(warn_text)
            else:
                self.config_warning_label.setVisible(False)
        except Exception:
            self.config_warning_label.setVisible(False)

        if n > 0:
            groups = self._get_groups()
            lines = []
            for f in self.csv_files:
                stem = f.stem
                image_name = _re.split(r"_full_SNRlabeled_", stem)[0]
                meta = extract_metadata_from_filename(
                    image_name, custom_groups=groups)
                lines.append(f"{image_name}")
                sub = meta.get('evoked_subtype') or '-'
                lines.append(
                    f"  Group: {meta['group']}  |  "
                    f"Type: {meta['experiment_type']}  |  "
                    f"Subtype: {sub}  |  "
                    f"AP: {meta['action_potentials'] or '-'}  |  "
                    f"Hz: {meta['stimulus_frequency_hz'] or '-'}")
            self.preview_text.setPlainText("\n".join(lines))
        else:
            self.preview_text.setPlainText("")

    # -----------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------
    def _log(self, msg):
        """Append a line to the log text and process events."""
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents()

    def _run_analysis(self):
        if not self.output_folder or not self.csv_files:
            return

        from trace_analysis import run_full_analysis

        params = self._get_params()

        self.status_label.setText("Running ...")
        self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.log_text.clear()
        self.run_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            result = run_full_analysis(
                self.output_folder, params,
                progress_callback=self._log,
            )

            if result is not None:
                roi_df = result["firings_per_roi"]
                grp_df = result["group_summary"]
                n_rois = len(roi_df)
                n_peaks = int(
                    result["combined_with_peaks"]["accepted_peak"].sum()
                )
                analysis_dir = result["analysis_dir"]

                summary_lines = [
                    f"ROIs analysed: {n_rois}",
                    f"Accepted peaks: {n_peaks}",
                    "",
                ]

                # Per-group summary
                if grp_df is not None and len(grp_df) > 0:
                    summary_lines.append("--- Group Summary ---")
                    for _, row in grp_df.iterrows():
                        parts = [f"{row.get('group', '?')}"]
                        if row.get("experiment_type"):
                            parts.append(str(row["experiment_type"]))
                        if pd.notna(row.get("action_potentials")):
                            parts.append(f"{int(row['action_potentials'])}AP")
                        if pd.notna(row.get("stimulus_frequency_hz")):
                            parts.append(f"{int(row['stimulus_frequency_hz'])}Hz")
                        label = " / ".join(parts)
                        rois_val = row.get("total_rois",
                                           row.get("n_rois", 0))
                        line = (
                            f"  {label}: "
                            f"n_images={int(row.get('n_images', 0))}, "
                            f"total_rois={int(rois_val)}, "
                            f"firings={row['mean_firings']:.2f} "
                            f"\u00b1 {row.get('sd_firings', 0):.2f}"
                        )
                        if pd.notna(row.get("mean_response_rate")):
                            line += (
                                f", resp_rate="
                                f"{row['mean_response_rate']:.3f}"
                            )
                        summary_lines.append(line)

                # PPR summary
                ppr_grp = result.get("ppr_group_summary")
                if ppr_grp is not None and len(ppr_grp) > 0:
                    summary_lines.append("")
                    summary_lines.append("--- PPR Summary ---")
                    for _, row in ppr_grp.iterrows():
                        parts = [f"{row.get('group', '?')}"]
                        if pd.notna(row.get("stimulus_frequency_hz")):
                            parts.append(
                                f"{int(row['stimulus_frequency_hz'])}Hz")
                        label = " / ".join(parts)
                        line = (
                            f"  {label}: "
                            f"n_images={int(row.get('n_images', 0))}, "
                            f"PPR={row.get('mean_ppr', 0):.3f} "
                            f"\u00b1 {row.get('sd_ppr', 0):.3f}, "
                            f"peak1={row.get('mean_peak1', 0):.1f}, "
                            f"peak2={row.get('mean_peak2', 0):.1f}"
                        )
                        summary_lines.append(line)

                summary_lines.append(f"\nOutputs: {analysis_dir}")
                self._log("\n".join(summary_lines))

                self.status_label.setText(
                    f"Done \u2014 {n_rois} ROIs, {n_peaks} peaks")
                self.status_label.setStyleSheet(
                    "color: #4CAF50; font-weight: bold;")
            else:
                self.status_label.setText("No data found")
                self.status_label.setStyleSheet(
                    "color: #f44336; font-weight: bold;")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:60]}")
            self.status_label.setStyleSheet(
                "color: #f44336; font-weight: bold;")
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)


# =============================================================================
# Main Viewer
# =============================================================================

class IGluSnFRViewer:
    """Main napari viewer with integrated processing and visualization."""
    
    def __init__(self):
        self.viewer = napari.Viewer()
        
        # Initialize attributes
        self.image_stack = None
        self.label_image = None
        self.segmented_image = None
        self.raw_image = None
        self.csv_data = None
        self.n_timepoints = 0
        self.n_channels = 0
        self.height = 0
        self.width = 0
        self.image_layers = []
        self.segmented_layer = None
        self.raw_layer = None
        self.labels_layer = None
        self.shapes_layer = None
        self.selected_rois = set()
        
        self._setup_ui()
        
        print("\n" + "=" * 60)
        print("iGluSnFR Analysis Toolbox v3")
        print("=" * 60)
        print("Use 'Processing' tab to run analysis")
        print("Use 'View Results' tab to browse datasets")
        print("Use 'Analyze Results' tab to combine CSVs + extract metadata")
        print("Shift+Click on ROIs to view time-series")
        print("=" * 60 + "\n")
    
    def _setup_ui(self):
        # Apply stylesheet
        self.viewer.window._qt_window.setStyleSheet(get_stylesheet())
        
        # Create main control widget with tabs
        self.tab_widget = QTabWidget()
        
        # Processing tab
        self.processing_widget = ProcessingWidget(viewer_instance=self)
        self.tab_widget.addTab(self.processing_widget, "Processing")
        
        # View Results tab
        self.loader_widget = DatasetLoaderWidget(viewer_instance=self)
        self.tab_widget.addTab(self.loader_widget, "View Results")
        
        # Analyze Results tab
        self.analyze_widget = AnalyzeResultsWidget(viewer_instance=self)
        self.tab_widget.addTab(self.analyze_widget, "Analyze Results")
        
        # Add to viewer
        self.viewer.window.add_dock_widget(
            self.tab_widget, area='left', name='iGluSnFR Analysis'
        )
        
        # Plot widget
        self.plot_widget = ROIPlotterWidget(
            napari_viewer=self.viewer, viewer_instance=self
        )
        self.viewer.window.add_dock_widget(
            self.plot_widget, area='right', name='ROI Time Series'
        )

        # --- Compact the layer list and move iGluSnFR panel to the top ---
        # This runs after the Qt event loop has processed the initial layout.
        QTimer.singleShot(200, self._adjust_left_panel_layout)

    def _adjust_left_panel_layout(self):
        """Shrink the built-in layer list / controls and place our panel above.

        Napari stacks three dock widgets on the left: 'layer list',
        'layer controls', and any user-added docks (ours).  We:
        1. Re-order so the iGluSnFR Analysis dock is at the top.
        2. Set size policies so the napari docks stay compact while our
           dock gets the bulk of the vertical space.
        3. Adjust the QSplitter stretch factors to enforce the ratio.
        """
        from qtpy.QtWidgets import QDockWidget, QSplitter

        qt_win = self.viewer.window._qt_window

        iglu_dock = None
        layer_list_dock = None
        layer_ctrl_dock = None

        for dock in qt_win.findChildren(QDockWidget):
            title = (dock.windowTitle() or "").strip().lower()
            if dock.widget() is self.tab_widget:
                iglu_dock = dock
            elif title == "layer list":
                layer_list_dock = dock
            elif title == "layer controls":
                layer_ctrl_dock = dock

        # Place our dock on top: split iGluSnFR above layer controls,
        # then layer controls above layer list.
        try:
            if iglu_dock and layer_ctrl_dock:
                qt_win.splitDockWidget(iglu_dock, layer_ctrl_dock,
                                       Qt.Vertical)
            if layer_ctrl_dock and layer_list_dock:
                qt_win.splitDockWidget(layer_ctrl_dock, layer_list_dock,
                                       Qt.Vertical)
        except Exception as e:
            print(f"  Layout rearrange note: {e}")

        # Tell the napari docks to stay as small as possible (Minimum
        # policy means "use sizeHint as minimum, don't grow beyond it
        # unless forced").  Our dock gets Expanding so it fills the rest.
        for dock in [layer_list_dock, layer_ctrl_dock]:
            if dock is not None:
                dock.setSizePolicy(QSizePolicy.Preferred,
                                   QSizePolicy.Maximum)
                w = dock.widget()
                if w is not None:
                    w.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Maximum)

        if iglu_dock is not None:
            iglu_dock.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Expanding)
            iglu_dock.setMinimumHeight(300)

        # Find the QSplitter that holds these docks on the left side
        # and set stretch factors so our dock gets ~80% of the space.
        def _fix_splitter():
            for splitter in qt_win.findChildren(QSplitter):
                if splitter.orientation() != Qt.Vertical:
                    continue
                # Check if this splitter contains our docks
                indices = {}
                for idx in range(splitter.count()):
                    child = splitter.widget(idx)
                    if child is iglu_dock:
                        indices["iglu"] = idx
                    elif child is layer_list_dock:
                        indices["list"] = idx
                    elif child is layer_ctrl_dock:
                        indices["ctrl"] = idx
                if "iglu" not in indices:
                    continue
                # Found it — set stretch factors
                for idx in range(splitter.count()):
                    if idx == indices.get("iglu"):
                        splitter.setStretchFactor(idx, 10)
                    else:
                        splitter.setStretchFactor(idx, 1)
                # Also set explicit sizes: give our dock ~70% of the
                # total height, split the rest between the napari docks.
                total = splitter.height()
                if total > 0:
                    sizes = []
                    for idx in range(splitter.count()):
                        if idx == indices.get("iglu"):
                            sizes.append(int(total * 0.70))
                        elif idx == indices.get("ctrl"):
                            sizes.append(int(total * 0.12))
                        else:
                            sizes.append(int(total * 0.18))
                    splitter.setSizes(sizes)
                break  # done

        # Run the splitter fix after a short delay so geometry is settled
        QTimer.singleShot(100, _fix_splitter)

    def load_new_dataset(self, image_stack, label_image, dataset_name=None,
                         csv_data=None, segmented_image=None, raw_image=None):
        """Load a new dataset into the viewer."""
        self._clear_layers()
        self.selected_rois.clear()
        self.plot_widget.clear_plot()
        
        self.image_stack = image_stack
        self.label_image = label_image
        self.csv_data = csv_data
        self.segmented_image = segmented_image
        self.raw_image = raw_image
        
        self.plot_widget.set_csv_data(csv_data)
        
        # Handle dimensions
        if image_stack.ndim == 3:
            self.n_timepoints, self.height, self.width = image_stack.shape
            self.n_channels = 1
            self.image_stack = image_stack[:, np.newaxis, :, :]
        elif image_stack.ndim == 4:
            self.n_timepoints, self.n_channels, self.height, self.width = image_stack.shape
        
        # Add subtracted image layers (main display)
        self.image_layers = []
        colormaps = ['green', 'magenta', 'cyan', 'yellow']
        for ch in range(self.n_channels):
            layer = self.viewer.add_image(
                self.image_stack[:, ch, :, :],
                name=f"Subtracted Ch{ch+1}" if self.n_channels > 1 else "Subtracted",
                colormap=colormaps[ch % 4],
                blending='additive'
            )
            self.image_layers.append(layer)
        
        # Add raw input image
        if raw_image is not None:
            self.raw_layer = self.viewer.add_image(
                raw_image, name='Raw Input', colormap='gray',
                blending='additive', visible=False
            )
        
        # Add segmented image
        if segmented_image is not None:
            seg_scaled = (segmented_image > 0).astype(np.uint16) * 65535
            self.segmented_layer = self.viewer.add_image(
                seg_scaled, name='Segmented', colormap='red',
                blending='additive', opacity=0.5, visible=False
            )
        
        # Add labels
        self.labels_layer = self.viewer.add_labels(
            label_image, name='ROIs', opacity=0.5
        )
        self.labels_layer.mode = 'pick'

        # Add ROI ID text labels at centroids
        self._roi_ids = []
        self._roi_centroids = []
        from scipy import ndimage as _ndi
        unique_ids = np.unique(label_image)
        unique_ids = unique_ids[unique_ids > 0]
        if len(unique_ids) > 0:
            centroids = _ndi.center_of_mass(
                label_image > 0, label_image, unique_ids)
            self._roi_ids = list(unique_ids)
            self._roi_centroids = np.array(centroids)  # (N, 2) — y, x
            text_props = {
                "string": [str(int(i)) for i in unique_ids],
                "color": "white",
                "size": 10,
                "anchor": "center",
            }
            self.roi_text_layer = self.viewer.add_points(
                self._roi_centroids,
                name="ROI Labels",
                text=text_props,
                size=1,
                face_color="transparent",
                border_color="transparent",
                visible=True,
            )
        else:
            self.roi_text_layer = None

        # Notify the plot widget of available ROI IDs
        self.plot_widget.set_roi_ids(self._roi_ids)

        # --- Outlier layer (created on demand via Outlier Explorer) ---
        self.outlier_layer = None

        # Add shapes for bounding boxes
        self.shapes_layer = self.viewer.add_shapes(
            name='ROI Boundaries', edge_color='yellow',
            edge_width=2, face_color='transparent'
        )
        
        # Connect click callback
        @self.labels_layer.mouse_drag_callbacks.append
        def on_click(layer, event):
            if event.type == 'mouse_press' and 'Shift' in event.modifiers:
                coords = layer.world_to_data(event.position)
                coords = np.round(coords).astype(int)
                
                if len(coords) >= 2:
                    y, x = coords[-2], coords[-1]
                    if 0 <= y < self.height and 0 <= x < self.width:
                        roi_id = self.label_image[y, x]
                        if roi_id > 0:
                            self._on_roi_selected(roi_id)
        
        # Track time - update plotly vertical line when napari frame changes
        self._vline_update_timer = QTimer()
        self._vline_update_timer.setSingleShot(True)
        self._vline_update_timer.setInterval(16)  # ~60fps
        self._pending_frame = 0
        
        def do_vline_update():
            self.plot_widget.update_vline(self._pending_frame)
        
        self._vline_update_timer.timeout.connect(do_vline_update)
        
        @self.viewer.dims.events.current_step.connect
        def on_time_change(event):
            step = self.viewer.dims.current_step
            if len(step) > 0:
                self._pending_frame = step[0]
                self.plot_widget.frame_label.setText(f"Frame: {self._pending_frame}")
                self._vline_update_timer.start()
        
        self.viewer.layers.selection.active = self.labels_layer
        
        print(f"Loaded: {dataset_name}")
        print(f"  Shape: {self.n_timepoints} frames, {self.height}x{self.width}")
        print(f"  ROIs: {len(np.unique(label_image)) - 1}")
    
    # ------------------------------------------------------------------
    # Outlier Explorer helpers
    # ------------------------------------------------------------------
    def _build_roi_metrics_df(self) -> "pd.DataFrame | None":
        """Compute per-ROI metrics for the current dataset.

        Returns a DataFrame with one row per ROI, columns: label,
        baseline_median, baseline_sd, max_signal, total_firings, area,
        solidity, circularity.  Returns *None* when data is missing.
        """
        csv_data = self.csv_data
        if (not csv_data or 'snr' not in csv_data
                or len(self._roi_ids) == 0):
            return None

        try:
            from trace_analysis import detect_firings

            snr_df = csv_data['snr']
            sig_col = ("mean_intensity_subtracted"
                       if "mean_intensity_subtracted" in snr_df.columns
                       else "mean_intensity")

            vr = self.plot_widget
            sd_m = vr.sd_multiplier_spin.value()
            rw = vr.rolling_window_spin.value()
            bl_s = vr.vr_bl_start_spin.value()
            bl_e = vr.vr_bl_end_spin.value()

            roi_rows = []
            for roi_id in self._roi_ids:
                roi_df = snr_df[snr_df["label"] == roi_id].sort_values("slice")
                if len(roi_df) < 3:
                    continue
                signal = roi_df[sig_col].values
                frames = roi_df["slice"].values
                det = detect_firings(signal, frames, sd_m, rw,
                                     baseline_start_frame=bl_s,
                                     baseline_end_frame=bl_e)
                row = {
                    "label": roi_id,
                    "total_firings": len(det["accepted_frames"]),
                    "baseline_median": det["baseline_median"],
                    "baseline_sd": det["baseline_sd"],
                    "max_signal": float(np.nanmax(signal)),
                    "area": (roi_df["area"].iloc[0]
                             if "area" in roi_df.columns else np.nan),
                    "solidity": (roi_df["solidity"].iloc[0]
                                 if "solidity" in roi_df.columns else np.nan),
                    "circularity": (roi_df["circularity"].iloc[0]
                                    if "circularity" in roi_df.columns
                                    else np.nan),
                }
                roi_rows.append(row)

            if not roi_rows:
                return None
            return pd.DataFrame(roi_rows)

        except Exception as e:
            print(f"  Could not compute ROI metrics: {e}")
            return None

    def _open_outlier_explorer(self):
        """Open the interactive Outlier Explorer dialog."""
        from outlier_explorer import OutlierExplorer

        roi_df = self._build_roi_metrics_df()
        if roi_df is None or len(roi_df) == 0:
            print("  No ROI metric data available for Outlier Explorer.")
            return

        # Close previous dialog if still open
        if hasattr(self, '_outlier_explorer') and self._outlier_explorer is not None:
            try:
                self._outlier_explorer.close()
            except Exception:
                pass

        self._outlier_explorer = OutlierExplorer(
            roi_df,
            apply_callback=self._apply_outlier_flags,
            parent=None,
        )
        self._outlier_explorer.show()

    def _apply_outlier_flags(self, flagged_labels: set):
        """Update (or create) the Outlier Score layer from a set of
        flagged ROI labels, supplied by the Outlier Explorer dialog.

        Flagged ROIs are red; others are green.  Layer data is updated
        in-place when possible to avoid segfaults.
        """
        if len(self._roi_ids) == 0 or len(self._roi_centroids) == 0:
            return

        face_colors = []
        pts = []
        for rid, centroid in zip(self._roi_ids, self._roi_centroids):
            if rid in flagged_labels:
                face_colors.append([1.0, 0.0, 0.0, 0.7])   # red
            else:
                face_colors.append([0.0, 0.8, 0.0, 0.5])   # green
            pts.append(centroid)

        pts = np.array(pts)
        face_colors = np.array(face_colors)

        # In-place update when the layer already exists
        if (self.outlier_layer is not None
                and self.outlier_layer in self.viewer.layers):
            self.outlier_layer.data = pts
            self.outlier_layer.face_color = face_colors
            self.outlier_layer.visible = True
        else:
            self.outlier_layer = self.viewer.add_points(
                pts,
                name="Outlier Score",
                face_color=face_colors,
                border_color="transparent",
                size=12,
                symbol="diamond",
                visible=True,
            )

        n_flagged = len(flagged_labels)
        print(f"  Outlier layer: {n_flagged} flagged out of "
              f"{len(self._roi_ids)} ROIs")

    def _clear_layers(self):
        self.viewer.layers.clear()
        self.image_layers = []
        self.labels_layer = None
        self.shapes_layer = None
        self.segmented_layer = None
        self.raw_layer = None
        self.roi_text_layer = None
        self.outlier_layer = None
        self._roi_ids = []
        self._roi_centroids = []
        # Close any open Outlier Explorer dialog
        if hasattr(self, '_outlier_explorer') and self._outlier_explorer is not None:
            try:
                self._outlier_explorer.close()
            except Exception:
                pass
            self._outlier_explorer = None
    
    def _on_roi_selected(self, roi_id):
        if not self.plot_widget.keep_traces_checkbox.isChecked():
            self.selected_rois.clear()
        self.selected_rois.add(roi_id)
        
        print(f"Selected ROI {roi_id}")
        
        if self.csv_data:
            self.plot_widget.update_plot_from_csv(roi_id)
        else:
            # Extract from image
            traces = {}
            mask = self.label_image == roi_id
            trace = [self.image_stack[t, 0, mask].mean() for t in range(self.n_timepoints)]
            traces['mean_intensity'] = np.array(trace)
            self.plot_widget.update_plot(roi_id, traces)
        
        self._draw_bounding_boxes()
    
    def _draw_bounding_boxes(self):
        if self.shapes_layer is None:
            return
        self.shapes_layer.data = []
        
        for roi_id in self.selected_rois:
            mask = self.label_image == roi_id
            if not np.any(mask):
                continue
            rows, cols = np.where(mask)
            rect = np.array([
                [rows.min(), cols.min()],
                [rows.min(), cols.max()],
                [rows.max(), cols.max()],
                [rows.max(), cols.min()],
            ])
            self.shapes_layer.add_rectangles(
                rect, edge_color='yellow', edge_width=2, face_color='transparent'
            )
    
    def run(self):
        # Schedule a theme refresh after startup to fix icon loading issue
        def refresh_theme():
            try:
                from napari.settings import get_settings
                settings = get_settings()
                current_theme = settings.appearance.theme
                temp_theme = 'light' if current_theme == 'dark' else 'dark'
                settings.appearance.theme = temp_theme
                QTimer.singleShot(50, lambda: setattr(settings.appearance, 'theme', current_theme))
            except Exception as e:
                print(f"Theme refresh skipped: {e}")
        
        QTimer.singleShot(500, refresh_theme)
        napari.run()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    viewer = IGluSnFRViewer()
    viewer.run()
