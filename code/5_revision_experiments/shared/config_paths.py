"""
Central path + constant configuration for all JBHI revision Colab experiments.

Derived verbatim from the author's existing notebooks:
  - COLAB_IEEE_FIGURES.ipynb      (paper baselines + EPHNOGRAM chunks + split)
  - cross_dataset_validation.ipynb (SensSmart-Tech + Trail-4 generator)

Every experiment script imports from this file so paths live in ONE place.
If your Drive layout differs, edit ONLY this file.
"""

import os
import torch

# ============================================================================
# >>> EDIT THIS BLOCK IF YOUR DRIVE MOVES <<<
# Updated 2026-07-14: the dataset is now a SHARED folder mounted at
# /content/drive/MyDrive/JBHI-Revision/October Thesis Dataset. If it moves again,
# change ONLY DRIVE_ROOT and OCT_DIR. NOTE: DATA is READ from OCT_DIR; everything
# the code WRITES (retrained checkpoints, outputs) goes under your own Drive
# (OUT_ROOT), so a read-only shared dataset still works.
# ============================================================================
DRIVE_ROOT = "/content/drive/MyDrive"
# Shared dataset root: contains "Refine-Chunked", "Best-epoch-trail-4", etc. (read-only OK).
OCT_DIR    = f"{DRIVE_ROOT}/JBHI-Revision/October Thesis Dataset"

# ----------------------------------------------------------------------------
# EPHNOGRAM data (training-domain test set = 453 segments, 3 subjects)
# .npz keys per chunk: 'ecg', 'pcg', 'subject_id', 'scenario'
#   ecg, pcg : float32 arrays, 64000 samples (8 s @ 8 kHz), RAW (not normalized)
#   subject_id : e.g. 'S009', 'S019', 'S023'
#   scenario   : free-text, e.g. 'rest', 'exercise bicycle stress test'
# ----------------------------------------------------------------------------
EPH_CHUNKS_DIR = f"{OCT_DIR}/Refine-Chunked"
# Subject split. keys: train/val/test_subjects. IF THIS FILE IS ABSENT the
# notebooks fall back to an arbitrary split, which gives the WRONG test set
# (the 2026-07-10 run hit this: 22 segments / 4 subjects instead of the paper's
# 453 segments / 3 subjects). Upload the real split JSON, or point this at it.
SPLIT_FILE     = f"{EPH_CHUNKS_DIR}/train_val_test_split.json"

# ----------------------------------------------------------------------------
# Model checkpoints  --  CONFIRM ALL FOUR EXIST IN THE NEW ACCOUNT
#
#   "Proposed" MUST be the Trail-4 generator (polarity fix, BatchNorm output),
#   i.e. trail4_best.pt or trail4_generator.pth. A "trial2_*" / Trail-2 file is
#   the WRONG (Tanh) model: it loads only partially under strict=False and
#   produces a near-flat, invalid reconstruction (see the 2026-07-10 run).
#
#   The three baselines lived under a SEPARATE top-level folder
#   (Paper1_ECG_Reconstruction) in the old account. Confirm they were migrated;
#   if they now sit under OCT_DIR, repoint them below.
# ----------------------------------------------------------------------------
CKPT = {
    # Baselines are RETRAINED by exp7 and SAVED to your own writable Drive
    # (DRIVE_ROOT/JBHI_Revision/baseline_checkpoints == OUT_ROOT/baseline_checkpoints),
    # so a read-only shared dataset is fine. Same paths are used to load them later.
    "CNN":      f"{DRIVE_ROOT}/JBHI_Revision/baseline_checkpoints/CNN_EncoderDecoder/best_model.pt",
    "UNet":     f"{DRIVE_ROOT}/JBHI_Revision/baseline_checkpoints/UNet_Baseline/best_model.pt",
    "WGAN-GP":  f"{DRIVE_ROOT}/JBHI_Revision/baseline_checkpoints/WGAN_GP/best_model.pt",
    "Proposed": f"{OCT_DIR}/Best-epoch-trail-4/trail4_best.pt",   # read from the shared dataset
}
CKPT_PROPOSED_ALT = f"{OCT_DIR}/trail4_generator.pth"   # fallback copy (state_dict)

# ----------------------------------------------------------------------------
# SensSmart-Tech cross-dataset (32 subjects, 338 records @ 1 kHz -> 8 kHz)
#   WFDB pairs: {base}_ecg.hea/.dat and {base}_pcg.hea/.dat ; ECG lead II
# ----------------------------------------------------------------------------
SENSSMART_WFDB  = f"{OCT_DIR}/senssmarttech-database/WFDB"
SENSSMART_PREP  = f"{OCT_DIR}/milestone2_preprocessed"     # cached .npz (keys ecg/pcg)

# ----------------------------------------------------------------------------
# Output root for ALL revision artifacts (created on demand)
#   Each experiment writes to a named sub-folder here.
# ----------------------------------------------------------------------------
OUT_ROOT = f"{DRIVE_ROOT}/JBHI_Revision"

def out_dir(name):
    """Return (and create) an output sub-folder under OUT_ROOT."""
    p = os.path.join(OUT_ROOT, name)
    os.makedirs(p, exist_ok=True)
    return p

# ----------------------------------------------------------------------------
# Signal + inference constants
# ----------------------------------------------------------------------------
FS          = 8000        # sampling rate (Hz)
SEG_LEN     = 64000       # samples per segment (8 s)
NUM_SCENARIOS = 3         # rest / exercise / recovery one-hot
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# R-peak detector settings (identical to metrics.ipynb; keep fixed for reproducibility)
RPEAK_DISTANCE   = 400    # min inter-peak samples = 50 ms @ 8 kHz
RPEAK_PROMINENCE = 0.5

# SensSmart preprocessing bands (from cross_dataset_validation.ipynb)
ECG_BAND = (0.5, 40.0)
PCG_BAND = (25.0, 400.0)
FILTER_ORDER = 4

# Scenario one-hots
SCEN_REST     = [1.0, 0.0, 0.0]
SCEN_EXERCISE = [0.0, 1.0, 0.0]
SCEN_RECOVERY = [0.0, 0.0, 1.0]


def scenario_to_index(scenario_str):
    """Map EPHNOGRAM free-text scenario -> class index {0,1,2}.

    Matches the mapping used in COLAB_IEEE_FIGURES.ipynb.
    """
    s = str(scenario_str).lower()
    if "rest" in s or "laying" in s:
        return 0
    if "exercise" in s or "bicycle" in s or "treadmill" in s or "running" in s:
        return 1
    return 2
