# %% [markdown]
# # exp1 -- Temporal R-peak detection metrics (Reviewer 1, comment 1)
#
# The submitted paper reports only an **amplitude** R-peak error
# (`calc_r_peak_error`, renamed here `calc_r_peak_amplitude_error`). Reviewer 1
# asks whether the reconstructed ECG recovers R-peaks at the **right times**.
# This script supplies that missing timing metric.
#
# Pipeline for every one of the 453 EPHNOGRAM test segments:
#   1. Load the raw `.npz` chunk (keys `ecg`, `pcg`, `subject_id`, `scenario`).
#   2. Polarity-fix the ground-truth ECG, then z-score the ECG target and the
#      PCG input (exactly the paper pipeline).
#   3. Run the Proposed generator, `model(pcg, scenario_onehot)`.
#   4. z-score the reconstruction and SAVE it to `out_dir('exp1_reconstructions')`
#      as `.npy` (exp3 reuses these waveforms).
#   5. Detect R-peaks independently in the ground truth and the reconstruction,
#      match them within +/-50 ms and +/-150 ms, and score
#      sensitivity, PPV, F1, FP, FN, timing error, RR error, and HR error.
#   6. Score a **PCG-only** S1 detector (Shannon-energy envelope of the
#      band-passed PCG) against the same ground-truth R-peaks, so the table
#      answers "does the reconstruction beat PCG alone?".
#
# Outputs (all under `config_paths.OUT_ROOT`):
#   * `exp1_reconstructions/<stem>.npy`         -- z-scored reconstructions
#   * `exp1_rpeak_timing/results.json`          -- every number cited by the LaTeX
#   * `exp1_rpeak_timing/exp1_temporal_table.tex`     -- main comparison table
#   * `exp1_rpeak_timing/exp1_persubject_table.tex`   -- per-subject / per-scenario
#   * `exp1_rpeak_timing/fig_exp1_overlay_hist.{pdf,png}`  -- double-column figure
#   * `exp1_rpeak_timing/fig_exp1_persubject_f1.{pdf,png}` -- single-column figure
#
# The script is idempotent and resumable: cached reconstructions are reused, and
# any missing checkpoint or data path prints a clear note instead of crashing.

# %%
# ---------------------------------------------------------------------------
# Cell 1 -- environment, Drive mount, shared imports
# ---------------------------------------------------------------------------
# NOTE: copy the LOCAL `shared/` folder (config_paths.py, models.py, metrics.py,
# ieee_style.py) to `MyDrive/JBHI_Revision/shared` BEFORE running, or paste the
# four modules into this runtime. The sys.path line below points at that folder.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("[note] not in Colab; assuming the shared modules are importable locally.")

import sys
sys.path.append('/content/drive/MyDrive/JBHI_Revision/shared')  # copy shared/ here first

import os
import glob
import json
import numpy as np
import torch
import scipy.signal as sp_signal

import config_paths
import models
import metrics
import ieee_style

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **kw):
        return x

ieee_style.set_ieee_style()
import matplotlib.pyplot as plt  # after set_ieee_style so rcParams apply

FS       = config_paths.FS
DIST     = config_paths.RPEAK_DISTANCE      # 400 samples = 50 ms @ 8 kHz
PROM     = config_paths.RPEAK_PROMINENCE    # 0.5
DEVICE   = config_paths.DEVICE
TOLS_MS  = (50, 150)                         # the two tolerance windows scored

RECON_DIR = config_paths.out_dir('exp1_reconstructions')  # feeds exp3
OUT_DIR   = config_paths.out_dir('exp1_rpeak_timing')

print(f"[cfg] device           : {DEVICE}")
print(f"[cfg] reconstructions  : {RECON_DIR}")
print(f"[cfg] results / figures: {OUT_DIR}")

# %% [markdown]
# ## Cell 2 -- helpers
#
# `temporal_metrics_from_peaks` mirrors `metrics.temporal_rpeak_metrics` but
# takes peak-index arrays directly, so the PCG-only S1 detector is scored with
# the exact same matching, timing, RR, and HR logic as the reconstruction.
# `detect_s1_peaks` is the PCG-only baseline: band-pass, normalized Shannon
# energy, smoothing, `find_peaks`, then a systole/diastole heuristic that labels
# S1 without ever looking at the ground-truth ECG.

# %%
def temporal_metrics_from_peaks(t_pk, p_pk, fs, tol_ms):
    """Score predicted peaks against true peaks; keys match temporal_rpeak_metrics."""
    tol = int(round(tol_ms / 1000 * fs))
    matches, TP, FP, FN, terr = metrics.match_peaks(t_pk, p_pk, tol)
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    ppv  = TP / (TP + FP) if (TP + FP) else np.nan
    f1   = (2 * sens * ppv / (sens + ppv)
            if (sens and ppv and not np.isnan(sens) and not np.isnan(ppv)
                and (sens + ppv) > 0) else np.nan)
    terr_ms = np.abs(terr) / fs * 1000.0
    mae_t = float(np.mean(terr_ms)) if terr_ms.size else np.nan
    med_t = float(np.median(terr_ms)) if terr_ms.size else np.nan
    iqr_t = (float(np.percentile(terr_ms, 75) - np.percentile(terr_ms, 25))
             if terr_ms.size else np.nan)
    rr_err = np.nan
    if len(matches) >= 2:
        ms = sorted(matches)
        rr_true, rr_pred = [], []
        for (t0, p0), (t1, p1) in zip(ms[:-1], ms[1:]):
            rr_true.append((t1 - t0) / fs * 1000.0)
            rr_pred.append((p1 - p0) / fs * 1000.0)
        rr_err = float(np.mean(np.abs(np.array(rr_true) - np.array(rr_pred))))

    def _hr(pk):
        pk = np.asarray(pk)
        if len(pk) < 2:
            return np.nan
        return 60000.0 / np.median(np.diff(pk) / fs * 1000.0)

    hr_t, hr_p = _hr(t_pk), _hr(p_pk)
    hr_err = float(abs(hr_t - hr_p)) if not (np.isnan(hr_t) or np.isnan(hr_p)) else np.nan
    return {
        "n_true": int(len(t_pk)), "n_pred": int(len(p_pk)),
        "TP": TP, "FP": FP, "FN": FN,
        "sensitivity": sens, "ppv": ppv, "f1": f1,
        "mae_timing_ms": mae_t, "median_timing_ms": med_t, "iqr_timing_ms": iqr_t,
        "rr_error_ms": rr_err,
        "hr_true_bpm": hr_t, "hr_pred_bpm": hr_p, "hr_error_bpm": hr_err,
    }


def _bandpass(x, lo, hi, fs, order):
    ny = 0.5 * fs
    b, a = sp_signal.butter(order, [lo / ny, hi / ny], btype='band')
    return sp_signal.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def detect_s1_peaks(pcg, fs):
    """PCG-only S1 baseline via a normalized Shannon-energy envelope.

    Returns (s1_peaks, envelope). No ground-truth ECG is used at any step.
    """
    lo, hi = config_paths.PCG_BAND
    x = _bandpass(pcg, lo, hi, fs, config_paths.FILTER_ORDER)
    x = x / (np.max(np.abs(x)) + 1e-8)
    se = -(x ** 2) * np.log(x ** 2 + 1e-12)          # normalized Shannon energy
    win = max(1, int(0.04 * fs))                     # 40 ms smoothing window
    env = np.convolve(se, np.ones(win) / win, mode='same')
    env = env / (np.max(np.abs(env)) + 1e-8)
    peaks, _ = sp_signal.find_peaks(env, distance=int(0.12 * fs), prominence=0.10)
    if len(peaks) < 3:
        return peaks, env
    # Systole (S1->S2) is shorter than diastole (S2->S1). Keep the peak set whose
    # leading intervals are the shorter ones; that set is S1.
    d = np.diff(peaks)
    mean_even = float(np.mean(d[0::2])) if len(d[0::2]) else np.inf
    mean_odd  = float(np.mean(d[1::2])) if len(d[1::2]) else np.inf
    s1 = peaks[0::2] if mean_even <= mean_odd else peaks[1::2]
    return s1, env


def _sanitize(obj):
    """Recursively convert numpy scalars and NaN to JSON-native values."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_sanitize(v) for v in obj.tolist()]
    return obj

# %% [markdown]
# ## Cell 3 -- load the test split and the Proposed generator
#
# The test set is every `.npz` chunk whose `subject_id` is in
# `split['test_subjects']`. The Proposed generator loads from
# `config_paths.CKPT['Proposed']`; if that file is missing the script still runs
# on any reconstructions already cached on disk.

# %%
split_path = config_paths.SPLIT_FILE
chunk_dir  = config_paths.EPH_CHUNKS_DIR

if not os.path.exists(split_path):
    raise SystemExit(f"[abort] split file not found: {split_path}\n"
                     f"        Fix config_paths.SPLIT_FILE to your Drive layout.")

with open(split_path, 'r') as f:
    split = json.load(f)
test_subjects = set(split['test_subjects'])
print(f"[data] test subjects: {sorted(test_subjects)}")

all_npz = sorted(glob.glob(os.path.join(chunk_dir, '*.npz')))
if not all_npz:
    raise SystemExit(f"[abort] no .npz chunks under {chunk_dir}")

test_chunks = []
for path in all_npz:
    try:
        d = np.load(path, allow_pickle=True)
        if str(d['subject_id']) in test_subjects:
            test_chunks.append(path)
    except Exception as e:
        print(f"[warn] could not read {os.path.basename(path)}: {e}")
print(f"[data] test segments: {len(test_chunks)}")

# Load the Proposed generator (guarded: cached reconstructions can stand in).
model = None
ckpt_path = config_paths.CKPT['Proposed']
if os.path.exists(ckpt_path):
    try:
        model, missing = models.load_model(models.ImprovedGeneratorNoTanh(),
                                            ckpt_path, device=DEVICE)
        print(f"[model] loaded Proposed generator "
              f"({models.count_params(model):,} params) from {ckpt_path}")
    except Exception as e:
        print(f"[warn] failed to load checkpoint ({e}); will use cached .npy only.")
elif os.path.exists(config_paths.CKPT_PROPOSED_ALT):
    try:
        model, missing = models.load_model(models.ImprovedGeneratorNoTanh(),
                                            config_paths.CKPT_PROPOSED_ALT, device=DEVICE)
        print(f"[model] loaded Proposed generator from fallback "
              f"{config_paths.CKPT_PROPOSED_ALT}")
    except Exception as e:
        print(f"[warn] fallback load failed ({e}); will use cached .npy only.")
else:
    print(f"[warn] no checkpoint at {ckpt_path}; will use cached .npy only.")

# %% [markdown]
# ## Cell 4 -- inference + per-segment scoring
#
# For each segment: z-score the ECG target and PCG input, run (or reload) the
# reconstruction, cache it, then score both the reconstruction and the PCG-only
# S1 detector at every tolerance. Matched-peak timing errors are accumulated for
# the histogram, and the best example is remembered for the overlay panel.

# %%
recon_rows = {t: [] for t in TOLS_MS}   # per-segment reconstruction metrics
pcg_rows   = {t: [] for t in TOLS_MS}   # per-segment PCG-only-baseline metrics
seg_meta   = []                          # subject / scenario / stem per segment
all_timing_err_ms = []                   # signed matched-peak errors (recon, 150 ms)
recon_index = {}                         # stem -> {subject, scenario, path}

for path in tqdm(test_chunks, desc="segments"):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        d = np.load(path, allow_pickle=True)
        ecg_raw = d['ecg'].astype(np.float32)
        pcg_raw = d['pcg'].astype(np.float32)
        subject = str(d['subject_id'])
        scen_idx = config_paths.scenario_to_index(d['scenario'])
    except Exception as e:
        print(f"[warn] skip {stem}: {e}")
        continue

    # Paper pipeline: polarity-fix the ECG target, z-score both model signals.
    ecg_z = metrics.normalize_zscore(metrics.ensure_positive_polarity(ecg_raw))
    pcg_z = metrics.normalize_zscore(pcg_raw)

    # Reconstruction: reuse cache if present, else run the generator.
    recon_path = os.path.join(RECON_DIR, f"{stem}.npy")
    if os.path.exists(recon_path):
        recon_z = np.load(recon_path).astype(np.float32)
    elif model is not None:
        pcg_t = torch.from_numpy(pcg_z).unsqueeze(0).unsqueeze(0).to(DEVICE)
        onehot = np.zeros(config_paths.NUM_SCENARIOS, dtype=np.float32)
        onehot[scen_idx] = 1.0
        scen_t = torch.from_numpy(onehot).unsqueeze(0).to(DEVICE)
        recon_raw = models.run_generator(model, pcg_t, scen_t, is_proposed=True)
        recon_z = metrics.normalize_zscore(recon_raw)   # z-score for scale-consistent detection
        np.save(recon_path, recon_z.astype(np.float32))
    else:
        print(f"[warn] no cache and no model for {stem}; skipping.")
        continue

    recon_index[stem] = {"subject": subject, "scenario": int(scen_idx),
                         "path": recon_path}
    seg_meta.append({"stem": stem, "subject": subject, "scenario": int(scen_idx)})

    # Ground-truth and PCG-only S1 peaks (shared once per segment).
    true_pk = metrics.detect_r_peaks(ecg_z, fs=FS, distance=DIST, prominence=PROM)
    s1_pk, _ = detect_s1_peaks(pcg_raw, FS)

    for tol in TOLS_MS:
        recon_rows[tol].append(
            metrics.temporal_rpeak_metrics(ecg_z, recon_z, fs=FS, tol_ms=tol,
                                           distance=DIST, prominence=PROM))
        pcg_rows[tol].append(
            temporal_metrics_from_peaks(true_pk, s1_pk, FS, tol))

    # Accumulate matched-peak timing errors for the histogram (recon, +/-150 ms).
    recon_pk = metrics.detect_r_peaks(recon_z, fs=FS, distance=DIST, prominence=PROM)
    _, _, _, _, terr = metrics.match_peaks(
        true_pk, recon_pk, int(round(150 / 1000 * FS)))
    all_timing_err_ms.extend((terr / FS * 1000.0).tolist())

with open(os.path.join(RECON_DIR, "_index.json"), "w") as f:
    json.dump(_sanitize(recon_index), f, indent=2)
print(f"[done] scored {len(seg_meta)} segments; "
      f"cached reconstructions in {RECON_DIR}")

# %% [markdown]
# ## Cell 5 -- aggregate overall, per-subject, per-scenario
#
# Detection totals pool TP/FP/FN before Se/PPV/F1 are recomputed (the correct way
# to aggregate detection over segments). Timing, RR, and HR errors average over
# the segments where they are defined, with the valid-N reported.

# %%
SCEN_NAME = {0: "rest", 1: "exercise", 2: "recovery"}

def _subset(rows, keep):
    return [r for r, m in zip(rows, seg_meta) if keep(m)]

results = {"n_segments": len(seg_meta), "tolerances_ms": list(TOLS_MS),
           "overall": {}, "per_subject": {}, "per_scenario": {}}

for tol in TOLS_MS:
    results["overall"][f"recon_{tol}ms"] = metrics.aggregate_temporal(recon_rows[tol])
    results["overall"][f"pcg_{tol}ms"]   = metrics.aggregate_temporal(pcg_rows[tol])

subjects  = sorted({m["subject"] for m in seg_meta})
scenarios = sorted({m["scenario"] for m in seg_meta})

for subj in subjects:
    results["per_subject"][subj] = {}
    for tol in TOLS_MS:
        results["per_subject"][subj][f"recon_{tol}ms"] = metrics.aggregate_temporal(
            _subset(recon_rows[tol], lambda m, s=subj: m["subject"] == s))
        results["per_subject"][subj][f"pcg_{tol}ms"] = metrics.aggregate_temporal(
            _subset(pcg_rows[tol], lambda m, s=subj: m["subject"] == s))

for sc in scenarios:
    name = SCEN_NAME.get(sc, str(sc))
    results["per_scenario"][name] = {}
    for tol in TOLS_MS:
        results["per_scenario"][name][f"recon_{tol}ms"] = metrics.aggregate_temporal(
            _subset(recon_rows[tol], lambda m, c=sc: m["scenario"] == c))
        results["per_scenario"][name][f"pcg_{tol}ms"] = metrics.aggregate_temporal(
            _subset(pcg_rows[tol], lambda m, c=sc: m["scenario"] == c))

results["n_matched_peaks_hist"] = len(all_timing_err_ms)

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(_sanitize(results), f, indent=2)
print(f"[done] wrote {os.path.join(OUT_DIR, 'results.json')}")

# Console summary
print("\n===== OVERALL (Se / PPV / F1) =====")
for tol in TOLS_MS:
    r = results["overall"][f"recon_{tol}ms"]
    p = results["overall"][f"pcg_{tol}ms"]
    print(f"  +/-{tol:>3} ms  recon: Se {r['sensitivity']:.3f}  PPV {r['ppv']:.3f}  "
          f"F1 {r['f1']:.3f}   |   PCG-only: Se {p['sensitivity']:.3f}  "
          f"PPV {p['ppv']:.3f}  F1 {p['f1']:.3f}")

# %% [markdown]
# ## Cell 6 -- LaTeX tables
#
# Main table: reconstruction vs PCG-only S1, at both tolerances, with the full
# metric set. Per-subject / per-scenario table: F1 and median timing broken down.

# %%
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def _row(label, agg):
    """Build one LaTeX row from an aggregate_temporal dict."""
    return (f"{label} & {_fmt(agg['sensitivity'])} & {_fmt(agg['ppv'])} & "
            f"{_fmt(agg['f1'])} & {agg['FP']} & {agg['FN']} & "
            f"{_fmt(agg['mae_timing_ms_mean'],1)} & "
            f"{_fmt(agg['median_timing_ms_mean'],1)} & "
            f"{_fmt(agg['iqr_timing_ms_mean'],1)} & "
            f"{_fmt(agg['rr_error_ms_mean'],1)} & "
            f"{_fmt(agg['hr_error_bpm_mean'],1)} \\\\")


tbl = []
tbl.append("% Auto-generated by exp1_R1_1_rpeak_timing.py -- do not edit by hand.")
tbl.append("% Paste into the manuscript; answers Reviewer 1, comment 1 (R1.1).")
tbl.append(r"\begin{table*}[t]")
tbl.append(r"\centering")
tbl.append(r"\caption{Temporal R-peak detection on the EPHNOGRAM test set. "
           r"Peaks are detected independently in each signal and matched to the "
           r"ground-truth R-peaks. The reconstructed ECG is compared against a "
           r"PCG-only S1 envelope detector at two tolerances. Timing statistics "
           r"average the per-segment mean, median, and IQR of matched-peak error.}")
tbl.append(r"\label{tab:rpeak_timing}")
tbl.append(r"\small")
tbl.append(r"\begin{tabular}{l l c c c c c c c c c c}")
tbl.append(r"\toprule")
tbl.append(r"Method & Tol. & Se & PPV & F1 & FP & FN & "
           r"Mean & Median & IQR & RR err. & HR err. \\")
tbl.append(r" &  &  &  &  &  &  & (ms) & (ms) & (ms) & (ms) & (bpm) \\")
tbl.append(r"\midrule")
for tol in TOLS_MS:
    tbl.append(_row(rf"Reconstructed ECG (Proposed) & $\pm${tol}\,ms",
                    results["overall"][f"recon_{tol}ms"]))
for tol in TOLS_MS:
    tbl.append(_row(rf"PCG-only S1 envelope & $\pm${tol}\,ms",
                    results["overall"][f"pcg_{tol}ms"]))
tbl.append(r"\bottomrule")
tbl.append(r"\end{tabular}")
tbl.append(r"\end{table*}")

with open(os.path.join(OUT_DIR, "exp1_temporal_table.tex"), "w") as f:
    f.write("\n".join(tbl) + "\n")
print(f"[done] wrote {os.path.join(OUT_DIR, 'exp1_temporal_table.tex')}")


# Per-subject / per-scenario table (reconstruction).
def _brk_row(label, meta_agg50, agg50, agg150):
    return (f"{label} & {meta_agg50['n_segments']} & {agg50['TP'] + agg50['FN']} & "
            f"{_fmt(agg50['sensitivity'])} & {_fmt(agg50['ppv'])} & "
            f"{_fmt(agg50['f1'])} & {_fmt(agg150['f1'])} & "
            f"{_fmt(agg50['median_timing_ms_mean'],1)} \\\\")


ptbl = []
ptbl.append("% Auto-generated by exp1_R1_1_rpeak_timing.py -- do not edit by hand.")
ptbl.append(r"\begin{table}[t]")
ptbl.append(r"\centering")
ptbl.append(r"\caption{Per-subject and per-scenario temporal R-peak detection for "
            r"the reconstructed ECG. F1 is reported at $\pm$50\,ms and $\pm$150\,ms; "
            r"the median column is the mean per-segment median timing error.}")
ptbl.append(r"\label{tab:rpeak_timing_breakdown}")
ptbl.append(r"\small")
ptbl.append(r"\begin{tabular}{l c c c c c c c}")
ptbl.append(r"\toprule")
ptbl.append(r"Group & Seg. & R-peaks & Se & PPV & F1$_{50}$ & F1$_{150}$ & Med. (ms) \\")
ptbl.append(r"\midrule")
ptbl.append(r"\multicolumn{8}{l}{\textit{Per subject}} \\")
for subj in subjects:
    a50  = results["per_subject"][subj]["recon_50ms"]
    a150 = results["per_subject"][subj]["recon_150ms"]
    ptbl.append(_brk_row(subj, a50, a50, a150))
ptbl.append(r"\midrule")
ptbl.append(r"\multicolumn{8}{l}{\textit{Per scenario}} \\")
for sc in scenarios:
    name = SCEN_NAME.get(sc, str(sc))
    a50  = results["per_scenario"][name]["recon_50ms"]
    a150 = results["per_scenario"][name]["recon_150ms"]
    ptbl.append(_brk_row(name.capitalize(), a50, a50, a150))
ptbl.append(r"\midrule")
o50, o150 = results["overall"]["recon_50ms"], results["overall"]["recon_150ms"]
ptbl.append(_brk_row(r"\textbf{Overall}", o50, o50, o150))
ptbl.append(r"\bottomrule")
ptbl.append(r"\end{tabular}")
ptbl.append(r"\end{table}")

with open(os.path.join(OUT_DIR, "exp1_persubject_table.tex"), "w") as f:
    f.write("\n".join(ptbl) + "\n")
print(f"[done] wrote {os.path.join(OUT_DIR, 'exp1_persubject_table.tex')}")

# %% [markdown]
# ## Cell 7 -- double-column figure (overlay + timing-error histogram)
#
# Panel (a): a representative segment with ground-truth vs reconstructed ECG, the
# detected GT peaks (circles), the reconstructed peaks (crosses), and the
# +/-50 ms tolerance band around each GT peak. Panel (b): the pooled histogram of
# matched-peak timing error.

# %%
# Pick the clearest example: best F1 at +/-50 ms among segments with >= 5 GT peaks.
best_i, best_key = -1, (-1.0, -1)
for i, r in enumerate(recon_rows[50]):
    if r["n_true"] >= 5 and r["f1"] is not None and not np.isnan(r["f1"]):
        key = (r["f1"], r["n_true"])
        if key > best_key:
            best_key, best_i = key, i
if best_i < 0:  # fall back to any segment with the most GT peaks
    best_i = int(np.argmax([r["n_true"] for r in recon_rows[50]])) if recon_rows[50] else -1

fig, axes = ieee_style.new_fig("double", height=2.7, nrows=1, ncols=2)
ax_a, ax_b = axes[0], axes[1]

if best_i >= 0:
    meta = seg_meta[best_i]
    stem = meta["stem"]
    ecg_z = metrics.normalize_zscore(
        metrics.ensure_positive_polarity(
            np.load(os.path.join(chunk_dir, f"{stem}.npz"),
                    allow_pickle=True)['ecg'].astype(np.float32)))
    recon_z = np.load(os.path.join(RECON_DIR, f"{stem}.npy")).astype(np.float32)
    t_pk = metrics.detect_r_peaks(ecg_z, fs=FS, distance=DIST, prominence=PROM)
    r_pk = metrics.detect_r_peaks(recon_z, fs=FS, distance=DIST, prominence=PROM)

    # 4 s window starting a little before the first GT peak, for legibility.
    win = int(4.0 * FS)
    start = max(0, int(t_pk[0]) - int(0.3 * FS)) if len(t_pk) else 0
    end = min(len(ecg_z), start + win)
    t = np.arange(start, end) / FS
    tol_s = 50 / 1000.0

    ax_a.plot(t, ecg_z[start:end], color=ieee_style.COLORS["truth"],
              lw=1.0, label="Ground truth")
    ax_a.plot(t, recon_z[start:end], color=ieee_style.COLORS["proposed"],
              lw=1.0, alpha=0.9, label="Reconstructed")
    band_lbl = r"$\pm$50 ms"
    for pk in t_pk:
        if start <= pk < end:
            ax_a.axvspan(pk / FS - tol_s, pk / FS + tol_s, color="0.6",
                         alpha=0.15, lw=0, label=band_lbl)
            band_lbl = None  # label only once
    gt_lbl, rc_lbl = "GT R-peak", "Recon R-peak"
    for pk in t_pk:
        if start <= pk < end:
            ax_a.plot(pk / FS, ecg_z[pk], 'o', mfc='none',
                      mec=ieee_style.COLORS["truth"], mew=1.2, ms=6, label=gt_lbl)
            gt_lbl = None
    for pk in r_pk:
        if start <= pk < end:
            ax_a.plot(pk / FS, recon_z[pk], 'x',
                      color=ieee_style.COLORS["proposed"], mew=1.2, ms=6, label=rc_lbl)
            rc_lbl = None
    ieee_style.finalize_axes(
        ax_a, xlabel="Time (s)", ylabel="Amplitude (z-score, a.u.)",
        title=f"(a) {meta['subject']}, {SCEN_NAME.get(meta['scenario'], '')}",
        legend=True, legend_loc="upper right", legend_ncol=2)
else:
    ax_a.text(0.5, 0.5, "no example segment", ha="center", va="center")

if all_timing_err_ms:
    errs = np.asarray(all_timing_err_ms)
    ax_b.hist(errs, bins=np.arange(-150, 151, 10),
              color=ieee_style.COLORS["proposed"], edgecolor="white", linewidth=0.4)
    ax_b.axvline(0.0, color="0.3", lw=0.9, ls="--")
    ieee_style.finalize_axes(
        ax_b, xlabel="Matched-peak timing error (ms)", ylabel="Count",
        title=f"(b) matched peaks (n = {len(errs)})")
else:
    ax_b.text(0.5, 0.5, "no matched peaks", ha="center", va="center")

ieee_style.save_fig(fig, OUT_DIR, "fig_exp1_overlay_hist")

# %% [markdown]
# ## Cell 8 -- single-column per-subject F1 bar chart
#
# Reconstruction vs PCG-only S1, F1 at +/-50 ms, grouped by subject with an
# overall pair.

# %%
labels = subjects + ["Overall"]
recon_f1 = [results["per_subject"][s]["recon_50ms"]["f1"] for s in subjects]
recon_f1.append(results["overall"]["recon_50ms"]["f1"])
pcg_f1 = [results["per_subject"][s]["pcg_50ms"]["f1"] for s in subjects]
pcg_f1.append(results["overall"]["pcg_50ms"]["f1"])

recon_f1 = [0.0 if (v is None or np.isnan(v)) else v for v in recon_f1]
pcg_f1   = [0.0 if (v is None or np.isnan(v)) else v for v in pcg_f1]

x = np.arange(len(labels))
w = 0.38
fig2, ax = ieee_style.new_fig("single", height=2.4)
ax.bar(x - w / 2, recon_f1, w, color=ieee_style.COLORS["proposed"],
       label="Reconstructed ECG")
ax.bar(x + w / 2, pcg_f1, w, color=ieee_style.COLORS["pcg"],
       label="PCG-only S1")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.set_ylim(0, 1.0)
ieee_style.finalize_axes(ax, xlabel="Subject", ylabel="R-peak F1 ($\\pm$50 ms)",
                         legend=True, legend_loc="upper right")
ieee_style.save_fig(fig2, OUT_DIR, "fig_exp1_persubject_f1")

print("\n[all done] exp1 temporal R-peak metrics complete.")
print(f"  results : {os.path.join(OUT_DIR, 'results.json')}")
print(f"  tables  : exp1_temporal_table.tex, exp1_persubject_table.tex")
print(f"  figures : fig_exp1_overlay_hist.(pdf|png), fig_exp1_persubject_f1.(pdf|png)")
print(f"  recons  : {RECON_DIR}  (reused by exp3)")
