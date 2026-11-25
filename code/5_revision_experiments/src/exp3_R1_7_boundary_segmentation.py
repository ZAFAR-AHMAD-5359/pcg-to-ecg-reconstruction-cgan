# %% [markdown]
# # exp3 - EPHNOGRAM three-arm cardiac-cycle boundary study (Reviewer 1, comment R1.7)
#
# This experiment delivers the same-dataset true-versus-reconstructed boundary
# comparison the reviewer requested. EPHNOGRAM supplies paired ECG for every test
# segment, so a true-ECG gold reference is available on the same signals.
#
# For each EPHNOGRAM test segment we derive cardiac-cycle phase boundaries from
# R-peaks obtained three ways.
#
# * Arm 1 (gold): R-peaks from the **true ECG**.
# * Arm 2: R-peaks from the **reconstructed ECG** (Proposed generator).
# * Arm 3: S1 onsets from **PCG only** (Shannon-energy envelope).
#
# The fixed-fraction phase model splits each cardiac cycle into S1 (0-15%),
# systole (15-50%), S2 (50-60%), and diastole (60-100%) of the cycle length.
# We report the mean absolute boundary error (ms) and the S1 and S2 onset
# localization error (ms) of arm 2 and arm 3 against arm 1.
#
# Outputs land in `config_paths.out_dir('exp3_boundary')`:
# a JSON of every number, a paste-ready LaTeX table, and one IEEE double-column
# figure (PDF plus 600 dpi PNG).
#
# **Reuse note.** This script reuses the Proposed reconstructions saved by exp1
# under `out_dir('exp1_reconstructions')`. If that cache is absent, this script
# regenerates the reconstructions with the Proposed checkpoint and writes the
# same cache, so the two experiments stay consistent.

# %% [markdown]
# ## Cell 1 - environment, Drive mount, shared imports
#
# The four shared modules live in `JBHI_Revision/shared` on Drive. Copy the local
# `Colab_Experiments/shared/` folder there once, or paste the modules into Colab.

# %%
# Mount Drive and expose the shared modules.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print('Not in Colab. Assuming shared modules are already importable.')

import sys
SHARED_DIR = '/content/drive/MyDrive/JBHI_Revision/shared'
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)

print('NOTE: copy the local Colab_Experiments/shared/ folder to')
print('      /content/drive/MyDrive/JBHI_Revision/shared  (or paste the modules in)')
print('      before running. All paths and helpers come from those modules.')

import os
import json
import glob
import numpy as np
import torch
import scipy.signal as sp_signal
from scipy.stats import pearsonr

import config_paths as CFG
import models as MDL
import metrics as MET
import ieee_style as IEEE

IEEE.set_ieee_style()
DEVICE = CFG.DEVICE
print(f'Device: {DEVICE}')

# %% [markdown]
# ## Cell 2 - phase model, tolerances, output folder
#
# The phase fractions and R-peak detector settings match the paper. The output
# folder is created on demand through `config_paths.out_dir`.

# %%
# Fixed-fraction cardiac-cycle phase model (start of each phase, as a cycle fraction).
PHASE_FRACTIONS = {
    'S1':       0.00,   # S1 onset  = cycle anchor
    'systole':  0.15,   # systole onset
    'S2':       0.50,   # S2 onset
    'diastole': 0.60,   # diastole onset
}
PHASE_ORDER = ['S1', 'systole', 'S2', 'diastole']

FS = CFG.FS                       # 8000 Hz
RPEAK_DISTANCE = CFG.RPEAK_DISTANCE
RPEAK_PROMINENCE = CFG.RPEAK_PROMINENCE
MATCH_TOL_MS = 150.0              # anchor-to-gold matching window (ms)

OUT_DIR = CFG.out_dir('exp3_boundary')
RECON_DIR = CFG.out_dir('exp1_reconstructions')          # shared with exp1
RECON_BUNDLE = os.path.join(RECON_DIR, 'proposed_reconstructions.npz')

# Preferred example segment for the figure (falls back to best correlation if absent).
PREFERRED_EXAMPLE = 'ECGPCG0001_chunk181'

print(f'Output folder:        {OUT_DIR}')
print(f'Reconstruction cache: {RECON_BUNDLE}')

# %% [markdown]
# ## Cell 3 - PCG-only S1 detection (Shannon-energy envelope)
#
# Arm 3 uses PCG alone. This detector isolates the heart-sound band, computes the
# average Shannon energy envelope, picks heart-sound peaks, then labels S1 with
# the systole-shorter-than-diastole duration asymmetry. No ECG is used.

# %%
def _bandpass(sig, fs, lo, hi, order=4):
    """Zero-phase Butterworth bandpass. Returns the input unchanged on failure."""
    try:
        nyq = fs / 2.0
        b, a = sp_signal.butter(order, [lo / nyq, hi / nyq], btype='band')
        return sp_signal.filtfilt(b, a, sig).astype(np.float32)
    except Exception:
        return np.asarray(sig, dtype=np.float32)


def shannon_envelope(pcg, fs=FS, band=(25.0, 400.0), smooth_s=0.02):
    """Average Shannon energy envelope of a PCG trace, z-scored to unit variance."""
    x = np.asarray(pcg, dtype=np.float32).ravel()
    x = x / (np.max(np.abs(x)) + 1e-8)
    xf = _bandpass(x, fs, band[0], band[1])
    xf = xf / (np.max(np.abs(xf)) + 1e-8)
    eps = 1e-8
    energy = -(xf ** 2) * np.log(xf ** 2 + eps)
    win = max(3, int(round(smooth_s * fs)))
    kernel = np.hanning(win)
    kernel = kernel / (kernel.sum() + eps)
    env = np.convolve(energy, kernel, mode='same')
    env = (env - env.mean()) / (env.std() + eps)
    return env.astype(np.float32)


def detect_s1_peaks(pcg, fs=FS, min_gap_s=0.08, prominence=0.20):
    """Detect S1 onsets from PCG only. Returns sorted integer sample indices."""
    env = shannon_envelope(pcg, fs)
    dist = max(1, int(round(min_gap_s * fs)))
    peaks, _ = sp_signal.find_peaks(env, distance=dist, prominence=prominence)
    if len(peaks) < 3:
        return peaks.astype(int)

    intervals = np.diff(peaks).astype(float)
    even_int = intervals[0::2]     # intervals that follow peaks 0, 2, 4, ...
    odd_int = intervals[1::2]      # intervals that follow peaks 1, 3, 5, ...
    mean_even = float(np.mean(even_int)) if even_int.size else np.inf
    mean_odd = float(np.mean(odd_int)) if odd_int.size else np.inf
    overall = float(np.mean(intervals))

    if abs(mean_even - mean_odd) < 0.10 * overall:
        # Ambiguous durations. Fall back to the louder peak set as S1.
        amp_even = float(np.mean(env[peaks[0::2]])) if peaks[0::2].size else -np.inf
        amp_odd = float(np.mean(env[peaks[1::2]])) if peaks[1::2].size else -np.inf
        s1 = peaks[0::2] if amp_even >= amp_odd else peaks[1::2]
    else:
        # Systole is the shorter interval. Its starting peak is S1.
        s1 = peaks[0::2] if mean_even <= mean_odd else peaks[1::2]
    return np.sort(np.asarray(s1, dtype=int))

# %% [markdown]
# ## Cell 4 - phase boundaries and boundary error
#
# Each anchor plus its following anchor defines one cycle. The phase boundaries
# sit at fixed fractions of that cycle. Boundary error is the absolute time gap
# (ms) between an arm boundary and the matched gold boundary.

# %%
def _next_interval_map(anchors):
    """Map each anchor (except the last) to its cycle length in samples."""
    anchors = np.sort(np.asarray(anchors, dtype=int))
    return {int(anchors[i]): int(anchors[i + 1] - anchors[i])
            for i in range(len(anchors) - 1)}


def boundary_errors_vs_gold(arm_anchors, gold_anchors, fs=FS, tol_ms=MATCH_TOL_MS):
    """Per-phase absolute boundary errors (ms) of an arm against the gold arm.

    Returns a dict {phase: [errors_ms]} plus the number of matched cycles.
    """
    out = {p: [] for p in PHASE_ORDER}
    gold_anchors = np.sort(np.asarray(gold_anchors, dtype=int))
    arm_anchors = np.sort(np.asarray(arm_anchors, dtype=int))
    if len(gold_anchors) < 2 or len(arm_anchors) < 2:
        return out, 0

    gold_len = _next_interval_map(gold_anchors)
    arm_len = _next_interval_map(arm_anchors)
    tol = int(round(tol_ms / 1000.0 * fs))

    matches, _, _, _, _ = MET.match_peaks(gold_anchors, arm_anchors, tol)
    n_cycles = 0
    for g, a in matches:
        lg = gold_len.get(int(g))
        la = arm_len.get(int(a))
        if lg is None or la is None:
            continue
        n_cycles += 1
        for phase in PHASE_ORDER:
            f = PHASE_FRACTIONS[phase]
            gold_b = g + f * lg
            arm_b = a + f * la
            out[phase].append(abs(arm_b - gold_b) / fs * 1000.0)
    return out, n_cycles

# %% [markdown]
# ## Cell 5 - load the EPHNOGRAM test set and the Proposed reconstructions
#
# This cell enumerates the test chunks, applies the paper preprocessing, and
# obtains each reconstructed ECG from the exp1 cache or the Proposed checkpoint.
# The reconstruction is rescaled to the true-ECG standard deviation, exactly as
# the paper pipeline does on this same dataset.

# %%
def _load_split_test_subjects():
    with open(CFG.SPLIT_FILE, 'r') as fh:
        splits = json.load(fh)
    for key in ('test_subjects', 'test'):
        if key in splits:
            return set(str(s) for s in splits[key])
    raise KeyError('No test-subject key found in split file.')


def _list_test_chunks():
    all_chunks = sorted(glob.glob(os.path.join(CFG.EPH_CHUNKS_DIR, '*.npz')))
    all_chunks = [p for p in all_chunks
                  if os.path.basename(p) != os.path.basename(CFG.SPLIT_FILE)]
    test_subjects = _load_split_test_subjects()
    test_chunks = []
    for path in all_chunks:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if 'subject_id' not in data:
            continue
        if str(data['subject_id']) in test_subjects:
            test_chunks.append(path)
    return test_chunks, test_subjects


def _load_recon_cache():
    if os.path.exists(RECON_BUNDLE):
        try:
            npz = np.load(RECON_BUNDLE, allow_pickle=True)
            cache = {k: npz[k].astype(np.float32) for k in npz.files}
            print(f'Loaded {len(cache)} cached reconstructions from exp1.')
            return cache
        except Exception as exc:
            print(f'Could not read reconstruction cache: {exc}. Will regenerate.')
    else:
        print('No exp1 reconstruction cache found. Will regenerate as needed.')
    return {}


_PROPOSED = {'model': None}


def _get_proposed_model():
    """Lazily load the Proposed generator (only when regeneration is needed)."""
    if _PROPOSED['model'] is not None:
        return _PROPOSED['model']
    ckpt = CFG.CKPT['Proposed']
    if not os.path.exists(ckpt):
        ckpt = CFG.CKPT_PROPOSED_ALT
    if not os.path.exists(ckpt):
        raise FileNotFoundError('Proposed checkpoint not found at either path.')
    gen, _ = MDL.load_model(MDL.ImprovedGeneratorNoTanh(), ckpt, device=DEVICE)
    _PROPOSED['model'] = gen
    print(f'Loaded Proposed generator from {ckpt}')
    return gen


def _reconstruct_raw(pcg_norm, scenario_idx):
    """Run the Proposed generator on one z-scored PCG. Returns the raw ECG output."""
    gen = _get_proposed_model()
    pcg_t = torch.from_numpy(pcg_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    scen_t = torch.zeros(1, CFG.NUM_SCENARIOS, device=DEVICE)
    scen_t[0, scenario_idx] = 1.0
    return MDL.run_generator(gen, pcg_t, scen_t, is_proposed=True).astype(np.float32)


# Build the per-segment records.
records = []
data_ok = os.path.exists(CFG.EPH_CHUNKS_DIR) and os.path.exists(CFG.SPLIT_FILE)
if not data_ok:
    print('EPHNOGRAM data or split file missing. Skipping. Check config_paths.')
else:
    test_chunks, test_subjects = _list_test_chunks()
    print(f'Test subjects: {sorted(test_subjects)}')
    print(f'Total test chunks: {len(test_chunks)}')

    recon_cache = _load_recon_cache()
    cache_dirty = False

    for idx, path in enumerate(test_chunks):
        key = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path, allow_pickle=True)
        ecg_raw = np.asarray(data['ecg'], dtype=np.float32)
        pcg_raw = np.asarray(data['pcg'], dtype=np.float32)
        subject = str(data['subject_id'])
        scenario_str = str(data['scenario'])
        scenario_idx = CFG.scenario_to_index(scenario_str)

        # Paper preprocessing: polarity fix on ECG only, then z-score both.
        ecg_norm = MET.normalize_zscore(MET.ensure_positive_polarity(ecg_raw))
        pcg_norm = MET.normalize_zscore(pcg_raw)

        if key in recon_cache:
            recon_raw = recon_cache[key]
        else:
            recon_raw = _reconstruct_raw(pcg_norm, scenario_idx)
            recon_cache[key] = recon_raw
            cache_dirty = True

        # Rescale the reconstruction to the true-ECG std (same-dataset pipeline step).
        real_std = float(np.std(ecg_norm))
        fake_std = float(np.std(recon_raw))
        recon_scaled = (recon_raw * (real_std / (fake_std + 1e-8))).astype(np.float32)

        records.append({
            'key': key,
            'subject': subject,
            'scenario': scenario_str,
            'path': path,
            'ecg_norm': ecg_norm,
            'pcg_norm': pcg_norm,
            'recon_scaled': recon_scaled,
        })

        if (idx + 1) % 50 == 0:
            print(f'  prepared {idx + 1}/{len(test_chunks)} segments')

    if cache_dirty:
        try:
            np.savez_compressed(RECON_BUNDLE, **recon_cache)
            print(f'Saved reconstruction cache -> {RECON_BUNDLE}')
        except Exception as exc:
            print(f'Could not save reconstruction cache: {exc}')

    print(f'Prepared {len(records)} test segments.')

# %% [markdown]
# ## Cell 6 - run the three-arm boundary analysis
#
# This cell detects anchors for the three arms, computes boundary errors of arm 2
# and arm 3 against the gold arm, then aggregates over every matched cycle.

# %%
def _aggregate(pooled):
    """Summarize a per-phase list-of-errors dict into means, std, and pooled mean."""
    summary = {}
    all_errs = []
    for phase in PHASE_ORDER:
        vals = np.asarray(pooled[phase], dtype=float)
        all_errs.extend(vals.tolist())
        if vals.size:
            summary[f'{phase}_err_ms_mean'] = float(np.mean(vals))
            summary[f'{phase}_err_ms_std'] = float(np.std(vals))
        else:
            summary[f'{phase}_err_ms_mean'] = float('nan')
            summary[f'{phase}_err_ms_std'] = float('nan')
    all_errs = np.asarray(all_errs, dtype=float)
    summary['mean_abs_boundary_err_ms'] = float(np.mean(all_errs)) if all_errs.size else float('nan')
    summary['boundary_err_ms_std'] = float(np.std(all_errs)) if all_errs.size else float('nan')
    summary['n_matched_cycles'] = len(pooled[PHASE_ORDER[0]])
    return summary


results = {}
if records:
    pooled_recon = {p: [] for p in PHASE_ORDER}
    pooled_pcg = {p: [] for p in PHASE_ORDER}
    n_with_gold = 0

    for rec in records:
        gold_anchors = MET.detect_r_peaks(
            rec['ecg_norm'], fs=FS, distance=RPEAK_DISTANCE, prominence=RPEAK_PROMINENCE)
        if len(gold_anchors) < 2:
            rec['n_gold_cycles'] = 0
            continue
        n_with_gold += 1

        recon_anchors = MET.detect_r_peaks(
            rec['recon_scaled'], fs=FS, distance=RPEAK_DISTANCE, prominence=RPEAK_PROMINENCE)
        s1_anchors = detect_s1_peaks(rec['pcg_norm'], fs=FS)

        err_recon, n_r = boundary_errors_vs_gold(recon_anchors, gold_anchors)
        err_pcg, n_p = boundary_errors_vs_gold(s1_anchors, gold_anchors)
        for phase in PHASE_ORDER:
            pooled_recon[phase].extend(err_recon[phase])
            pooled_pcg[phase].extend(err_pcg[phase])

        # Keep light per-segment fields for figure selection.
        rec['n_gold_cycles'] = int(len(gold_anchors) - 1)
        rec['gold_anchors'] = gold_anchors
        rec['recon_anchors'] = recon_anchors
        rec['s1_anchors'] = s1_anchors
        try:
            rec['corr'] = float(pearsonr(rec['ecg_norm'], rec['recon_scaled'])[0])
        except Exception:
            rec['corr'] = float('nan')
        rec['recon_cycle_err_mean'] = (
            float(np.mean([v for p in PHASE_ORDER for v in err_recon[p]]))
            if n_r else float('nan'))

    results = {
        'description': ('Same-dataset true-versus-reconstructed cardiac-cycle boundary '
                        'study on the EPHNOGRAM test set. EPHNOGRAM provides paired ECG, '
                        'so arm 1 (true ECG) is a genuine gold reference on the same signals.'),
        'phase_model_fractions': PHASE_FRACTIONS,
        'fs_hz': FS,
        'match_tolerance_ms': MATCH_TOL_MS,
        'rpeak_distance_samples': RPEAK_DISTANCE,
        'rpeak_prominence': RPEAK_PROMINENCE,
        'n_test_segments': len(records),
        'n_segments_with_gold': n_with_gold,
        'arms': {
            'arm2_reconstructed_ecg': _aggregate(pooled_recon),
            'arm3_pcg_only_s1': _aggregate(pooled_pcg),
        },
    }

    json_path = os.path.join(OUT_DIR, 'exp3_boundary_results.json')
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'Saved results JSON -> {json_path}')

    a2 = results['arms']['arm2_reconstructed_ecg']
    a3 = results['arms']['arm3_pcg_only_s1']
    print('\nMean absolute boundary error (ms):')
    print(f"  Reconstructed ECG : {a2['mean_abs_boundary_err_ms']:.2f}  "
          f"(cycles={a2['n_matched_cycles']})")
    print(f"  PCG-only S1       : {a3['mean_abs_boundary_err_ms']:.2f}  "
          f"(cycles={a3['n_matched_cycles']})")
else:
    print('No records available. Skipping analysis.')

# %% [markdown]
# ## Cell 7 - paste-ready LaTeX table
#
# The table lists the S1 onset error, S2 onset error, systole-boundary error,
# diastole-boundary error, and mean absolute boundary error for each arm.

# %%
def _fmt(x):
    return f'{x:.1f}' if (x == x) else '--'   # x != x guards NaN


if results:
    a2 = results['arms']['arm2_reconstructed_ecg']
    a3 = results['arms']['arm3_pcg_only_s1']

    lines = []
    lines.append('% ===================================================================')
    lines.append('% exp3 - EPHNOGRAM three-arm cardiac-cycle boundary error (R1.7)')
    lines.append('% Same-dataset true-versus-reconstructed comparison (paired ECG).')
    lines.append('% Paste into the Results section. Requires \\usepackage{booktabs}.')
    lines.append('% ===================================================================')
    lines.append('\\begin{table}[!t]')
    lines.append('\\centering')
    lines.append('\\caption{Cardiac-cycle phase-boundary error on the EPHNOGRAM test set. '
                 'Arm 1 (true ECG) is the gold reference. Errors are the mean absolute '
                 'time gap between each arm boundary and the gold boundary, in\\,ms. '
                 'The reconstructed-ECG arm recovers R-peak timing, so its boundaries '
                 'track the gold cycle. The PCG-only arm anchors on S1, so it lags by '
                 'the electromechanical delay.}')
    lines.append('\\label{tab:exp3_boundary}')
    lines.append('\\begin{tabular}{lccccc}')
    lines.append('\\toprule')
    lines.append('Arm & S1 onset & S2 onset & Systole & Diastole & Mean \\\\')
    lines.append(' & (ms) & (ms) & (ms) & (ms) & (ms) \\\\')
    lines.append('\\midrule')
    lines.append('True ECG (gold) & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\')
    lines.append('Reconstructed ECG & '
                 f"{_fmt(a2['S1_err_ms_mean'])} & {_fmt(a2['S2_err_ms_mean'])} & "
                 f"{_fmt(a2['systole_err_ms_mean'])} & {_fmt(a2['diastole_err_ms_mean'])} & "
                 f"{_fmt(a2['mean_abs_boundary_err_ms'])} \\\\")
    lines.append('PCG-only S1 & '
                 f"{_fmt(a3['S1_err_ms_mean'])} & {_fmt(a3['S2_err_ms_mean'])} & "
                 f"{_fmt(a3['systole_err_ms_mean'])} & {_fmt(a3['diastole_err_ms_mean'])} & "
                 f"{_fmt(a3['mean_abs_boundary_err_ms'])} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    table_tex = '\n'.join(lines)
    table_path = os.path.join(OUT_DIR, 'exp3_boundary_table.tex')
    with open(table_path, 'w') as fh:
        fh.write(table_tex + '\n')
    print(f'Saved LaTeX table -> {table_path}\n')
    print(table_tex)
else:
    print('No results. Skipping LaTeX table.')

# %% [markdown]
# ## Cell 8 - IEEE double-column figure
#
# The left panel shows one example PCG with the three sets of phase boundaries
# overlaid. True boundaries are solid, reconstructed boundaries are dashed, and
# PCG-only boundaries are dotted. The gold phases are shaded for context. The
# right panel shows the mean boundary error per arm.

# %%
def _choose_example():
    """Pick a clean example segment. Prefer the named chunk, else best correlation."""
    eligible = [r for r in records
                if r.get('n_gold_cycles', 0) >= 3
                and len(r.get('recon_anchors', [])) >= 3
                and len(r.get('s1_anchors', [])) >= 3]
    if not eligible:
        return None
    for r in eligible:
        if r['key'] == PREFERRED_EXAMPLE:
            return r
    eligible.sort(key=lambda r: (r.get('corr') if r.get('corr') == r.get('corr') else -1),
                  reverse=True)
    return eligible[0]


def _arm_boundary_times(anchors, win_start, win_end):
    """Boundary sample positions inside a window for one arm."""
    anchors = np.sort(np.asarray(anchors, dtype=int))
    lens = _next_interval_map(anchors)
    pts = []
    for a in anchors:
        la = lens.get(int(a))
        if la is None:
            continue
        for phase in PHASE_ORDER:
            b = a + PHASE_FRACTIONS[phase] * la
            if win_start <= b <= win_end:
                pts.append(b)
    return pts


if results:
    from matplotlib.lines import Line2D

    ex = _choose_example()
    fig, (axA, axB) = IEEE.new_fig('double', height=3.0, nrows=1, ncols=2,
                                   gridspec_kw={'width_ratios': [2.6, 1.0]})

    # ---- Panel A: example PCG with three sets of phase boundaries ----
    if ex is not None:
        pcg = ex['pcg_norm']
        gold = np.sort(ex['gold_anchors'])
        # Window over two gold cycles near the middle of the segment.
        mid = max(0, len(gold) // 2 - 1)
        mid = min(mid, len(gold) - 3)
        t0, t2 = int(gold[mid]), int(gold[mid + 2])
        pad = int(0.06 * FS)
        win_start = max(0, t0 - pad)
        win_end = min(len(pcg) - 1, t2 + pad)

        tt = (np.arange(win_start, win_end) - win_start) / FS
        axA.plot(tt, pcg[win_start:win_end], color=IEEE.COLORS['pcg'],
                 linewidth=0.8, zorder=2)

        # Shade gold phases for the two cycles and label them once.
        phase_colors = {'S1': IEEE.COLORS['truth'], 'systole': IEEE.COLORS['proposed'],
                        'S2': IEEE.COLORS['wgan'], 'diastole': IEEE.COLORS['muted']}
        span = {'S1': (0.00, 0.15), 'systole': (0.15, 0.50),
                'S2': (0.50, 0.60), 'diastole': (0.60, 1.00)}
        y0, y1 = float(np.min(pcg[win_start:win_end])), float(np.max(pcg[win_start:win_end]))
        yr = y1 - y0
        for c in (mid, mid + 1):
            g = int(gold[c])
            lg = int(gold[c + 1] - gold[c])
            for phase in PHASE_ORDER:
                f_lo, f_hi = span[phase]
                x_lo = (g + f_lo * lg - win_start) / FS
                x_hi = (g + f_hi * lg - win_start) / FS
                axA.axvspan(x_lo, x_hi, color=phase_colors[phase], alpha=0.12, zorder=1)
                if c == mid:
                    axA.text((x_lo + x_hi) / 2.0, y1 + 0.06 * yr, phase,
                             ha='center', va='bottom', fontsize=6.0,
                             color=phase_colors[phase], fontweight='bold')

        # Overlay each arm's phase boundaries: color and linestyle both encode the arm.
        arm_style = [
            ('gold', ex['gold_anchors'], 'black', '-', 1.0),
            ('recon', ex['recon_anchors'], IEEE.COLORS['proposed'], '--', 1.0),
            ('pcg', ex['s1_anchors'], IEEE.COLORS['cnn'], ':', 1.1),
        ]
        for _, anchors, color, ls, lw in arm_style:
            for b in _arm_boundary_times(anchors, win_start, win_end):
                axA.axvline((b - win_start) / FS, color=color, linestyle=ls,
                            linewidth=lw, alpha=0.9, zorder=3)

        axA.set_ylim(y0 - 0.08 * yr, y1 + 0.22 * yr)
        axA.set_xlim(tt[0], tt[-1])
        IEEE.finalize_axes(axA, xlabel='Time (s)', ylabel='PCG amplitude (a.u.)',
                           title=f"Phase boundaries: {ex['key']}")
        legend_handles = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=1.1,
                   label='True ECG (gold)'),
            Line2D([0], [0], color=IEEE.COLORS['proposed'], linestyle='--', linewidth=1.1,
                   label='Reconstructed ECG'),
            Line2D([0], [0], color=IEEE.COLORS['cnn'], linestyle=':', linewidth=1.3,
                   label='PCG-only S1'),
        ]
        axA.legend(handles=legend_handles, loc='lower right', fontsize=6.5,
                   framealpha=0.92, ncol=1)
    else:
        axA.text(0.5, 0.5, 'No clean example segment', ha='center', va='center',
                 transform=axA.transAxes)
        IEEE.finalize_axes(axA, xlabel='Time (s)', ylabel='PCG amplitude (a.u.)')

    # ---- Panel B: mean boundary error per arm ----
    a2 = results['arms']['arm2_reconstructed_ecg']
    a3 = results['arms']['arm3_pcg_only_s1']
    bar_labels = ['True ECG\n(ref)', 'Recon.\nECG', 'PCG-only\nS1']
    bar_vals = [0.0, a2['mean_abs_boundary_err_ms'], a3['mean_abs_boundary_err_ms']]
    bar_err = [0.0, a2['boundary_err_ms_std'], a3['boundary_err_ms_std']]
    bar_colors = [IEEE.COLORS['truth'], IEEE.COLORS['proposed'], IEEE.COLORS['cnn']]

    xpos = np.arange(len(bar_labels))
    axB.bar(xpos, bar_vals, yerr=bar_err, width=0.66, color=bar_colors,
            edgecolor='black', linewidth=0.7, capsize=2.5, error_kw={'elinewidth': 0.8})
    ymax = max([v + e for v, e in zip(bar_vals, bar_err)] + [1.0])
    for x, v, e in zip(xpos, bar_vals, bar_err):
        axB.text(x, v + (e if e == e else 0.0) + 0.03 * ymax,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    axB.set_ylim(0, ymax * 1.18)
    axB.set_xticks(xpos)
    axB.set_xticklabels(bar_labels)
    IEEE.finalize_axes(axB, xlabel='Reconstruction arm',
                       ylabel='Mean boundary error (ms)')

    IEEE.save_fig(fig, OUT_DIR, 'exp3_boundary_figure')
    print('Figure saved (PDF + PNG).')
else:
    print('No results. Skipping figure.')

# %% [markdown]
# ## Cell 9 - manifest
#
# The output folder now holds the results JSON, the LaTeX table, and the figure.
# The reconstruction cache is shared with exp1.

# %%
if results:
    print('exp3 outputs written to:', OUT_DIR)
    for name in ('exp3_boundary_results.json', 'exp3_boundary_table.tex',
                 'exp3_boundary_figure.pdf', 'exp3_boundary_figure.png'):
        p = os.path.join(OUT_DIR, name)
        print(f'  {"[ok] " if os.path.exists(p) else "[missing] "}{p}')
    print('\nThis is the same-dataset true-versus-reconstructed comparison the')
    print('reviewer requested. EPHNOGRAM supplies paired ECG, so arm 1 is a true')
    print('gold reference on the same signals.')
else:
    print('No outputs produced. Verify config_paths data and checkpoint paths.')
