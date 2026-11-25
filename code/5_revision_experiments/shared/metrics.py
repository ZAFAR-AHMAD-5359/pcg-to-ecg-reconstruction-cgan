"""
Metrics for the JBHI revision.

Two families:
  (A) Signal-fidelity metrics  -- identical to the submitted paper's metrics.ipynb
                                  (PSNR, SNR, SSIM, Pearson r, MAE, RMSE,
                                   R-peak AMPLITUDE error, QRS-width error).
  (B) Temporal R-peak DETECTION metrics  -- NEW, added to answer Reviewer 1 #1.
       These detect R-peaks INDEPENDENTLY in the reconstructed ECG and match
       them to ground-truth peaks within a tolerance window, then report
       sensitivity, PPV, F1, FP, FN, timing-error stats, RR-interval error,
       and heart-rate error. This is the metric the paper was missing.

Keep the detector settings fixed (config_paths.RPEAK_*) so every experiment and
the paper agree.
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# R-peak detection (single, shared definition)
# ---------------------------------------------------------------------------
def detect_r_peaks(ecg, fs=8000, distance=400, prominence=0.5):
    """scipy find_peaks on an ECG trace. Returns integer sample indices."""
    peaks, _ = signal.find_peaks(np.asarray(ecg).ravel(),
                                 distance=distance, prominence=prominence)
    return peaks


# ---------------------------------------------------------------------------
# (A) Signal-fidelity metrics
# ---------------------------------------------------------------------------
def calc_psnr(real, fake):
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    mse = np.mean((real - fake) ** 2)
    if mse < 1e-10:
        return 100.0
    peak = np.max(np.abs(real))
    return float(20 * np.log10(peak / np.sqrt(mse)))


def calc_snr(real, fake):
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    noise = real - fake
    return float(10 * np.log10(np.sum(real ** 2) / (np.sum(noise ** 2) + 1e-12)))


def calc_correlation(real, fake):
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    try:
        c, _ = pearsonr(real, fake)
        return float(c)
    except Exception:
        return 0.0


def calc_ssim(real, fake):
    from skimage.metrics import structural_similarity as compare_ssim  # lazy: Colab-only dep
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    r = (real - real.min()) / (real.max() - real.min() + 1e-8)
    f = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
    return float(compare_ssim(r, f, data_range=1.0))


def calc_mae(real, fake):
    return float(np.mean(np.abs(np.asarray(real).ravel() - np.asarray(fake).ravel())))


def calc_rmse(real, fake):
    d = np.asarray(real).ravel() - np.asarray(fake).ravel()
    return float(np.sqrt(np.mean(d ** 2)))


def calc_r_peak_amplitude_error(real, fake, fs=8000, normalize01=True, **kw):
    """Paper's 'R-peak error' -> renamed 'R-peak AMPLITUDE error'.

    Detects R-peaks on the GROUND TRUTH only, then averages the amplitude error at
    those locations. Kept ONLY as a fidelity measure; NOT a timing metric (R1 #1).

    normalize01=True (DEFAULT) min-max normalizes both signals to [0,1] before the
    error, which REPRODUCES the paper's ~0.126 scale (confirmed: the paper's metric
    is [0,1]-normalized). normalize01=False gives the raw z-scored form, which is
    ~8x larger (z-scored ECG spans ~6-8 units vs 1 for [0,1]) -- that is what the
    first exp4/6/7 runs reported (~1.0). Use the default for consistency with the
    manuscript. Returns np.nan when the ground truth has no detectable R-peaks.
    """
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    peaks = detect_r_peaks(real, fs=fs, **kw)
    if len(peaks) == 0:
        return np.nan
    if normalize01:
        real = (real - real.min()) / (real.max() - real.min() + 1e-8)
        fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
    return float(np.mean(np.abs(real[peaks] - fake[peaks])))


def _qrs_width_at_peak(ecg, peak_idx, fs=8000, threshold=0.5, search_ms=25):
    amp = abs(ecg[peak_idx]); thr = threshold * amp
    win = int(search_ms / 1000 * fs)
    q = peak_idx
    for j in range(peak_idx - 1, max(peak_idx - win, 0), -1):
        if abs(ecg[j]) < thr:
            q = j; break
    s = peak_idx
    for j in range(peak_idx + 1, min(peak_idx + win, len(ecg))):
        if abs(ecg[j]) < thr:
            s = j; break
    return (s - q) / fs * 1000.0


def calc_qrs_width_error(real, fake, fs=8000, **kw):
    real, fake = np.asarray(real).ravel(), np.asarray(fake).ravel()
    peaks = detect_r_peaks(real, fs=fs, **kw)
    if len(peaks) == 0:
        return np.nan
    errs = [abs(_qrs_width_at_peak(real, p, fs) - _qrs_width_at_peak(fake, p, fs))
            for p in peaks]
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# (B) Temporal R-peak DETECTION metrics  (NEW -- Reviewer 1 #1)
# ---------------------------------------------------------------------------
def match_peaks(true_peaks, pred_peaks, tol_samples):
    """Greedy nearest-neighbour matching within a tolerance window.

    Each true peak claims the closest unused predicted peak within tol_samples.
    Returns (matches, TP, FP, FN, signed_timing_errors_samples).
    signed error = pred_index - true_index (positive = reconstructed peak late).
    """
    true_peaks = list(np.asarray(true_peaks).astype(int))
    pred_peaks = list(np.asarray(pred_peaks).astype(int))
    used, matches, terr = set(), [], []
    for tp in true_peaks:
        cands = [(abs(pp - tp), pp) for pp in pred_peaks
                 if pp not in used and abs(pp - tp) <= tol_samples]
        if cands:
            _, pp = min(cands)
            used.add(pp); matches.append((tp, pp)); terr.append(pp - tp)
    TP = len(matches)
    FN = len(true_peaks) - TP
    FP = len(pred_peaks) - TP
    return matches, TP, FP, FN, np.asarray(terr, dtype=float)


def temporal_rpeak_metrics(true_ecg, pred_ecg, fs=8000, tol_ms=50,
                           distance=400, prominence=0.5):
    """Full temporal detection report for ONE segment.

    Returns a dict with detection (Se/PPV/F1/FP/FN), timing-error (ms),
    RR-interval error (ms), and heart-rate error (bpm). Segments with no
    ground-truth peaks return n_true=0 and NaN rates (exclude from means, but
    COUNT them -- a reconstruction that yields peaks where truth has none is a
    real failure; see caller handling).
    """
    tol = int(round(tol_ms / 1000 * fs))
    t_pk = detect_r_peaks(true_ecg, fs, distance, prominence)
    p_pk = detect_r_peaks(pred_ecg, fs, distance, prominence)

    matches, TP, FP, FN, terr = match_peaks(t_pk, p_pk, tol)
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    ppv  = TP / (TP + FP) if (TP + FP) else np.nan
    f1   = (2 * sens * ppv / (sens + ppv)
            if (sens and ppv and not np.isnan(sens) and not np.isnan(ppv)
                and (sens + ppv) > 0) else np.nan)

    terr_ms = np.abs(terr) / fs * 1000.0
    mae_t   = float(np.mean(terr_ms)) if terr_ms.size else np.nan
    med_t   = float(np.median(terr_ms)) if terr_ms.size else np.nan
    iqr_t   = float(np.percentile(terr_ms, 75) - np.percentile(terr_ms, 25)) \
              if terr_ms.size else np.nan

    # RR-interval error on consecutive MATCHED pairs only (avoids missed-beat bias)
    rr_err = np.nan
    if len(matches) >= 2:
        tp_sorted = sorted(matches)
        rr_true, rr_pred = [], []
        for (t0, p0), (t1, p1) in zip(tp_sorted[:-1], tp_sorted[1:]):
            rr_true.append((t1 - t0) / fs * 1000.0)
            rr_pred.append((p1 - p0) / fs * 1000.0)
        rr_err = float(np.mean(np.abs(np.array(rr_true) - np.array(rr_pred))))

    # Heart-rate error (bpm), estimator = 60000 / median-RR
    def _hr(pk):
        if len(pk) < 2:
            return np.nan
        return 60000.0 / np.median(np.diff(pk) / fs * 1000.0)
    hr_true, hr_pred = _hr(t_pk), _hr(p_pk)
    hr_err = float(abs(hr_true - hr_pred)) if not (np.isnan(hr_true) or np.isnan(hr_pred)) else np.nan

    return {
        "n_true": int(len(t_pk)), "n_pred": int(len(p_pk)),
        "TP": TP, "FP": FP, "FN": FN,
        "sensitivity": sens, "ppv": ppv, "f1": f1,
        "mae_timing_ms": mae_t, "median_timing_ms": med_t, "iqr_timing_ms": iqr_t,
        "rr_error_ms": rr_err,
        "hr_true_bpm": hr_true, "hr_pred_bpm": hr_pred, "hr_error_bpm": hr_err,
    }


def aggregate_temporal(rows):
    """Aggregate a list of per-segment temporal_rpeak_metrics dicts.

    Detection totals (TP/FP/FN) are POOLED then Se/PPV/F1 recomputed (the
    correct way to aggregate detection over segments). Timing/RR/HR errors are
    averaged over segments where they are defined; the valid-N is reported so
    the denominator is transparent.
    """
    rows = [r for r in rows if r is not None]
    TP = sum(r["TP"] for r in rows)
    FP = sum(r["FP"] for r in rows)
    FN = sum(r["FN"] for r in rows)
    sens = TP / (TP + FN) if (TP + FN) else np.nan
    ppv  = TP / (TP + FP) if (TP + FP) else np.nan
    f1   = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else np.nan

    def _mean(key):
        v = [r[key] for r in rows if r[key] is not None and not np.isnan(r[key])]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)

    out = {"n_segments": len(rows), "TP": TP, "FP": FP, "FN": FN,
           "sensitivity": sens, "ppv": ppv, "f1": f1}
    for k in ["mae_timing_ms", "median_timing_ms", "iqr_timing_ms",
              "rr_error_ms", "hr_error_bpm"]:
        m, s, n = _mean(k)
        out[k + "_mean"], out[k + "_std"], out[k + "_validN"] = m, s, n
    return out


# ---------------------------------------------------------------------------
# Preprocessing helpers (shared with the paper pipeline)
# ---------------------------------------------------------------------------
def ensure_positive_polarity(ecg):
    """Flip so R-peaks point up. Applied to the ECG TARGET only (never PCG)."""
    ecg = np.asarray(ecg)
    return -ecg if np.abs(ecg.min()) > np.abs(ecg.max()) else ecg


def normalize_zscore(sig):
    sig = np.asarray(sig, dtype=np.float32)
    return ((sig - sig.mean()) / (sig.std() + 1e-8)).astype(np.float32)
