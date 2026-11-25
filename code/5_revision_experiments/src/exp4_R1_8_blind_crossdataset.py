# %% [markdown]
# # exp4 - Blind cross-dataset PSNR (Reviewer 1, comment 8)
#
# The submitted paper claims that "amplitude reconstruction transfers" across
# datasets because SensSmart-Tech PSNR (17.04 dB) nearly matches the EPHNOGRAM
# PSNR (17.10 dB). That match is an artifact. The cross-dataset pipeline rescales
# every generator output by the target standard deviation:
#
#     fake_ecg = fake_raw * (real_std / fake_std)
#
# `real_std` is measured on the GROUND-TRUTH ECG, so the rescaling injects target
# information into the prediction. The honest, blind number uses the raw generator
# output with no target-std rescaling. This script computes that blind number and
# reports it next to the rescaled 17.04 dB.
#
# The script also runs four supporting checks:
#   1. mean std ratio (fake_std / real_std ~ 0.28), the size of the rescaling gap;
#   2. a no-bandpass control (training used no digital bandpass), which re-runs the
#      blind PSNR without the 25-400 Hz filter to test the filtering confound;
#   3. WFDB header verification (base_time, base_date, fs) for the paired _ecg and
#      _pcg records, evidence for or against sample-level synchronization;
#   4. a scale-invariance proof that SSIM and Pearson r are unchanged by the
#      rescaling, so the correlation collapse is genuine, not an amplitude effect.
#
# Outputs: a JSON results file, a paste-ready LaTeX table (metric, EPHNOGRAM,
# SmartTech-rescaled, SmartTech-blind), and one IEEE figure (Welch PSD of the two
# datasets' PCG on a single axis, the device-response cause of the collapse).

# %%
# --- Standard agent header (this cell is dropped by build_notebooks.py, which
#     mounts Drive and writes the shared modules for you). Kept so the file also
#     runs as a stand-alone Colab notebook.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import sys
    sys.path.append('/content/drive/MyDrive/JBHI_Revision/shared')  # copy shared/ here first
    print("NOTE: this stand-alone header expects the local shared/ folder copied to")
    print("      /content/drive/MyDrive/JBHI_Revision/shared  (or paste the four")
    print("      modules into /content). build_notebooks.py does this automatically.")
except Exception as _e:
    print(f"Not on Colab or Drive already mounted ({_e}); assuming shared modules on path.")

import config_paths, models, metrics, ieee_style

# %%
# --- Experiment imports, config, and output folder ---
import os
import json
import datetime
import numpy as np
import torch
import scipy.signal as ss

# wfdb is Colab-only; install on demand so the header verification and no-filter
# control can read the raw SensSmart-Tech records.
try:
    import wfdb
except ImportError:
    import subprocess
    print("Installing wfdb ...")
    subprocess.check_call(["pip", "install", "-q", "wfdb"])
    import wfdb

ieee_style.set_ieee_style()

EXP_NAME = "exp4_R1_8_blind_crossdataset"
OUT      = config_paths.out_dir(EXP_NAME)
DEVICE   = config_paths.DEVICE

# Re-run from scratch instead of loading cached stage results.
FORCE = False
# Optional caps (None = use every record). Keep None for the paper numbers.
MAX_CONTROL_RECORDS = None    # WFDB no-bandpass control
MAX_PSD_SEGMENTS    = 40      # segments per dataset averaged into the Welch PSD

# EPHNOGRAM reference column (paper Table / VERIFIED locally). Recomputed below
# if the EPHNOGRAM chunks are present; otherwise these constants are used.
EPH_REF = {
    "psnr": 17.10, "ssim": 0.903, "correlation": 0.610, "rpeak_amp_err": 0.126,
    "source": "paper reported / verified",
}

print(f"Device      : {DEVICE}")
print(f"Output dir  : {OUT}")
print(f"WFDB dir    : {config_paths.SENSSMART_WFDB}")
print(f"Cached npz  : {config_paths.SENSSMART_PREP}")


def _save_json(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved  {path}")
    return path


def _cache_failed(d):
    """True when a cached stage holds no records, or holds records whose every
    metric is NaN. The July 14 run cached 338 records with validN 0 for every
    metric, from the unstable 8 kHz ECG bandpass. A record count alone does not
    show that a stage succeeded, so test the aggregates for a single valid value.
    """
    if not isinstance(d, dict):
        return True
    counts = [v["validN"] for v in d.values()
              if isinstance(v, dict) and "validN" in v]
    for agg in (v for v in d.values() if isinstance(v, dict)):
        counts += [v["validN"] for v in agg.values()
                   if isinstance(v, dict) and "validN" in v]
    return not counts or max(counts) == 0


def _load_json(name):
    path = os.path.join(OUT, name)
    if (not FORCE) and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _agg(values):
    """Mean/std/min/max/median over finite values, with the valid-N denominator."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "median": None, "validN": 0}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v)), "validN": int(v.size)}

# %% [markdown]
# ## Load the proposed generator
#
# The proposed model (`ImprovedGeneratorNoTanh`, 19.32M parameters) is loaded from
# the Trail-4 checkpoint. Inference conditions on the exercise scenario one-hot,
# matching the cross-dataset protocol in the paper.

# %%
generator = None
GEN_PARAMS = None
ckpt_path = config_paths.CKPT["Proposed"]
if not os.path.exists(ckpt_path):
    ckpt_path = config_paths.CKPT_PROPOSED_ALT

if os.path.exists(ckpt_path):
    generator = models.ImprovedGeneratorNoTanh()
    generator, missing = models.load_model(generator, ckpt_path, device=DEVICE)
    GEN_PARAMS = models.count_params(generator)
    print(f"Generator loaded from {ckpt_path}")
    print(f"  trainable parameters: {GEN_PARAMS:,}")
    if missing is not None and (getattr(missing, 'missing_keys', None) or
                                getattr(missing, 'unexpected_keys', None)):
        print(f"  load_state_dict report: {missing}")
else:
    print(f"CHECKPOINT NOT FOUND at {ckpt_path}. Inference stages will be skipped.")

SCEN = torch.tensor(config_paths.SCEN_EXERCISE, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def reconstruct(pcg):
    """Run the generator on one z-scored PCG segment. Returns raw output (no rescale)."""
    x = torch.from_numpy(np.asarray(pcg, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return models.run_generator(generator, x, SCEN, is_proposed=True)

# %% [markdown]
# ## Helpers: WFDB loading and preprocessing
#
# The main filtered numbers come from the cached, already-preprocessed npz files
# (identical to the pipeline that produced the paper's 17.04 dB). The no-bandpass
# control and the header verification read the raw WFDB records directly, so the
# only difference between the two blind numbers is the 25-400 Hz filter.

# %%
def list_wfdb_records():
    """Record bases that have BOTH an _ecg and a _pcg record (the valid pairs).

    SensSmart-Tech stores each modality as a SEPARATE WFDB record named
    {subject}_{timestamp}_{ecg|pcg|ppg|acc}. Only sessions that have both an _ecg
    and a _pcg record form an ECG-PCG pair. Requiring both avoids the
    'No such file: ..._pcg.hea' failures seen on a partially-uploaded folder.
    The full PhysioNet database has 338 such pairs.
    """
    d = config_paths.SENSSMART_WFDB
    if not os.path.isdir(d):
        return []
    files = set(os.listdir(d))
    ecg = {f[:-len("_ecg.hea")] for f in files if f.endswith("_ecg.hea")}
    pcg = {f[:-len("_pcg.hea")] for f in files if f.endswith("_pcg.hea")}
    return sorted(ecg & pcg)


def _resample_to_target(sig, fs_in, fs_out=config_paths.FS):
    if int(fs_in) == int(fs_out):
        return np.asarray(sig, dtype=np.float32)
    g = np.gcd(int(fs_out), int(fs_in))
    return ss.resample_poly(sig, int(fs_out) // g, int(fs_in) // g).astype(np.float32)


def _bandpass(sig, fs, band, order=config_paths.FILTER_ORDER):
    # SOS + sosfiltfilt is numerically stable at low normalized cutoffs, where the
    # (b, a) form with filtfilt blows up to NaN (e.g. 0.5 Hz at 8 kHz -> cutoff
    # 0.000125). Guard the band against the Nyquist limit as well.
    nyq = fs / 2.0
    hi = min(band[1], nyq * 0.99)
    sos = ss.butter(order, [band[0] / nyq, hi / nyq], btype="band", output="sos")
    return ss.sosfiltfilt(sos, sig).astype(np.float32)


def read_wfdb_pair(base):
    """Load one raw ECG (lead II) + PCG pair. Returns (ecg, pcg, fs) or (None, None, None)."""
    try:
        d = config_paths.SENSSMART_WFDB
        ecg_rec = wfdb.rdrecord(os.path.join(d, f"{base}_ecg"))
        pcg_rec = wfdb.rdrecord(os.path.join(d, f"{base}_pcg"))
        names = list(ecg_rec.sig_name)
        lead = names.index("II") if "II" in names else 1
        ecg = ecg_rec.p_signal[:, lead].astype(np.float32)
        pcg = pcg_rec.p_signal.squeeze().astype(np.float32)
        return ecg, pcg, int(ecg_rec.fs)
    except Exception as e:
        print(f"  read failed for {base}: {e}")
        return None, None, None


def preprocess_pair(ecg_raw, pcg_raw, fs_in, apply_bandpass):
    """Resample -> optional bandpass -> ECG polarity fix -> z-score -> truncate.

    apply_bandpass=False reproduces the training condition (no digital bandpass).
    Returns (ecg, pcg) each of length SEG_LEN, or (None, None) when too short.
    """
    # Filter at the NATIVE rate (fs_in), BEFORE resampling. At 1 kHz the ECG
    # 0.5-40 Hz band is well conditioned; at 8 kHz it is not (all-NaN).
    if apply_bandpass:
        ecg_raw = _bandpass(ecg_raw, fs_in, config_paths.ECG_BAND)
        pcg_raw = _bandpass(pcg_raw, fs_in, config_paths.PCG_BAND)
    ecg = _resample_to_target(ecg_raw, fs_in)
    pcg = _resample_to_target(pcg_raw, fs_in)
    ecg = metrics.normalize_zscore(metrics.ensure_positive_polarity(ecg))
    pcg = metrics.normalize_zscore(pcg)
    T = config_paths.SEG_LEN
    if len(ecg) < T or len(pcg) < T:
        return None, None
    return ecg[:T], pcg[:T]

# %% [markdown]
# ## Stage 1 - Blind vs rescaled on the cached SmartTech records
#
# For each cached record, the script computes the raw (blind) reconstruction and
# the target-std-rescaled reconstruction, then scores both. PSNR separates the two
# columns; SSIM and Pearson r stay identical because they are scale-invariant.

# %%
def score_pair(real, fake_blind):
    """Return blind and rescaled metrics for one segment, plus the std ratio."""
    real_std = float(np.std(real))
    fake_std = float(np.std(fake_blind))
    ratio = fake_std / (real_std + 1e-8)                 # ~0.28 cross-dataset
    fake_rescaled = fake_blind * (real_std / (fake_std + 1e-8))
    return {
        "psnr_blind":     metrics.calc_psnr(real, fake_blind),
        "psnr_rescaled":  metrics.calc_psnr(real, fake_rescaled),
        "ssim_blind":     metrics.calc_ssim(real, fake_blind),
        "ssim_rescaled":  metrics.calc_ssim(real, fake_rescaled),
        "corr_blind":     metrics.calc_correlation(real, fake_blind),
        "corr_rescaled":  metrics.calc_correlation(real, fake_rescaled),
        "rpeak_blind":    metrics.calc_r_peak_amplitude_error(real, fake_blind, fs=config_paths.FS),
        "rpeak_rescaled": metrics.calc_r_peak_amplitude_error(real, fake_rescaled, fs=config_paths.FS),
        "std_ratio":      ratio,
    }


stage1 = _load_json("stage1_filtered.json")
if stage1 is not None and _cache_failed(stage1):
    print("Stage 1: cached result is empty or all-NaN -> discarding it and recomputing.")
    stage1 = None
stage1_recomputed = stage1 is None
if stage1 is None and generator is not None:
    prep = config_paths.SENSSMART_PREP
    use_cache = os.path.isdir(prep) and any(f.endswith(".npz") for f in os.listdir(prep))
    rows = []
    if use_cache:
        npz_files = sorted(f for f in os.listdir(prep) if f.endswith(".npz"))
        print(f"Stage 1: scoring {len(npz_files)} cached SmartTech records ...")
        for i, fn in enumerate(npz_files):
            data = np.load(os.path.join(prep, fn))          # already preprocessed
            real = metrics.normalize_zscore(metrics.ensure_positive_polarity(data["ecg"]))[:config_paths.SEG_LEN]
            pcg = metrics.normalize_zscore(data["pcg"])[:config_paths.SEG_LEN]
            if len(real) < config_paths.SEG_LEN or len(pcg) < config_paths.SEG_LEN:
                continue
            m = score_pair(real, reconstruct(pcg)); m["record"] = fn[:-4]; rows.append(m)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(npz_files)}")
    else:
        # No cache -> read the raw WFDB pairs directly (the full 338-pair database).
        bases = list_wfdb_records()
        print(f"Stage 1: no npz cache; reading {len(bases)} raw WFDB ECG-PCG pairs ...")
        for i, base in enumerate(bases):
            ecg_raw, pcg_raw, fs_in = read_wfdb_pair(base)
            if ecg_raw is None:
                continue
            real, pcg = preprocess_pair(ecg_raw, pcg_raw, fs_in, apply_bandpass=True)
            if real is None:
                continue
            m = score_pair(real, reconstruct(pcg)); m["record"] = base; rows.append(m)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(bases)}  (kept {len(rows)})")

    keys = ["psnr_blind", "psnr_rescaled", "ssim_blind", "ssim_rescaled",
            "corr_blind", "corr_rescaled", "rpeak_blind", "rpeak_rescaled", "std_ratio"]
    stage1 = {"n_records": len(rows),
              "agg": {k: _agg([r[k] for r in rows]) for k in keys},
              "per_record": rows}
    _save_json(stage1, "stage1_filtered.json")
elif stage1 is not None:
    print(f"Stage 1: loaded cached results ({stage1['n_records']} records).")
else:
    print("Stage 1 skipped (generator missing).")

def _fmt(x, spec=".2f"):
    return format(x, spec) if isinstance(x, (int, float)) else "n/a"

if stage1 is not None and stage1.get("n_records", 0) > 0:
    a = stage1["agg"]
    print("\n  SmartTech (filtered):")
    print(f"    PSNR blind    : {_fmt(a['psnr_blind']['mean'])} dB   (validN {a['psnr_blind']['validN']})")
    print(f"    PSNR rescaled : {_fmt(a['psnr_rescaled']['mean'])} dB   (paper headline 17.04)")
    print(f"    SSIM blind/rescaled : {_fmt(a['ssim_blind']['mean'], '.3f')} / {_fmt(a['ssim_rescaled']['mean'], '.3f')}")
    print(f"    Corr blind/rescaled : {_fmt(a['corr_blind']['mean'], '.3f')} / {_fmt(a['corr_rescaled']['mean'], '.3f')}")
    print(f"    mean std ratio : {_fmt(a['std_ratio']['mean'], '.3f')}   (expected ~0.28)")
else:
    print("\n  SmartTech: 0 records scored. Upload the full WFDB (338 ECG-PCG pairs)")
    print("  to the shared drive, delete the old exp4 stage*.json, and re-run.")

# %% [markdown]
# ## Stage 2 - No-bandpass control (filtering confound)
#
# Training applied no digital bandpass. This control re-reads the raw WFDB records
# and computes the blind PSNR with and without the 25-400 Hz filter on the same
# records, so any gap is attributable to the filter alone.

# %%
stage2 = _load_json("stage2_bandpass_control.json")
if stage2 is not None and _cache_failed(stage2):
    print("Stage 2: cached result is empty or all-NaN -> discarding it and recomputing.")
    stage2 = None
if stage2 is None and generator is not None:
    bases = list_wfdb_records()
    if MAX_CONTROL_RECORDS is not None:
        bases = bases[:MAX_CONTROL_RECORDS]
    print(f"Stage 2: no-bandpass control over {len(bases)} WFDB records ...")
    filt_blind, nofilt_blind = [], []
    for i, base in enumerate(bases):
        ecg_raw, pcg_raw, fs_in = read_wfdb_pair(base)
        if ecg_raw is None:
            continue
        ef, pf = preprocess_pair(ecg_raw, pcg_raw, fs_in, apply_bandpass=True)
        en, pn = preprocess_pair(ecg_raw, pcg_raw, fs_in, apply_bandpass=False)
        if ef is not None:
            filt_blind.append(metrics.calc_psnr(ef, reconstruct(pf)))
        if en is not None:
            nofilt_blind.append(metrics.calc_psnr(en, reconstruct(pn)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(bases)}")

    stage2 = {
        "n_filtered":   len(filt_blind),
        "n_nofilter":   len(nofilt_blind),
        "psnr_blind_filtered":  _agg(filt_blind),
        "psnr_blind_nofilter":  _agg(nofilt_blind),
    }
    _save_json(stage2, "stage2_bandpass_control.json")
elif stage2 is not None:
    print("Stage 2: loaded cached control results.")
else:
    print("Stage 2 skipped (generator missing).")

if stage2 is not None and stage2["psnr_blind_filtered"]["mean"] is not None:
    f_m = stage2["psnr_blind_filtered"]["mean"]
    n_m = stage2["psnr_blind_nofilter"]["mean"]
    print(f"\n  Blind PSNR with bandpass    : {f_m:.2f} dB (n={stage2['n_filtered']})")
    print(f"  Blind PSNR without bandpass : {n_m:.2f} dB (n={stage2['n_nofilter']})")
    print(f"  Filtering shifts blind PSNR by {n_m - f_m:+.2f} dB.")

# %% [markdown]
# ## Stage 3 - Header verification (synchronization evidence)
#
# The script reads the WFDB headers of the paired _ecg and _pcg records and checks
# whether they share a base date, base time, and sampling rate. Matching timestamps
# are evidence of a common clock; missing or mismatched timestamps mean sample-level
# alignment is not guaranteed, which explains the correlation collapse.

# %%
stage3 = _load_json("stage3_sync_headers.json")
if stage3 is None:
    bases = list_wfdb_records()
    print(f"Stage 3: reading headers for {len(bases)} record pairs ...")
    records, n_time_present, n_time_match, n_fs_match = [], 0, 0, 0
    examples = []
    for base in bases:
        try:
            he = wfdb.rdheader(os.path.join(config_paths.SENSSMART_WFDB, f"{base}_ecg"))
            hp = wfdb.rdheader(os.path.join(config_paths.SENSSMART_WFDB, f"{base}_pcg"))
        except Exception as e:
            examples.append({"record": base, "error": str(e)})
            continue

        def _s(x):
            return None if x is None else str(x)

        et, ed, pt, pd = _s(he.base_time), _s(he.base_date), _s(hp.base_time), _s(hp.base_date)
        fs_match = int(he.fs) == int(hp.fs)
        time_present = (et is not None) and (pt is not None)
        time_match = time_present and (et == pt) and (ed == pd)
        n_fs_match += int(fs_match)
        n_time_present += int(time_present)
        n_time_match += int(time_match)

        offset_s = None
        if time_present:
            try:
                fmt = "%H:%M:%S"
                t_e = datetime.datetime.strptime(et.split(".")[0], fmt)
                t_p = datetime.datetime.strptime(pt.split(".")[0], fmt)
                offset_s = (t_p - t_e).total_seconds()
            except Exception:
                offset_s = None

        rec = {"record": base, "ecg_base_date": ed, "ecg_base_time": et,
               "pcg_base_date": pd, "pcg_base_time": pt,
               "ecg_fs": int(he.fs), "pcg_fs": int(hp.fs),
               "fs_match": fs_match, "time_present": time_present,
               "time_match": time_match, "ecg_pcg_time_offset_s": offset_s}
        records.append(rec)
        if len(examples) < 5:
            examples.append(rec)

    n = len(records)
    stage3 = {
        "n_pairs": n,
        "n_fs_match": n_fs_match,
        "n_base_time_present": n_time_present,
        "n_ecg_pcg_time_match": n_time_match,
        "frac_time_present": (n_time_present / n) if n else None,
        "frac_time_match": (n_time_match / n) if n else None,
        "examples": examples,
        "records": records,
    }
    _save_json(stage3, "stage3_sync_headers.json")
else:
    print("Stage 3: loaded cached header results.")

if stage3 is not None and stage3["n_pairs"]:
    print(f"\n  Header pairs read      : {stage3['n_pairs']}")
    print(f"  sampling-rate matches  : {stage3['n_fs_match']}/{stage3['n_pairs']}")
    print(f"  base_time present      : {stage3['n_base_time_present']}/{stage3['n_pairs']}")
    print(f"  ecg/pcg time match     : {stage3['n_ecg_pcg_time_match']}/{stage3['n_pairs']}")

# %% [markdown]
# ## Stage 4 - Scale-invariance proof
#
# SSIM min-max normalizes both signals and Pearson r divides out amplitude, so a
# positive rescaling cannot change either. Stage 1 already stored blind and rescaled
# values for both; this stage measures the maximum per-record difference (expected
# ~0) and contrasts it with the PSNR gap the rescaling opens.

# %%
stage4 = _load_json("stage4_scale_invariance.json")
if stage4 is not None and stage1_recomputed:
    print("Stage 4: Stage 1 was recomputed -> discarding the derived cache.")
    stage4 = None            # stage 4 reads stage1["per_record"] and cannot outlive it
if stage4 is None and stage1 is not None:
    rows = stage1["per_record"]
    ssim_diffs = [abs(r["ssim_blind"] - r["ssim_rescaled"]) for r in rows
                  if r["ssim_blind"] is not None and r["ssim_rescaled"] is not None]
    corr_diffs = [abs(r["corr_blind"] - r["corr_rescaled"]) for r in rows
                  if r["corr_blind"] is not None and r["corr_rescaled"] is not None]
    psnr_gaps = [r["psnr_rescaled"] - r["psnr_blind"] for r in rows
                 if r["psnr_blind"] is not None and r["psnr_rescaled"] is not None]

    # Independent synthetic sanity check: scale a segment by 5x and confirm equality.
    rng = np.random.default_rng(0)
    real_d = rng.standard_normal(4000).astype(np.float32)
    fake_d = (0.28 * rng.standard_normal(4000)).astype(np.float32)
    synth = {
        "ssim_1x_vs_5x_diff": abs(metrics.calc_ssim(real_d, fake_d) -
                                  metrics.calc_ssim(real_d, 5.0 * fake_d)),
        "corr_1x_vs_5x_diff": abs(metrics.calc_correlation(real_d, fake_d) -
                                  metrics.calc_correlation(real_d, 5.0 * fake_d)),
    }
    stage4 = {
        "max_ssim_abs_diff": float(np.max(ssim_diffs)) if ssim_diffs else None,
        "max_corr_abs_diff": float(np.max(corr_diffs)) if corr_diffs else None,
        "mean_psnr_gap_db":  float(np.mean(psnr_gaps)) if psnr_gaps else None,
        "synthetic_check":   synth,
    }
    _save_json(stage4, "stage4_scale_invariance.json")
elif stage4 is not None:
    print("Stage 4: loaded cached scale-invariance results.")
else:
    print("Stage 4 skipped (Stage 1 results missing).")

if stage4 is not None:
    print(f"\n  max |SSIM blind - rescaled| : {stage4['max_ssim_abs_diff']}")
    print(f"  max |corr blind - rescaled| : {stage4['max_corr_abs_diff']}")
    print(f"  mean PSNR gap (rescaled-blind): {stage4['mean_psnr_gap_db']} dB")
    print(f"  synthetic 1x-vs-5x diffs      : {stage4['synthetic_check']}")

# %% [markdown]
# ## Stage 5 (optional) - Recompute the EPHNOGRAM reference
#
# When the EPHNOGRAM test chunks are present, the script recomputes the in-domain
# blind and rescaled metrics. In domain, the std ratio is near 1, so blind and
# rescaled PSNR nearly coincide, unlike the cross-dataset case. When the chunks are
# absent, the constants in `EPH_REF` are used.

# %%
def _scen_onehot(idx):
    v = [0.0, 0.0, 0.0]
    v[idx] = 1.0
    return v


stage5 = _load_json("stage5_ephnogram.json")
if stage5 is None and generator is not None and os.path.isdir(config_paths.EPH_CHUNKS_DIR):
    test_subjects = None
    if os.path.exists(config_paths.SPLIT_FILE):
        with open(config_paths.SPLIT_FILE, encoding="utf-8") as f:
            split = json.load(f)
        test_subjects = set(split.get("test_subjects", split.get("test", [])))
    files = sorted(f for f in os.listdir(config_paths.EPH_CHUNKS_DIR)
                   if f.endswith(".npz") and "split" not in f)
    print(f"Stage 5: recomputing EPHNOGRAM reference over up to {len(files)} chunks ...")
    rows = []
    for fn in files:
        try:
            data = np.load(os.path.join(config_paths.EPH_CHUNKS_DIR, fn), allow_pickle=True)
        except Exception:
            continue
        if "ecg" not in data or "pcg" not in data:
            continue
        subj = str(np.asarray(data["subject_id"]).ravel()[0]) if "subject_id" in data else ""
        if test_subjects and subj not in test_subjects:
            continue
        real = metrics.normalize_zscore(metrics.ensure_positive_polarity(data["ecg"]))[:config_paths.SEG_LEN]
        pcg  = metrics.normalize_zscore(data["pcg"])[:config_paths.SEG_LEN]
        if len(real) < config_paths.SEG_LEN or len(pcg) < config_paths.SEG_LEN:
            continue
        scen_idx = config_paths.scenario_to_index(np.asarray(data["scenario"]).ravel()[0]) \
            if "scenario" in data else 1
        x = torch.from_numpy(pcg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        s = torch.tensor(_scen_onehot(scen_idx), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        fake = models.run_generator(generator, x, s, is_proposed=True)
        rows.append(score_pair(real, fake))

    if rows:
        keys = ["psnr_blind", "psnr_rescaled", "ssim_blind", "ssim_rescaled",
                "corr_blind", "corr_rescaled", "rpeak_blind", "rpeak_rescaled", "std_ratio"]
        stage5 = {"n_segments": len(rows),
                  "agg": {k: _agg([r[k] for r in rows]) for k in keys}}
        _save_json(stage5, "stage5_ephnogram.json")
        print(f"  EPHNOGRAM recomputed over {len(rows)} segments.")
    else:
        print("  No matching EPHNOGRAM chunks; keeping EPH_REF constants.")
elif stage5 is not None:
    print("Stage 5: loaded cached EPHNOGRAM recompute.")
else:
    print("Stage 5 skipped (generator or EPHNOGRAM chunks missing); using EPH_REF constants.")

# Resolve the EPHNOGRAM reference column: recomputed rescaled values if available.
if stage5 is not None and stage5.get("agg"):
    a = stage5["agg"]
    EPH_COL = {
        "psnr": a["psnr_rescaled"]["mean"], "ssim": a["ssim_rescaled"]["mean"],
        "correlation": a["corr_rescaled"]["mean"], "rpeak_amp_err": a["rpeak_rescaled"]["mean"],
        "psnr_blind": a["psnr_blind"]["mean"], "std_ratio": a["std_ratio"]["mean"],
        "source": "recomputed",
    }
else:
    EPH_COL = dict(EPH_REF)
    EPH_COL["psnr_blind"] = None
    EPH_COL["std_ratio"] = None

# %% [markdown]
# ## Assemble the results JSON and the LaTeX table
#
# The table columns are EPHNOGRAM, SmartTech (rescaled), and SmartTech (blind).
# PSNR is the only row that moves between the two SmartTech columns; SSIM and
# Pearson r are identical, which is exactly the point of R1.8.

# %%
def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


results = {
    "experiment": EXP_NAME,
    "generator_params": GEN_PARAMS,
    "ephnogram_reference": EPH_COL,
    "smarttech_filtered": (stage1["agg"] if stage1 else None),
    "smarttech_n": (stage1["n_records"] if stage1 else None),
    "bandpass_control": stage2,
    "synchronization": (None if stage3 is None else
                        {k: stage3[k] for k in
                         ["n_pairs", "n_fs_match", "n_base_time_present",
                          "n_ecg_pcg_time_match", "frac_time_present", "frac_time_match"]}),
    "scale_invariance": stage4,
}
_save_json(results, "exp4_results.json")

# Pull the SmartTech column values (fall back to the paper headline when uncomputed).
if stage1 is not None:
    st = stage1["agg"]
    st_col = {
        "psnr_rescaled": st["psnr_rescaled"]["mean"], "psnr_blind": st["psnr_blind"]["mean"],
        "ssim_rescaled": st["ssim_rescaled"]["mean"], "ssim_blind": st["ssim_blind"]["mean"],
        "corr_rescaled": st["corr_rescaled"]["mean"], "corr_blind": st["corr_blind"]["mean"],
        "rpeak_rescaled": st["rpeak_rescaled"]["mean"], "rpeak_blind": st["rpeak_blind"]["mean"],
        "std_ratio": st["std_ratio"]["mean"], "n": stage1["n_records"],
    }
else:
    st_col = {"psnr_rescaled": 17.04, "psnr_blind": None, "ssim_rescaled": 0.742,
              "ssim_blind": None, "corr_rescaled": -0.003, "corr_blind": None,
              "rpeak_rescaled": 1.400, "rpeak_blind": None, "std_ratio": 0.28, "n": 310}

eph = EPH_COL
n_st = st_col["n"]
ratio_txt = _fmt(st_col["std_ratio"], 2)

latex = r"""% ---------------------------------------------------------------------------
% exp4 (R1.8): blind cross-dataset fidelity.
% Paste into the manuscript near the cross-dataset paragraph and cite as
% Table~\ref{tab:blind_crossdataset}. Replace the prior "amplitude transfers"
% sentence with the honest reading in the caption.
% ---------------------------------------------------------------------------
\begin{table}[!t]
\centering
\caption{Cross-dataset signal fidelity on SensSmart-Tech (%N records), evaluated
blind and after target-standard-deviation rescaling. The rescaled PSNR uses the
ground-truth standard deviation, so it overstates transfer. The blind PSNR is the
honest number. SSIM and Pearson $r$ are scale-invariant and therefore identical in
both columns, so the correlation collapse reflects temporal misalignment, not
amplitude. The mean std ratio is %RATIO.}
\label{tab:blind_crossdataset}
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & EPHNOGRAM & SmartTech (rescaled) & SmartTech (blind) \\
\midrule
PSNR (dB)        & %EP_PSNR & %ST_PSNR_R & %ST_PSNR_B \\
SSIM             & %EP_SSIM & %ST_SSIM_R & %ST_SSIM_B \\
Pearson $r$      & %EP_CORR & %ST_CORR_R & %ST_CORR_B \\
R-peak amp. err. & %EP_RPK  & %ST_RPK_R  & %ST_RPK_B \\
\bottomrule
\end{tabular}
\end{table}
"""

repl = {
    "%N": str(n_st), "%RATIO": ratio_txt,
    "%EP_PSNR": _fmt(eph["psnr"], 2), "%EP_SSIM": _fmt(eph["ssim"], 3),
    "%EP_CORR": _fmt(eph["correlation"], 3), "%EP_RPK": _fmt(eph["rpeak_amp_err"], 3),
    "%ST_PSNR_R": _fmt(st_col["psnr_rescaled"], 2), "%ST_PSNR_B": _fmt(st_col["psnr_blind"], 2),
    "%ST_SSIM_R": _fmt(st_col["ssim_rescaled"], 3), "%ST_SSIM_B": _fmt(st_col["ssim_blind"], 3),
    "%ST_CORR_R": _fmt(st_col["corr_rescaled"], 3), "%ST_CORR_B": _fmt(st_col["corr_blind"], 3),
    "%ST_RPK_R": _fmt(st_col["rpeak_rescaled"], 3), "%ST_RPK_B": _fmt(st_col["rpeak_blind"], 3),
}
for k, v in repl.items():
    latex = latex.replace(k, v)

tex_path = os.path.join(OUT, "table_blind_crossdataset.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"  saved  {tex_path}\n")
print(latex)

# %% [markdown]
# ## IEEE figure - Welch PSD of SmartTech vs EPHNOGRAM PCG
#
# One axis, two averaged spectra. Different sensor front ends give different PCG
# frequency responses; that device-response mismatch is the physical cause of the
# correlation collapse. Axis labels are bold and carry units.

# %%
def _avg_welch(pcg_segments, fs=config_paths.FS, nperseg=8192):
    acc, cnt = None, 0
    for seg in pcg_segments:
        f, pxx = ss.welch(np.asarray(seg, dtype=np.float64), fs=fs, nperseg=nperseg)
        acc = pxx if acc is None else acc + pxx
        cnt += 1
    if cnt == 0:
        return None, None
    return f, acc / cnt


# Collect unfiltered, z-scored PCG from each dataset (unfiltered so the native
# device response is visible; both training pipelines used no digital bandpass).
smart_pcg = []
for base in list_wfdb_records()[:MAX_PSD_SEGMENTS]:
    ecg_raw, pcg_raw, fs_in = read_wfdb_pair(base)
    if pcg_raw is None:
        continue
    _, p = preprocess_pair(ecg_raw, pcg_raw, fs_in, apply_bandpass=False)
    if p is not None:
        smart_pcg.append(p)

eph_pcg = []
if os.path.isdir(config_paths.EPH_CHUNKS_DIR):
    ef = sorted(f for f in os.listdir(config_paths.EPH_CHUNKS_DIR)
                if f.endswith(".npz") and "split" not in f)[:MAX_PSD_SEGMENTS]
    for fn in ef:
        try:
            d = np.load(os.path.join(config_paths.EPH_CHUNKS_DIR, fn), allow_pickle=True)
        except Exception:
            continue
        if "pcg" in d:
            eph_pcg.append(metrics.normalize_zscore(d["pcg"])[:config_paths.SEG_LEN])

f_s, p_s = _avg_welch(smart_pcg) if smart_pcg else (None, None)
f_e, p_e = _avg_welch(eph_pcg) if eph_pcg else (None, None)

if p_s is not None or p_e is not None:
    fig, ax = ieee_style.new_fig("double", height=2.8)
    if p_e is not None:
        ax.plot(f_e, 10.0 * np.log10(p_e + 1e-20),
                color=ieee_style.COLORS["proposed"], linewidth=1.0,
                label=f"EPHNOGRAM PCG (n={len(eph_pcg)})")
    if p_s is not None:
        ax.plot(f_s, 10.0 * np.log10(p_s + 1e-20),
                color=ieee_style.COLORS["cnn"], linewidth=1.0,
                label=f"SensSmart-Tech PCG (n={len(smart_pcg)})")
    ax.set_xlim(0, 1000)
    ieee_style.finalize_axes(ax, xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)",
                             legend=True, legend_loc="upper right")
    ieee_style.save_fig(fig, OUT, "fig_pcg_psd_device_response")
else:
    print("Figure skipped: no PCG segments available for either dataset.")

print("\nexp4 complete. Artifacts in:", OUT)
