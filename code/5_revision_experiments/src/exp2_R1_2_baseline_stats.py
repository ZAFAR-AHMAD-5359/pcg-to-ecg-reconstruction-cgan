# %% [markdown]
# # exp2 - Per-segment baseline metrics + paired statistics (Reviewer 1, comment 2)
#
# **Goal.** Reviewer 1 asks for a statistically defensible comparison of the
# Proposed model against the three paper baselines, and for honest uncertainty
# on the headline correlation given that the EPHNOGRAM test set is only three
# continuous single-session recordings (effective N ~ 3).
#
# This script:
# 1. loads all four paper models (CNN, U-Net, WGAN-GP, Proposed);
# 2. runs each over every one of the 453 EPHNOGRAM test segments and computes
#    per-segment PSNR, Pearson correlation, SSIM, RMSE, and R-peak amplitude
#    error; writes one tidy long CSV (one row per segment x model);
# 3. runs paired Wilcoxon signed-rank tests (Proposed vs each baseline, per
#    metric) with Holm-Bonferroni correction across the whole family;
# 4. computes a subject-clustered bootstrap 95% CI for the Proposed correlation
#    (resampling the three recordings) alongside the naive segment-level CI;
# 5. writes per-subject and per-scenario summary tables;
# 6. emits JSON, paste-ready LaTeX tables with significance markers, and two
#    IEEE double-column figures.
#
# **Pipeline note (must match the paper).** Every ECG target is polarity-fixed
# then z-scored; the PCG input is z-scored; all fidelity metrics are computed on
# the z-scored ECG versus the raw generator output. No target-std rescaling is
# applied here (that rescaling is the separate R1.8 confound). This exact
# pipeline reproduces the verified per-subject numbers (e.g. S023 R-peak
# amplitude error 0.067).
#
# All paths come from `config_paths`. Outputs land in
# `config_paths.out_dir('exp2_baseline_stats')`. The script is resumable: the
# per-segment CSV is cached and reused, and the inference loop skips segments
# already present in the cache.

# %%
# ---------------------------------------------------------------------------
# Cell 1 - Environment setup (Colab)
# ---------------------------------------------------------------------------
# On Colab, mount Drive and put the shared/ modules on the path. You MUST copy
# the local shared/ folder to Drive first (or paste the four modules into a cell):
#     E:\...\Revision\Colab_Experiments\shared\  ->  MyDrive/JBHI_Revision/shared/
#
# If you are running this OUTSIDE Colab, comment out the drive.mount line and set
# SHARED_DIR to the local shared/ path.
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("[note] google.colab not available; assuming a local run.")

import sys, os

SHARED_DIR = '/content/drive/MyDrive/JBHI_Revision/shared'
if not os.path.isdir(SHARED_DIR):
    # Local fallback: shared/ sits next to this script.
    _local = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'shared')
    SHARED_DIR = _local if os.path.isdir(_local) else SHARED_DIR
sys.path.append(SHARED_DIR)
print(f"[note] importing shared modules from: {SHARED_DIR}")
print("[note] If import fails, copy the local shared/ folder to Drive OR paste "
      "config_paths.py, models.py, metrics.py, ieee_style.py into a cell.")

if IN_COLAB:
    # scikit-image supplies SSIM; the rest ship with Colab.
    os.system('pip install -q scikit-image')

import config_paths
import models as M
import metrics as MET
import ieee_style as IE

# %%
# ---------------------------------------------------------------------------
# Cell 2 - Config, constants, output folder, reproducibility
# ---------------------------------------------------------------------------
import json
import glob
import numpy as np
import pandas as pd
import torch

DEVICE = config_paths.DEVICE
OUT = config_paths.out_dir('exp2_baseline_stats')
print(f"[note] device: {DEVICE}")
print(f"[note] output folder: {OUT}")

# Set True to force a full recompute of the per-segment metrics (ignore cache).
FORCE_RECOMPUTE = False

# Bootstrap settings (fixed seed => reproducible CIs).
N_BOOT = 10000
RNG = np.random.default_rng(2026)

# Metric registry: key, display label (with unit), direction ('max' better or
# 'min' better), and print precision. RMSE and R-peak amplitude error are on the
# z-scored ECG, so their unit is arbitrary units (a.u.).
METRICS = [
    ("psnr",          "PSNR (dB)",              "max", 2),
    ("corr",          "Corr. r",                "max", 3),
    ("ssim",          "SSIM",                   "max", 3),
    ("rmse",          "RMSE (a.u.)",            "min", 3),
    ("rpeak_amp_err", "R-peak amp. err. (a.u.)","min", 3),
]
METRIC_KEYS = [m[0] for m in METRICS]

MODEL_ORDER = ["CNN", "UNet", "WGAN-GP", "Proposed"]
BASELINES = ["CNN", "UNet", "WGAN-GP"]

# Colour + pretty-name maps (colour-blind-safe roles from ieee_style).
MODEL_COLOR = {
    "CNN":      IE.COLORS["cnn"],
    "UNet":     IE.COLORS["unet"],
    "WGAN-GP":  IE.COLORS["wgan"],
    "Proposed": IE.COLORS["proposed"],
}
MODEL_PRETTY = {"CNN": "CNN", "UNet": "U-Net", "WGAN-GP": "WGAN-GP", "Proposed": "Proposed"}
SCEN_NAME = {0: "rest", 1: "exercise", 2: "recovery"}

# %%
# ---------------------------------------------------------------------------
# Cell 3 - Load the four paper models (guarded / resumable)
# ---------------------------------------------------------------------------
# Preferred path: models.build_paper_models loads all four at once. If any
# checkpoint is missing we fall back to loading each model individually so a
# single missing baseline does not abort the whole run.
def load_models_guarded():
    loaded = {}
    try:
        loaded = M.build_paper_models(config_paths.CKPT, device=DEVICE)
        print("[ok] build_paper_models loaded all four checkpoints.")
        return loaded
    except Exception as e:
        print(f"[warn] build_paper_models failed ({e}); loading models individually.")

    specs = [
        ("CNN",      M.CNNEncoderDecoder,      config_paths.CKPT["CNN"]),
        ("UNet",     M.UNetGenerator,          config_paths.CKPT["UNet"]),
        ("WGAN-GP",  M.UNetGenerator,          config_paths.CKPT["WGAN-GP"]),
        ("Proposed", M.ImprovedGeneratorNoTanh, config_paths.CKPT["Proposed"]),
    ]
    for name, cls, path in specs:
        if not os.path.exists(path):
            print(f"[warn] checkpoint missing for {name}: {path} -> skipping model.")
            continue
        try:
            mdl, missing = M.load_model(cls(), path, device=DEVICE)
            loaded[name] = mdl
            print(f"[ok] loaded {name} from {path}")
        except Exception as ee:
            print(f"[warn] could not load {name}: {ee}")
    return loaded

MODELS = load_models_guarded()
AVAILABLE = [m for m in MODEL_ORDER if m in MODELS]
if not AVAILABLE:
    raise SystemExit("[fatal] no models could be loaded; check config_paths.CKPT.")
print(f"[note] models available: {AVAILABLE}")

# Report verified parameter counts for the loaded models (feeds R2.2/R2.5).
for name in AVAILABLE:
    print(f"       params[{name}] = {M.count_params(MODELS[name]):,}")

# %%
# ---------------------------------------------------------------------------
# Cell 4 - Build the EPHNOGRAM test-segment index
# ---------------------------------------------------------------------------
# Each chunk .npz holds ONE segment (ecg, pcg = 64000 samples, plus subject_id
# and scenario). The test set is every chunk whose subject_id is a test subject.
def build_test_index():
    if not os.path.exists(config_paths.SPLIT_FILE):
        print(f"[warn] split file missing: {config_paths.SPLIT_FILE}")
        return []
    with open(config_paths.SPLIT_FILE, "r") as f:
        splits = json.load(f)
    test_subjects = set(splits["test_subjects"])
    print(f"[note] test subjects from split: {sorted(test_subjects)}")

    all_chunks = sorted(glob.glob(os.path.join(config_paths.EPH_CHUNKS_DIR, "*.npz")))
    index = []
    for path in all_chunks:
        try:
            data = np.load(path, allow_pickle=True)
            sid = str(data["subject_id"])
            if sid in test_subjects:
                scen = config_paths.scenario_to_index(str(data["scenario"]))
                index.append({"chunk": os.path.basename(path), "path": path,
                              "subject_id": sid, "scenario": scen})
        except Exception as e:
            print(f"[warn] could not read {path}: {e}")
    return index

TEST_INDEX = build_test_index()
print(f"[note] test segments found: {len(TEST_INDEX)} (expected 453)")
if TEST_INDEX:
    _sc = pd.Series([r["subject_id"] for r in TEST_INDEX]).value_counts()
    print("[note] segments per subject:\n" + _sc.to_string())

# %%
# ---------------------------------------------------------------------------
# Cell 5 - Per-segment metrics for every model (resumable) -> tidy CSV
# ---------------------------------------------------------------------------
PERSEG_CSV = os.path.join(OUT, "exp2_persegment_metrics.csv")

def preprocess_pair(ecg_raw, pcg_raw):
    """Paper pipeline: polarity-fix ECG, then z-score ECG and PCG."""
    ecg = MET.normalize_zscore(MET.ensure_positive_polarity(ecg_raw.astype(np.float32)))
    pcg = MET.normalize_zscore(pcg_raw.astype(np.float32))
    return ecg, pcg

def segment_metrics(real_z, fake):
    """All five fidelity metrics on the z-scored ECG vs the generator output."""
    out = {
        "psnr": MET.calc_psnr(real_z, fake),
        "corr": MET.calc_correlation(real_z, fake),
        "rmse": MET.calc_rmse(real_z, fake),
        "rpeak_amp_err": MET.calc_r_peak_amplitude_error(
            real_z, fake, fs=config_paths.FS,
            distance=config_paths.RPEAK_DISTANCE,
            prominence=config_paths.RPEAK_PROMINENCE),
    }
    try:
        out["ssim"] = MET.calc_ssim(real_z, fake)
    except Exception:
        out["ssim"] = np.nan
    return out

def compute_persegment(index, models_dict):
    # Resume: reuse rows already cached for (chunk, model) pairs.
    done = set()
    cached_rows = []
    if os.path.exists(PERSEG_CSV) and not FORCE_RECOMPUTE:
        prev = pd.read_csv(PERSEG_CSV)
        cached_rows = prev.to_dict("records")
        done = {(r["chunk"], r["model"]) for r in cached_rows}
        print(f"[note] resuming: {len(done)} (chunk,model) rows already cached.")

    rows = list(cached_rows)
    n_new = 0
    for i, rec in enumerate(index):
        needed = [m for m in models_dict if (rec["chunk"], m) not in done]
        if not needed:
            continue
        try:
            data = np.load(rec["path"], allow_pickle=True)
            real_z, pcg_z = preprocess_pair(data["ecg"], data["pcg"])
        except Exception as e:
            print(f"[warn] skip {rec['chunk']}: {e}")
            continue

        pcg_t = torch.from_numpy(pcg_z).unsqueeze(0).unsqueeze(0).to(DEVICE)
        onehot = np.zeros(config_paths.NUM_SCENARIOS, dtype=np.float32)
        onehot[rec["scenario"]] = 1.0
        scen_t = torch.from_numpy(onehot).unsqueeze(0).to(DEVICE)

        for name in needed:
            is_prop = (name == "Proposed")
            try:
                fake = M.run_generator(models_dict[name], pcg_t, scen_t, is_prop)
                fake = np.asarray(fake).ravel()[:real_z.size]
                mvals = segment_metrics(real_z, fake)
            except Exception as e:
                print(f"[warn] inference failed {name} on {rec['chunk']}: {e}")
                mvals = {k: np.nan for k in METRIC_KEYS}
            rows.append({"chunk": rec["chunk"], "subject_id": rec["subject_id"],
                         "scenario": rec["scenario"],
                         "scenario_name": SCEN_NAME.get(rec["scenario"], "other"),
                         "model": name, **mvals})
            n_new += 1

        # Periodic checkpoint so a Colab disconnect never loses much work.
        if (i + 1) % 40 == 0:
            pd.DataFrame(rows).to_csv(PERSEG_CSV, index=False)
            print(f"  ...processed {i + 1}/{len(index)} segments "
                  f"({n_new} new rows) - checkpoint saved.")

    df = pd.DataFrame(rows)
    df.to_csv(PERSEG_CSV, index=False)
    print(f"[ok] per-segment metrics written: {PERSEG_CSV} "
          f"({len(df)} rows, {n_new} new)")
    return df

DF = compute_persegment(TEST_INDEX, {k: MODELS[k] for k in AVAILABLE})

# Keep only available models, in canonical order, and coerce numeric types.
DF = DF[DF["model"].isin(AVAILABLE)].copy()
for k in METRIC_KEYS:
    DF[k] = pd.to_numeric(DF[k], errors="coerce")

# Sanity check against the verified pooled Proposed numbers (17.10 dB, 0.610).
if "Proposed" in AVAILABLE:
    dp = DF[DF["model"] == "Proposed"]
    print(f"[check] Proposed pooled: PSNR={dp['psnr'].mean():.2f} dB, "
          f"corr={dp['corr'].mean():.3f}, "
          f"R-peak-amp-err valid N={dp['rpeak_amp_err'].notna().sum()}/{len(dp)}")

# %%
# ---------------------------------------------------------------------------
# Cell 6 - Paired Wilcoxon signed-rank + Holm-Bonferroni
# ---------------------------------------------------------------------------
from scipy.stats import wilcoxon

def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj

def star(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""

# Wide pivot per metric: index=chunk, columns=model. This pairs by segment.
def wide(metric):
    return DF.pivot_table(index="chunk", columns="model", values=metric)

paired = []   # one entry per (metric, baseline) comparison
if "Proposed" in AVAILABLE:
    for mkey, mlabel, direction, _ in METRICS:
        W = wide(mkey)
        if "Proposed" not in W.columns:
            continue
        for base in [b for b in BASELINES if b in W.columns]:
            sub = W[["Proposed", base]].dropna()
            n = len(sub)
            stat = pval = np.nan
            med_diff = np.nan
            prop_better = None
            if n >= 1:
                dvec = (sub["Proposed"] - sub[base]).values
                med_diff = float(np.median(dvec))
                if direction == "max":
                    prop_better = med_diff > 0
                else:
                    prop_better = med_diff < 0
                if np.any(dvec != 0):
                    try:
                        stat, pval = wilcoxon(sub["Proposed"].values, sub[base].values,
                                              zero_method="wilcox", alternative="two-sided")
                        stat, pval = float(stat), float(pval)
                    except Exception as e:
                        print(f"[warn] wilcoxon {mkey} vs {base}: {e}")
            paired.append({"metric": mkey, "metric_label": mlabel, "baseline": base,
                           "n_pairs": int(n), "statistic": stat, "p_raw": pval,
                           "median_diff_proposed_minus_base": med_diff,
                           "proposed_better": prop_better})

    # Holm-Bonferroni across the whole family of valid comparisons.
    valid = [i for i, r in enumerate(paired) if np.isfinite(r["p_raw"])]
    if valid:
        adj = holm_bonferroni([paired[i]["p_raw"] for i in valid])
        for j, i in enumerate(valid):
            paired[i]["p_holm"] = float(adj[j])
    for r in paired:
        r.setdefault("p_holm", np.nan)
        # Marker: stars if Proposed significantly better; dagger if significantly worse.
        sig = np.isfinite(r["p_holm"]) and r["p_holm"] < 0.05
        if sig and r["proposed_better"]:
            r["marker"] = star(r["p_holm"])
        elif sig and r["proposed_better"] is False:
            r["marker"] = "\\dagger"
        else:
            r["marker"] = ""

PAIRED_DF = pd.DataFrame(paired)
PAIRED_CSV = os.path.join(OUT, "exp2_paired_wilcoxon.csv")
PAIRED_DF.to_csv(PAIRED_CSV, index=False)
print(f"[ok] paired tests written: {PAIRED_CSV}")
if not PAIRED_DF.empty:
    print(PAIRED_DF[["metric", "baseline", "n_pairs", "statistic",
                     "p_raw", "p_holm", "marker"]].to_string(index=False))

# %%
# ---------------------------------------------------------------------------
# Cell 7 - Bootstrap 95% CIs for the Proposed correlation
# ---------------------------------------------------------------------------
# Subject-clustered (block) bootstrap resamples the THREE recordings with
# replacement and recomputes the pooled segment-mean correlation. This honestly
# reflects the effective N ~ 3 and is far wider than the naive per-segment CI.
def bootstrap_cis():
    if "Proposed" not in AVAILABLE:
        return {}
    dp = DF[DF["model"] == "Proposed"].dropna(subset=["corr"])
    vals = dp["corr"].values
    point = float(np.mean(vals))

    # Naive segment-level bootstrap (treats 453 segments as independent).
    naive = np.array([np.mean(RNG.choice(vals, size=vals.size, replace=True))
                      for _ in range(N_BOOT)])
    naive_ci = [float(np.percentile(naive, 2.5)), float(np.percentile(naive, 97.5))]

    # Naive normal-approx CI for reference.
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    normal_ci = [point - 1.96 * sem, point + 1.96 * sem]

    # Subject-clustered bootstrap.
    subjects = sorted(dp["subject_id"].unique())
    subj_vals = {s: dp[dp["subject_id"] == s]["corr"].values for s in subjects}
    cluster_means = []
    for _ in range(N_BOOT):
        chosen = RNG.choice(subjects, size=len(subjects), replace=True)
        pooled = np.concatenate([subj_vals[s] for s in chosen])
        cluster_means.append(np.mean(pooled))
    cluster_means = np.array(cluster_means)
    cluster_ci = [float(np.percentile(cluster_means, 2.5)),
                  float(np.percentile(cluster_means, 97.5))]

    per_subject = {s: {"n": int(subj_vals[s].size),
                       "corr_mean": float(np.mean(subj_vals[s]))} for s in subjects}

    out = {
        "point_estimate_pooled_corr": point,
        "n_segments": int(vals.size),
        "n_subjects": len(subjects),
        "naive_segment_bootstrap_ci95": naive_ci,
        "naive_normal_approx_ci95": normal_ci,
        "subject_clustered_bootstrap_ci95": cluster_ci,
        "n_boot": N_BOOT,
        "per_subject": per_subject,
    }
    print(f"[boot] Proposed pooled corr = {point:.3f}")
    print(f"[boot] naive segment CI95      = [{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]")
    print(f"[boot] subject-clustered CI95  = [{cluster_ci[0]:.3f}, {cluster_ci[1]:.3f}]  "
          f"(much wider: effective N ~ {len(subjects)})")
    return out

BOOT = bootstrap_cis()

# %%
# ---------------------------------------------------------------------------
# Cell 8 - Per-subject and per-scenario summary tables
# ---------------------------------------------------------------------------
def summarize(group_cols):
    recs = []
    for keys, g in DF.groupby(group_cols + ["model"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols + ["model"], keys))
        row["n_segments"] = int(g.shape[0])
        for k in METRIC_KEYS:
            v = g[k].dropna()
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{k}_validN"] = int(len(v))
        recs.append(row)
    out = pd.DataFrame(recs)
    # Stable ordering by model.
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(group_cols + ["model"]).reset_index(drop=True)

OVERALL = summarize([])            # per-model over all segments
BY_SUBJECT = summarize(["subject_id"])
BY_SCENARIO = summarize(["scenario_name"])

OVERALL.to_csv(os.path.join(OUT, "exp2_summary_overall.csv"), index=False)
BY_SUBJECT.to_csv(os.path.join(OUT, "exp2_summary_by_subject.csv"), index=False)
BY_SCENARIO.to_csv(os.path.join(OUT, "exp2_summary_by_scenario.csv"), index=False)
print("[ok] summary tables written (overall / by-subject / by-scenario).")

# One machine-readable JSON with everything a LaTeX snippet might cite.
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

RESULTS_JSON = os.path.join(OUT, "exp2_results.json")
with open(RESULTS_JSON, "w") as f:
    json.dump({
        "n_test_segments": len(TEST_INDEX),
        "models_available": AVAILABLE,
        "metrics": METRIC_KEYS,
        "overall": OVERALL.to_dict("records"),
        "by_subject": BY_SUBJECT.to_dict("records"),
        "by_scenario": BY_SCENARIO.to_dict("records"),
        "paired_wilcoxon": PAIRED_DF.to_dict("records") if not PAIRED_DF.empty else [],
        "bootstrap_correlation": BOOT,
    }, f, indent=2, default=_json_default)
print(f"[ok] results JSON written: {RESULTS_JSON}")

# %%
# ---------------------------------------------------------------------------
# Cell 9 - Paste-ready LaTeX tables
# ---------------------------------------------------------------------------
# Marker lookup: (metric, baseline) -> significance marker vs Proposed.
MARK = {}
if not PAIRED_DF.empty:
    for _, r in PAIRED_DF.iterrows():
        MARK[(r["metric"], r["baseline"])] = r["marker"]

def best_model_for(mkey, direction):
    means = {m: OVERALL.loc[OVERALL["model"] == m, f"{mkey}_mean"].values for m in AVAILABLE}
    means = {m: v[0] for m, v in means.items() if len(v) and np.isfinite(v[0])}
    if not means:
        return None
    return (max if direction == "max" else min)(means, key=means.get)

def write_main_table():
    lines = []
    lines.append("% ============================================================")
    lines.append("% exp2 main summary table - Reviewer 1, comment 2")
    lines.append("% Per-segment metrics over all 453 EPHNOGRAM test segments.")
    lines.append("% Markers: significance of Proposed vs that baseline,")
    lines.append("%   paired Wilcoxon signed-rank, Holm-Bonferroni corrected.")
    lines.append("%   * p<0.05, ** p<0.01, *** p<0.001; dagger = Proposed worse.")
    lines.append("% FIND the current Table II and REPLACE it with this block "
                 "(update the label if needed).")
    lines.append("% ============================================================")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-segment reconstruction quality on the EPHNOGRAM "
                 "test set ($N=453$ segments, three recordings). Values are "
                 "mean\\,$\\pm$\\,standard deviation. Significance markers denote "
                 "the paired Wilcoxon signed-rank comparison of the Proposed "
                 "model against each baseline, after Holm-Bonferroni correction "
                 "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$). Best value per "
                 "column is in bold.}")
    lines.append("\\label{tab:persegment_stats}")
    ncols = len(METRICS)
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "c" * ncols + "}")
    lines.append("\\toprule")
    header = "Model & " + " & ".join(m[1] for m in METRICS) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    best = {mkey: best_model_for(mkey, d) for mkey, _, d, _ in METRICS}
    for model in AVAILABLE:
        cells = [MODEL_PRETTY[model]]
        for mkey, _, _, prec in METRICS:
            row = OVERALL.loc[OVERALL["model"] == model]
            mean = row[f"{mkey}_mean"].values[0] if len(row) else np.nan
            std = row[f"{mkey}_std"].values[0] if len(row) else np.nan
            if not np.isfinite(mean):
                cells.append("--")
                continue
            txt = f"{mean:.{prec}f}$\\pm${std:.{prec}f}" if np.isfinite(std) \
                  else f"{mean:.{prec}f}"
            marker = MARK.get((mkey, model), "")
            if marker:
                txt = txt + f"$^{{{marker}}}$"
            if best.get(mkey) == model:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path = os.path.join(OUT, "exp2_summary_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] LaTeX main table: {path}")
    return path

def write_persubject_table():
    """Per-subject correlation for each model - the R1.2 headline evidence."""
    lines = []
    lines.append("% ============================================================")
    lines.append("% exp2 per-subject correlation table - Reviewer 1, comment 2")
    lines.append("% Shows the headline pooled correlation is one strong subject")
    lines.append("% (S023) averaged with one weak subject (S019); S009 has n=3.")
    lines.append("% ============================================================")
    subjects = list(BY_SUBJECT["subject_id"].unique())
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-recording Pearson correlation $r$ "
                 "(mean\\,$\\pm$\\,standard deviation) on the EPHNOGRAM test set. "
                 "The pooled correlation is dominated by subject variance: a "
                 "strong exercise recording (S023) is averaged with a weak one "
                 "(S019). The rest recording (S009) contributes only three "
                 "segments.}")
    lines.append("\\label{tab:persubject_corr}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(subjects) + "c}")
    lines.append("\\toprule")
    subj_hdr = " & ".join(f"{s}" for s in subjects)
    lines.append(f"Model & {subj_hdr} & Pooled \\\\")
    lines.append("\\midrule")
    for model in AVAILABLE:
        cells = [MODEL_PRETTY[model]]
        for s in subjects:
            row = BY_SUBJECT[(BY_SUBJECT["subject_id"] == s) & (BY_SUBJECT["model"] == model)]
            if len(row):
                mean = row["corr_mean"].values[0]
                std = row["corr_std"].values[0]
                n = int(row["n_segments"].values[0])
                cell = f"{mean:.3f}$\\pm${std:.3f}" if np.isfinite(std) else f"{mean:.3f}"
                cell = cell + f" ($n{{=}}{n}$)"
            else:
                cell = "--"
            cells.append(cell)
        prow = OVERALL[OVERALL["model"] == model]
        pooled = prow["corr_mean"].values[0] if len(prow) else np.nan
        cells.append(f"{pooled:.3f}" if np.isfinite(pooled) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path = os.path.join(OUT, "exp2_persubject_corr_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ok] LaTeX per-subject table: {path}")
    return path

def write_bootstrap_note():
    """A one-paragraph LaTeX note stating both CIs for the response letter."""
    if not BOOT:
        return None
    p = BOOT["point_estimate_pooled_corr"]
    nlo, nhi = BOOT["naive_segment_bootstrap_ci95"]
    clo, chi = BOOT["subject_clustered_bootstrap_ci95"]
    txt = (
        "% exp2 bootstrap CI note - drop into the R1.2 response.\n"
        f"The Proposed model reaches a pooled Pearson correlation of "
        f"{p:.3f} on the EPHNOGRAM test set. A naive segment-level bootstrap "
        f"gives a narrow 95\\% confidence interval of "
        f"$[{nlo:.3f},\\,{nhi:.3f}]$. That interval is optimistic, because the "
        f"453 segments come from only three continuous recordings. A "
        f"subject-clustered bootstrap that resamples the three recordings "
        f"widens the interval to $[{clo:.3f},\\,{chi:.3f}]$, which honestly "
        f"reflects the effective sample size.\n")
    path = os.path.join(OUT, "exp2_bootstrap_note.tex")
    with open(path, "w") as f:
        f.write(txt)
    print(f"[ok] LaTeX bootstrap note: {path}")
    return path

TAB_MAIN = write_main_table()
TAB_SUBJ = write_persubject_table()
NOTE_BOOT = write_bootstrap_note()

# %%
# ---------------------------------------------------------------------------
# Cell 10 - IEEE FIGURE A (double column): correlation box plots by subject
# ---------------------------------------------------------------------------
# Box plot of per-segment Pearson correlation for each model, faceted by
# recording plus a pooled panel. Boxes coloured by model; one figure-level
# legend (no overlap); bold axis label with units.
import matplotlib.patches as mpatches

def figure_boxplot_by_subject():
    IE.set_ieee_style()
    subjects = list(pd.Series([r["subject_id"] for r in TEST_INDEX]).unique())
    subjects = sorted(subjects)
    panels = subjects + ["Pooled"]
    fig, axes = IE.new_fig("double", height=2.7, nrows=1, ncols=len(panels),
                           sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "Pooled":
            sub = DF
            n_seg = DF["chunk"].nunique()
            title = f"Pooled ($n{{=}}{n_seg}$)"
        else:
            sub = DF[DF["subject_id"] == panel]
            n_seg = sub["chunk"].nunique()
            title = f"{panel} ($n{{=}}{n_seg}$)"

        data, colors, positions = [], [], []
        for i, model in enumerate(AVAILABLE):
            vals = sub[sub["model"] == model]["corr"].dropna().values
            data.append(vals if vals.size else [np.nan])
            colors.append(MODEL_COLOR[model])
            positions.append(i + 1)

        bp = ax.boxplot(data, positions=positions, widths=0.62,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.1),
                        whiskerprops=dict(linewidth=0.9),
                        capprops=dict(linewidth=0.9),
                        boxprops=dict(linewidth=0.8))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        # Overlay jittered points for the tiny-n panels so single boxes read.
        for i, model in enumerate(AVAILABLE):
            vals = sub[sub["model"] == model]["corr"].dropna().values
            if 0 < vals.size <= 12:
                jit = (RNG.random(vals.size) - 0.5) * 0.28
                ax.scatter(np.full(vals.size, i + 1) + jit, vals, s=6,
                           color="black", alpha=0.5, zorder=3, linewidths=0)

        ax.set_xticks(range(1, len(AVAILABLE) + 1))
        ax.set_xticklabels([])                       # models carried by legend
        ax.set_title(title)
        ax.axhline(0.0, color=IE.COLORS["muted"], linewidth=0.6, linestyle="--")
        IE.finalize_axes(ax)

    axes[0].set_ylabel("Pearson correlation $r$", fontweight="bold")

    handles = [mpatches.Patch(facecolor=MODEL_COLOR[m], edgecolor="0.3",
                              label=MODEL_PRETTY[m]) for m in AVAILABLE]
    try:
        fig.legend(handles=handles, loc="outside lower center",
                   ncol=len(AVAILABLE), frameon=True)
    except Exception:
        fig.legend(handles=handles, loc="lower center", ncol=len(AVAILABLE),
                   bbox_to_anchor=(0.5, -0.04), frameon=True)
    IE.save_fig(fig, OUT, "exp2_fig_corr_boxplot_by_subject")

if len(DF):
    figure_boxplot_by_subject()

# %%
# ---------------------------------------------------------------------------
# Cell 11 - IEEE FIGURE B (double column): grouped bars across metrics
# ---------------------------------------------------------------------------
# One panel per metric; four coloured bars (models) with std error bars.
# Bold per-panel axis label carrying units; shared figure legend, no overlap.
def figure_grouped_bars():
    IE.set_ieee_style()
    fig, axes = IE.new_fig("double", height=2.6, nrows=1, ncols=len(METRICS))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (mkey, mlabel, direction, prec) in zip(axes, METRICS):
        means, stds, colors = [], [], []
        for model in AVAILABLE:
            row = OVERALL[OVERALL["model"] == model]
            means.append(row[f"{mkey}_mean"].values[0] if len(row) else np.nan)
            stds.append(row[f"{mkey}_std"].values[0] if len(row) else np.nan)
            colors.append(MODEL_COLOR[model])
        x = np.arange(len(AVAILABLE))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.9,
               edgecolor="0.25", linewidth=0.6,
               error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, ecolor="0.3"))
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel(mlabel, fontweight="bold")
        arrow = r"$\uparrow$" if direction == "max" else r"$\downarrow$"
        ax.set_title(arrow, fontweight="bold")
        IE.finalize_axes(ax)

    handles = [mpatches.Patch(facecolor=MODEL_COLOR[m], edgecolor="0.3",
                              label=MODEL_PRETTY[m]) for m in AVAILABLE]
    try:
        fig.legend(handles=handles, loc="outside lower center",
                   ncol=len(AVAILABLE), frameon=True)
    except Exception:
        fig.legend(handles=handles, loc="lower center", ncol=len(AVAILABLE),
                   bbox_to_anchor=(0.5, -0.04), frameon=True)
    IE.save_fig(fig, OUT, "exp2_fig_metric_bars")

if len(OVERALL):
    figure_grouped_bars()

# %%
# ---------------------------------------------------------------------------
# Cell 12 - Manifest
# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print("exp2 COMPLETE - artifacts in:", OUT)
for fn in sorted(os.listdir(OUT)):
    print("   ", fn)
print("=" * 64)
