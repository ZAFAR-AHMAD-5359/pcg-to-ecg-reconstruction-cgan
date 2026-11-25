# %% [markdown]
# # exp5 - Computational Complexity + Inference Latency (Reviewer 2, comment R2.2)
#
# Reviewer 2 asked us to report the algorithm's computational complexity and to
# justify the deployment cost. This script measures, for every paper model:
#
#   * trainable parameter count (asserted against the VERIFIED local counts),
#   * MACs / FLOPs per single 8 s (64,000-sample) forward pass (thop -> ptflops -> fvcore),
#   * inference latency on CPU and (if present) GPU: >=10 warmup + >=50 timed passes,
#     reported as mean +/- SD with a real-time factor (RTF = 8 s / latency),
#   * weight memory (MB) and GPU peak activation memory (MB).
#
# It emits a JSON of raw numbers plus a paste-ready IEEE LaTeX complexity table,
# and one IEEE double-column figure. Only the generator runs at inference; the
# discriminator is discarded after training and is reported for context only.
#
# VERIFIED expected parameter counts (asserted, warn-on-mismatch):
#   CNN 9,776,385 | U-Net 13,609,345 | WGAN-GP 13,609,345 | Proposed 19,315,715 | Disc 2,176,833
# The submitted paper's "14.0M" for the Proposed generator is an error; the true
# generator has 19.3M parameters.

# %%
# --- Standard shared-module header (see SHARED GROUND TRUTH) ----------------
# When run as a stand-alone .py in Colab you must FIRST copy the local
# `shared/` folder (config_paths.py, models.py, metrics.py, ieee_style.py) to
#   /content/drive/MyDrive/JBHI_Revision/shared
# OR paste those four modules next to this script. build_notebooks.py writes
# them into /content automatically, so a built notebook needs no upload.
from google.colab import drive
drive.mount('/content/drive')

import sys
for _p in ('/content', '/content/drive/MyDrive/JBHI_Revision/shared'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib
import config_paths
import models
import metrics          # noqa: F401  (imported for pipeline parity / availability check)
import ieee_style
for _m in (config_paths, models, ieee_style):
    importlib.reload(_m)
ieee_style.set_ieee_style()
print('shared modules ready | device =', config_paths.DEVICE)

# %%
# --- Install a FLOPs/MACs profiler if none is present -----------------------
# Order of preference: thop -> ptflops -> fvcore. We install thop (the smallest)
# and additionally try fvcore, which traces matmuls and therefore also counts
# the bottleneck self-attention that thop/ptflops hooks miss.
import subprocess

def _ensure(pkg):
    try:
        __import__(pkg)
        print(f'  {pkg}: already available')
        return True
    except Exception:
        print(f'  {pkg}: installing ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
        try:
            __import__(pkg)
            print(f'  {pkg}: installed')
            return True
        except Exception:
            print(f'  {pkg}: NOT available (continuing)')
            return False

_HAS_THOP    = _ensure('thop')
_HAS_FVCORE  = _ensure('fvcore')
try:
    import ptflops          # noqa: F401
    _HAS_PTFLOPS = True
except Exception:
    _HAS_PTFLOPS = False
print('profilers  | thop:', _HAS_THOP, '| ptflops:', _HAS_PTFLOPS, '| fvcore:', _HAS_FVCORE)

# %%
# --- Imports + run configuration --------------------------------------------
import os
import gc
import json
import time
import copy
import warnings

import numpy as np
import torch

OUT_DIR   = config_paths.out_dir('exp5_complexity')
SEG_LEN   = config_paths.SEG_LEN          # 64000 samples
FS        = config_paths.FS               # 8000 Hz
SIG_SEC   = SEG_LEN / FS                  # 8.0 s of signal per segment
WARMUP    = 10                            # >= 10 warmup passes (spec)
ITERS     = 50                            # >= 50 timed passes (spec)

CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda') if torch.cuda.is_available() else None

print(f'output dir : {OUT_DIR}')
print(f'segment    : {SEG_LEN} samples = {SIG_SEC:.1f} s @ {FS} Hz')
print(f'benchmark  : {WARMUP} warmup + {ITERS} timed passes per device')
print(f'GPU        : {"available (" + torch.cuda.get_device_name(0) + ")" if GPU_DEVICE else "NOT available; CPU only"}')

# %%
# --- Model registry ---------------------------------------------------------
# Each entry: constructor, checkpoint key, forward signature, expected params,
# role. `is_proposed` -> forward(pcg, scenario); baselines -> forward(pcg);
# the discriminator -> forward(pcg, ecg, scenario) and is training-only.
EXPECTED = {
    'CNN':      9_776_385,
    'U-Net':    13_609_345,
    'WGAN-GP':  13_609_345,
    'Proposed': 19_315_715,
    'Discriminator': 2_176_833,
}

def _build_registry():
    reg = [
        dict(name='CNN',      ctor=models.CNNEncoderDecoder,      ckpt='CNN',
             sig='pcg',        role='generator'),
        dict(name='U-Net',    ctor=models.UNetGenerator,          ckpt='UNet',
             sig='pcg',        role='generator'),
        dict(name='WGAN-GP',  ctor=models.UNetGenerator,          ckpt='WGAN-GP',
             sig='pcg',        role='generator'),
        dict(name='Proposed', ctor=models.ImprovedGeneratorNoTanh, ckpt='Proposed',
             sig='pcg+scen',   role='generator'),
        dict(name='Discriminator', ctor=models.SpectralNormDiscriminator, ckpt=None,
             sig='disc',       role='discriminator'),
    ]
    return reg

REGISTRY   = _build_registry()
GEN_ORDER  = ['CNN', 'U-Net', 'WGAN-GP', 'Proposed']   # inference-cost table order


def make_inputs(sig, device):
    """Return the positional-argument tuple this forward signature expects."""
    pcg  = torch.randn(1, 1, SEG_LEN, device=device)
    scen = torch.tensor([config_paths.SCEN_EXERCISE], dtype=torch.float32, device=device)
    if sig == 'pcg':
        return (pcg,)
    if sig == 'pcg+scen':
        return (pcg, scen)
    if sig == 'disc':
        ecg = torch.randn(1, 1, SEG_LEN, device=device)
        return (pcg, ecg, scen)
    raise ValueError(f'unknown signature {sig}')


def build_model(entry, device):
    """Instantiate and (best-effort) load the checkpoint. Complexity and latency
    depend only on the architecture, so a missing checkpoint is non-fatal:
    the fresh-initialised model is used and a note is printed."""
    model = entry['ctor']()
    loaded = False
    ckpt_key = entry['ckpt']
    if ckpt_key is not None:
        path = config_paths.CKPT.get(ckpt_key)
        if path and os.path.exists(path):
            try:
                model, missing = models.load_model(model, path, device=str(device))
                loaded = True
                print(f"  [{entry['name']}] loaded checkpoint ({os.path.basename(path)})")
            except Exception as e:
                print(f"  [{entry['name']}] checkpoint load FAILED ({e}); using fresh weights")
        else:
            print(f"  [{entry['name']}] checkpoint not found; using fresh weights (counts unaffected)")
    model.to(device).eval()
    return model, loaded

# %%
# --- (1) Parameter counts, asserted against the VERIFIED numbers ------------
print('\n=== Parameter counts ===')
param_results = {}
for entry in REGISTRY:
    name = entry['name']
    model, loaded = build_model(entry, CPU_DEVICE)
    n = models.count_params(model)
    exp = EXPECTED[name]
    ok = (n == exp)
    flag = 'OK' if ok else 'MISMATCH'
    if not ok:
        warnings.warn(f'{name}: param count {n:,} != expected {exp:,}')
    print(f'  {name:<14} params = {n:>12,}   expected = {exp:>12,}   [{flag}]')
    param_results[name] = dict(
        params=int(n), expected=int(exp), matches_expected=bool(ok),
        weights_mb=float(n * 4 / 1e6),      # float32 weights, decimal MB
        checkpoint_loaded=bool(loaded), role=entry['role'],
    )
    del model
    gc.collect()

# %%
# --- (2) MACs / FLOPs per 8 s forward pass ----------------------------------
# thop and ptflops register hooks on known layer types and therefore miss the
# raw matmuls inside the multi-head self-attention bottleneck. fvcore traces the
# graph and does count them, so where fvcore is available we report it too and
# treat it as the more complete estimate for the attention-bearing Proposed model.

def macs_thop(model, inputs):
    from thop import profile as thop_profile
    m = copy.deepcopy(model)                 # thop injects buffers; keep original clean
    macs, _ = thop_profile(m, inputs=inputs, verbose=False)
    del m
    return float(macs)

def macs_ptflops(model, inputs, sig):
    from ptflops import get_model_complexity_info
    m = copy.deepcopy(model)
    # ptflops feeds a single constructed input; map our positional args to kwargs.
    if sig == 'pcg':
        kw = dict(x=inputs[0])
    elif sig == 'pcg+scen':
        kw = dict(x=inputs[0], scenario_onehot=inputs[1])
    elif sig == 'disc':
        kw = dict(pcg=inputs[0], ecg=inputs[1], scenario_onehot=inputs[2])
    else:
        return float('nan')
    def _ctor(_res):
        return {k: v.clone() for k, v in kw.items()}
    macs, _ = get_model_complexity_info(
        m, (1, SEG_LEN), input_constructor=_ctor,
        as_strings=False, print_per_layer_stat=False, verbose=False)
    del m
    return float(macs)

def macs_fvcore(model, inputs):
    from fvcore.nn import FlopCountAnalysis
    m = copy.deepcopy(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fca = FlopCountAnalysis(m, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        val = float(fca.total())            # fvcore counts one MAC as one "flop"
    del m
    return val

print('\n=== MACs / FLOPs per 8 s forward pass ===')
mac_results = {}
for entry in REGISTRY:
    name = entry['name']
    model, _ = build_model(entry, CPU_DEVICE)
    inputs = make_inputs(entry['sig'], CPU_DEVICE)
    per_backend = {}
    if _HAS_THOP:
        try:
            per_backend['thop'] = macs_thop(model, inputs)
        except Exception as e:
            print(f'    [{name}] thop failed: {e}')
    if _HAS_PTFLOPS:
        try:
            per_backend['ptflops'] = macs_ptflops(model, inputs, entry['sig'])
        except Exception as e:
            print(f'    [{name}] ptflops failed: {e}')
    if _HAS_FVCORE:
        try:
            per_backend['fvcore'] = macs_fvcore(model, inputs)
        except Exception as e:
            print(f'    [{name}] fvcore failed: {e}')

    # Headline MACs: prefer fvcore (counts attention) for attention models,
    # else thop, else ptflops, else NaN.
    headline_backend = None
    for cand in (('fvcore', 'thop', 'ptflops') if name == 'Proposed'
                 else ('thop', 'ptflops', 'fvcore')):
        if cand in per_backend and np.isfinite(per_backend[cand]):
            headline_backend = cand
            break
    macs = per_backend.get(headline_backend, float('nan')) if headline_backend else float('nan')
    gmac = macs / 1e9 if np.isfinite(macs) else float('nan')
    gflop = 2 * gmac if np.isfinite(gmac) else float('nan')
    print(f'  {name:<14} MACs = {gmac:8.2f} GMac  FLOPs = {gflop:8.2f} GFLOP  '
          f'(backend: {headline_backend})  all: '
          + ', '.join(f'{k}={v/1e9:.2f}G' for k, v in per_backend.items()))
    mac_results[name] = dict(
        macs=float(macs), gmac=float(gmac), gflop=float(gflop),
        backend=headline_backend,
        per_backend_gmac={k: float(v / 1e9) for k, v in per_backend.items()},
    )
    del model, inputs
    gc.collect()

# %%
# --- (3) Inference latency: CPU and (if present) GPU ------------------------
# Generators only. RTF = signal_seconds / latency_seconds (RTF > 1 => faster
# than real-time). Peak GPU activation memory is read from the CUDA allocator.

def benchmark(model, inputs, device, warmup=WARMUP, iters=ITERS):
    model.to(device).eval()
    inputs = tuple(t.to(device) for t in inputs)
    times_ms = []
    is_cuda = (device.type == 'cuda')
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iters):
            if is_cuda:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                _ = model(*inputs)
                ev1.record()
                torch.cuda.synchronize()
                times_ms.append(ev0.elapsed_time(ev1))
            else:
                t0 = time.perf_counter()
                _ = model(*inputs)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1e3)
    times_ms = np.asarray(times_ms, dtype=float)
    mean_ms = float(times_ms.mean())
    std_ms  = float(times_ms.std())
    rtf     = float(SIG_SEC / (mean_ms / 1e3)) if mean_ms > 0 else float('nan')
    peak_mb = float(torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float('nan')
    return dict(mean_ms=mean_ms, std_ms=std_ms, rtf=rtf, peak_mem_mb=peak_mb,
                n_timed=int(iters), n_warmup=int(warmup))

print('\n=== Inference latency (generators) ===')
lat_results = {}
for name in GEN_ORDER:
    entry = next(e for e in REGISTRY if e['name'] == name)
    rec = {}

    # CPU
    model, _ = build_model(entry, CPU_DEVICE)
    inputs = make_inputs(entry['sig'], CPU_DEVICE)
    rec['cpu'] = benchmark(model, inputs, CPU_DEVICE)
    print(f'  {name:<14} CPU  {rec["cpu"]["mean_ms"]:9.1f} +/- {rec["cpu"]["std_ms"]:6.1f} ms   '
          f'RTF {rec["cpu"]["rtf"]:6.2f}x')
    del model, inputs
    gc.collect()

    # GPU (optional)
    if GPU_DEVICE is not None:
        model, _ = build_model(entry, GPU_DEVICE)
        inputs = make_inputs(entry['sig'], GPU_DEVICE)
        rec['gpu'] = benchmark(model, inputs, GPU_DEVICE)
        print(f'  {name:<14} GPU  {rec["gpu"]["mean_ms"]:9.2f} +/- {rec["gpu"]["std_ms"]:6.2f} ms   '
              f'RTF {rec["gpu"]["rtf"]:8.1f}x   peak {rec["gpu"]["peak_mem_mb"]:7.1f} MB')
        del model, inputs
        torch.cuda.empty_cache()
        gc.collect()
    else:
        rec['gpu'] = None
    lat_results[name] = rec

# %%
# --- (4) Assemble + save the JSON of raw numbers ----------------------------
results = dict(
    meta=dict(
        experiment='exp5_complexity_latency',
        reviewer_comment='R2.2',
        segment_samples=int(SEG_LEN), fs_hz=int(FS), signal_seconds=float(SIG_SEC),
        warmup=int(WARMUP), timed_iters=int(ITERS),
        gpu_name=(torch.cuda.get_device_name(0) if GPU_DEVICE is not None else None),
        cpu_only=bool(GPU_DEVICE is None),
        torch_version=str(torch.__version__),
        profilers=dict(thop=_HAS_THOP, ptflops=_HAS_PTFLOPS, fvcore=_HAS_FVCORE),
        note=('Only the generator runs at inference; the discriminator is '
              'discarded after training. thop and ptflops do not count the '
              'bottleneck self-attention matmuls, so fvcore is preferred for '
              'the Proposed model where available.'),
    ),
    params=param_results,
    macs=mac_results,
    latency=lat_results,
)
json_path = os.path.join(OUT_DIR, 'exp5_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f'\nsaved  {json_path}')

# %%
# --- (5) Paste-ready IEEE LaTeX complexity table ----------------------------
# Values are filled from the measured numbers above. Prose in the caption obeys
# the manuscript writing rules (active voice, thin space before units, no em dash).

def _fmt(v, nd=1):
    return ('%.*f' % (nd, v)) if (v is not None and np.isfinite(v)) else '--'

def _lat_cell(rec):
    if rec is None:
        return '--'
    return r'%s\,$\pm$\,%s' % (_fmt(rec['mean_ms'], 1), _fmt(rec['std_ms'], 1))

gpu_present = GPU_DEVICE is not None
lines = []
lines.append('% ---------------------------------------------------------------------------')
lines.append('% exp5 complexity + latency table (auto-generated from exp5_results.json)')
lines.append('% Requires \\usepackage{booktabs}. Paste into Section III (Method).')
lines.append('% ---------------------------------------------------------------------------')
lines.append(r'\begin{table}[!t]')
lines.append(r'\centering')
lines.append(r'\caption{Computational complexity and inference latency per 8\,s segment '
             r'(64{,}000 samples). Only the generator runs at inference; the discriminator '
             r'is discarded after training and is listed for context. Latency is the mean '
             r'$\pm$\,SD over %d timed passes after %d warmup passes.}' % (ITERS, WARMUP))
lines.append(r'\label{tab:complexity}')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\begin{tabular}{lrrrrr}')
lines.append(r'\toprule')
lines.append(r'\textbf{Model} & \textbf{Params (M)} & \textbf{MACs (GMac)} & '
             r'\textbf{CPU (ms)} & \textbf{GPU (ms)} & \textbf{RTF} \\')
lines.append(r'\midrule')

for name in GEN_ORDER:
    p   = param_results[name]
    m   = mac_results[name]
    lat = lat_results[name]
    cpu_cell = _lat_cell(lat['cpu'])
    gpu_cell = _lat_cell(lat['gpu']) if gpu_present else '--'
    # RTF: report the deployment device (GPU when present, else CPU).
    if gpu_present and lat['gpu'] is not None:
        rtf_cell = r'%s$\times$' % _fmt(lat['gpu']['rtf'], 1)
    else:
        rtf_cell = r'%s$\times$' % _fmt(lat['cpu']['rtf'], 1)
    bold = (name == 'Proposed')
    row = r'%s & %s & %s & %s & %s & %s \\' % (
        (r'\textbf{%s}' % name) if bold else name,
        _fmt(p['params'] / 1e6, 2),
        _fmt(m['gmac'], 1),
        cpu_cell, gpu_cell, rtf_cell)
    lines.append(row)

lines.append(r'\midrule')
dp = param_results['Discriminator']
dm = mac_results['Discriminator']
lines.append(r'Discriminator$^{\dagger}$ & %s & %s & -- & -- & -- \\' % (
    _fmt(dp['params'] / 1e6, 2), _fmt(dm['gmac'], 1)))
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\vspace{2pt}')
lines.append(r'\\[-2pt]{\footnotesize $^{\dagger}$Training only; not part of the deployed model.}')
lines.append(r'\end{table}')

latex_table = '\n'.join(lines)
table_path = os.path.join(OUT_DIR, 'exp5_complexity_table.tex')
with open(table_path, 'w', encoding='utf-8') as f:
    f.write(latex_table + '\n')
print(f'saved  {table_path}\n')
print(latex_table)

# %%
# --- (6) IEEE double-column figure: parameters, MACs, and latency -----------
import matplotlib.pyplot as plt

CMAP = {
    'CNN':      ieee_style.COLORS['cnn'],
    'U-Net':    ieee_style.COLORS['unet'],
    'WGAN-GP':  ieee_style.COLORS['wgan'],
    'Proposed': ieee_style.COLORS['proposed'],
}
names   = GEN_ORDER
xpos    = np.arange(len(names))
params_m = [param_results[n]['params'] / 1e6 for n in names]
gmacs    = [mac_results[n]['gmac'] for n in names]
cpu_ms   = [lat_results[n]['cpu']['mean_ms'] for n in names]
cpu_sd   = [lat_results[n]['cpu']['std_ms'] for n in names]
if gpu_present:
    gpu_ms = [lat_results[n]['gpu']['mean_ms'] for n in names]
    gpu_sd = [lat_results[n]['gpu']['std_ms'] for n in names]

fig, axes = ieee_style.new_fig('double', height=2.7, ncols=3)
ax0, ax1, ax2 = axes

# Panel (a): parameters (M)
bars0 = ax0.bar(xpos, params_m, color=[CMAP[n] for n in names],
                width=0.68, edgecolor='black', linewidth=0.5)
ax0.set_xticks(xpos)
ax0.set_xticklabels(names, rotation=30, ha='right')
ieee_style.finalize_axes(ax0, xlabel='Model', ylabel='Parameters (M)')

# Panel (b): MACs (GMac)
ax1.bar(xpos, gmacs, color=[CMAP[n] for n in names],
        width=0.68, edgecolor='black', linewidth=0.5)
ax1.set_xticks(xpos)
ax1.set_xticklabels(names, rotation=30, ha='right')
ieee_style.finalize_axes(ax1, xlabel='Model', ylabel='MACs (GMac)')

# Panel (c): latency, CPU vs GPU (log scale, ms)
w = 0.38
ax2.bar(xpos - (w/2 if gpu_present else 0), cpu_ms, width=w, yerr=cpu_sd,
        capsize=2, color=ieee_style.COLORS['muted'], edgecolor='black',
        linewidth=0.5, label='CPU')
if gpu_present:
    ax2.bar(xpos + w/2, gpu_ms, width=w, yerr=gpu_sd, capsize=2,
            color=ieee_style.COLORS['accent'], edgecolor='black',
            linewidth=0.5, label='GPU')
ax2.set_yscale('log')
ax2.set_xticks(xpos)
ax2.set_xticklabels(names, rotation=30, ha='right')
ieee_style.finalize_axes(ax2, xlabel='Model', ylabel='Latency (ms)',
                         legend=True, legend_loc='upper left')

ieee_style.save_fig(fig, OUT_DIR, 'exp5_complexity_latency')

# %%
# --- (7) Console summary ----------------------------------------------------
print('\n=== exp5 summary ===')
print(f'{"Model":<14}{"Params(M)":>11}{"GMac":>9}{"CPU ms":>12}{"GPU ms":>14}{"RTF":>10}')
for name in GEN_ORDER:
    p = param_results[name]; m = mac_results[name]; lat = lat_results[name]
    cpu = lat['cpu']; gpu = lat['gpu']
    cpu_s = f'{cpu["mean_ms"]:.1f}+/-{cpu["std_ms"]:.1f}'
    gpu_s = (f'{gpu["mean_ms"]:.2f}+/-{gpu["std_ms"]:.2f}' if gpu else 'n/a')
    rtf   = (gpu['rtf'] if gpu else cpu['rtf'])
    print(f'{name:<14}{p["params"]/1e6:>11.2f}{m["gmac"]:>9.1f}{cpu_s:>12}{gpu_s:>14}{rtf:>9.1f}x')
d = param_results['Discriminator']
print(f'{"Disc (train)":<14}{d["params"]/1e6:>11.2f}{mac_results["Discriminator"]["gmac"]:>9.1f}'
      f'{"--":>12}{"--":>14}{"--":>10}')
print('\nOnly the generator runs at inference; the discriminator is discarded after training.')
print(f'Artifacts in: {OUT_DIR}')
print('  - exp5_results.json')
print('  - exp5_complexity_table.tex')
print('  - exp5_complexity_latency.pdf / .png')
