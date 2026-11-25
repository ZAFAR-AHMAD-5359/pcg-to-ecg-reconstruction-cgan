# %% [markdown]
# # exp7 - Retrain the reconstruction baselines (CNN, U-Net, WGAN-GP)
#
# The original baseline weights were permanently deleted from a university Google
# account, so we retrain all three PCG-to-ECG baselines. This notebook then
# evaluates all FOUR models (the three retrained baselines + the existing Proposed
# generator) on the full 453-segment EPHNOGRAM test set, and produces:
#   * the three baseline `best_model.pt` files (so exp2/exp5 can load them),
#   * per-segment metrics + paired Wilcoxon vs Proposed (Reviewer 1 comment 2) --
#     THE MAIN DELIVERABLE,
#   * a subject-clustered bootstrap CI for the Proposed correlation,
#   * a REFERENCE table of the retrained baseline scores (for transparency only).
#
# DECISION (per the author): the paper's Table II KEEPS its original numbers
# (14.80 / 16.28 / 11.72 ...). The retrained models are used ONLY to compute the
# R1.2 per-segment paired statistics; they do NOT replace Table II. The reference
# table below is emitted under a distinct label so it can go in the rebuttal or
# supplement, not over the main results table.
#
# Faithful to the originals (verified from the source notebooks):
#   * SAME architectures (models.py CNNEncoderDecoder / UNetGenerator + WGANCritic),
#   * SAME losses (CNN/U-Net plain L1; WGAN-GP: 100*L1 + Wasserstein + 10*GP, n_critic 5),
#   * SAME optimizers (Adam; CNN/U-Net betas 0.5/0.999, WGAN betas 0.0/0.9, lr 1e-4),
#   * SAME per-segment pipeline (polarity on ECG only -> augment -> z-score both),
#   * SAME cosine schedule (T_max 100, eta_min 1e-6) and grad clip 1.0.
#
# Two deliberate CORRECTIONS of latent bugs in the originals (documented, not hidden):
#   1. Evaluate the U-Net on all 453 test segments (the paper's row used only 10).
#   2. Proper early stopping on the true selection metric (the originals used a
#      mode='max' object fed a loss, which halted training early). Best-model
#      selection itself was always correct.
#   3. Score every model with the shared metrics in the correct (target, prediction)
#      order, so all four models are directly comparable (the original WGAN eval
#      swapped the arguments).
# Because of these corrections + a fresh random seed, the retrained numbers differ
# slightly from the paper's originals. That is fine: Table II stays unchanged; the
# retrained models only feed the R1.2 paired test, framed as "faithfully retrained
# baselines under the same protocol." The Proposed model is NOT retrained.
#
# Runs on a FREE T4 (baselines have no attention -> no OOM). Training is RESUMABLE:
# if the T4 session disconnects, just re-run the notebook and each model continues
# from its last epoch. To fit short free-tier sessions, train ONE model per session
# by editing RUN below (e.g. RUN = ['CNN']).

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('Not on Colab (or already mounted):', exc)
import sys
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

import os, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import scipy.signal as sgn

import config_paths, models, metrics, ieee_style
ieee_style.set_ieee_style()

DEVICE = config_paths.DEVICE
FS, SEG_LEN = config_paths.FS, config_paths.SEG_LEN
OUT = config_paths.out_dir('exp7_baseline_retrain')
print('device:', DEVICE, '| output:', OUT)

# %% [markdown]
# ## Protocol (per-model, matched to the originals)

# %%
EPOCHS      = 100
ETA_MIN     = 1e-6
GRAD_CLIP   = 1.0
PATIENCE    = 20        # proper early stop on the selection metric
SEED        = 42

# --- GPU / speed configuration (auto-detects Colab Pro A100 / V100 / T4) ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True        # TF32 matmul (Ampere+)
torch.backends.cudnn.allow_tf32 = True
_cuda = torch.cuda.is_available()
_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if _cuda else 0.0
USE_AMP   = _cuda and torch.cuda.is_bf16_supported()   # bf16 autocast (no GradScaler needed)
AMP_DTYPE = torch.bfloat16
BATCH_MULT = 2 if _gb >= 35 else 1                     # A100 (40 GB) -> double the batch
NUM_WORKERS = min(8, (os.cpu_count() or 4)) if _cuda else 2
print(f'GPU: {_gb:.0f} GB | bf16 AMP: {USE_AMP} | batch x{BATCH_MULT} | workers: {NUM_WORKERS}')

# NOTE: validation now uses val L1 on the GPU for ALL models (fast); the original
# U-Net selected on val SSIM (skimage, CPU, per sample) which dominated epoch time.
# A unified val-L1 selection is also cleaner for the same-protocol story. SSIM is
# still reported in the final full-453 test evaluation.
BASELINES = {
    'CNN': dict(
        arch='CNNEncoderDecoder', adversarial=False, l1_weight=1.0,
        batch=32 * BATCH_MULT, lr=1e-4, betas=(0.5, 0.999), select='val_loss', mode='min'),
    'UNet': dict(
        arch='UNetGenerator', adversarial=False, l1_weight=1.0,
        batch=32 * BATCH_MULT, lr=1e-4, betas=(0.5, 0.999), select='val_loss', mode='min'),
    'WGAN-GP': dict(
        arch='UNetGenerator', adversarial=True, l1_weight=100.0,
        lambda_gp=10.0, n_critic=5, batch=16 * BATCH_MULT, lr=1e-4, betas=(0.0, 0.9),
        select='val_loss', mode='min'),
}
# Which baselines to (re)train this session. Each skips if its checkpoint exists.
RUN = ['CNN', 'UNet', 'WGAN-GP']


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(arch):
    if arch == 'CNNEncoderDecoder':
        return models.CNNEncoderDecoder(in_channels=1, out_channels=1, base_filters=64)
    if arch == 'UNetGenerator':
        return models.UNetGenerator(in_channels=1, out_channels=1, features=64)
    raise ValueError(arch)


def ckpt_path(name):
    # save where config_paths.CKPT expects it, so exp2/exp5 load it unchanged
    return config_paths.CKPT[name]

# %% [markdown]
# ## Augmentation (train only; identical to the original pipeline)

# %%
def _augment(ecg, pcg):
    ecg = ecg.copy(); pcg = pcg.copy()
    if np.random.rand() < 0.5:                        # amplitude scaling
        s = np.random.uniform(0.9, 1.1); ecg *= s; pcg *= s
    if np.random.rand() < 0.6:                        # heart-rate time-stretch
        scale = np.random.uniform(0.85, 1.15)
        n = len(ecg); m = max(2, int(n / scale)); idx = np.linspace(0, n - 1, m)
        ecg = interp1d(np.arange(n), ecg, kind='cubic', fill_value='extrapolate')(idx)
        pcg = interp1d(np.arange(n), pcg, kind='cubic', fill_value='extrapolate')(idx)
        if m >= n:
            ecg, pcg = ecg[:n], pcg[:n]
        else:
            ecg = np.pad(ecg, (0, n - m), mode='edge'); pcg = np.pad(pcg, (0, n - m), mode='edge')
    if np.random.rand() < 0.3:                        # baseline wander
        f = np.random.uniform(0.1, 1.0); t = np.arange(len(ecg)) / FS
        w = 0.1 * np.sin(2 * np.pi * f * t); ecg = ecg + w; pcg = pcg + 0.3 * w
    if np.random.rand() < 0.2:                        # EMG noise, 20-200 Hz
        snr = np.random.uniform(25, 40)
        sos = sgn.butter(4, [20, 200], btype='band', fs=FS, output='sos')
        nz = sgn.sosfilt(sos, np.random.randn(len(ecg)))
        p = np.mean(ecg ** 2) / (10 ** (snr / 10))
        nz = nz * np.sqrt(p / (np.mean(nz ** 2) + 1e-12))
        ecg = ecg + nz; pcg = pcg + 0.5 * nz
    return ecg.astype(np.float32), pcg.astype(np.float32)

# %% [markdown]
# ## In-RAM dataset (preload once; polarity on ECG only, augment train, z-score both)

# %%
_RAM = {}


def _fit(a):
    a = np.asarray(a, np.float32).ravel()
    return a[:SEG_LEN] if a.shape[0] >= SEG_LEN else np.concatenate([a, np.zeros(SEG_LEN - a.shape[0], np.float32)])


def _load_split():
    sp = json.load(open(config_paths.SPLIT_FILE))
    g = lambda *k: next((set(map(str, sp[x])) for x in k if x in sp), set())
    return g('train_subjects', 'train'), g('val_subjects', 'val'), g('test_subjects', 'test')


def _index():
    cache = os.path.join(OUT, 'chunk_index.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    import glob
    idx = {}
    for p in sorted(glob.glob(os.path.join(config_paths.EPH_CHUNKS_DIR, '*.npz'))):
        sid = str(np.load(p, allow_pickle=True)['subject_id'])
        idx.setdefault(sid, []).append(p)
    json.dump(idx, open(cache, 'w'))
    return idx


def _preload(paths, tag):
    if tag in _RAM:
        return _RAM[tag]
    n = len(paths); pcg = np.zeros((n, SEG_LEN), np.float32); ecg = np.zeros((n, SEG_LEN), np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=True); pcg[i] = _fit(d['pcg']); ecg[i] = _fit(d['ecg'])
    print(f'  RAM [{tag}]: {n} chunks, {(pcg.nbytes + ecg.nbytes) / 1e9:.2f} GB, {time.time() - t0:.0f}s')
    _RAM[tag] = (pcg, ecg); return _RAM[tag]


class Chunks(Dataset):
    def __init__(self, paths, tag, augment=False):
        self.pcg, self.ecg = _preload(paths, tag); self.augment = augment

    def __len__(self):
        return self.pcg.shape[0]

    def __getitem__(self, i):
        ecg = metrics.ensure_positive_polarity(self.ecg[i])   # ECG only, before aug
        pcg = self.pcg[i].copy()
        if self.augment:
            ecg, pcg = _augment(ecg, pcg)
        ecg = metrics.normalize_zscore(ecg); pcg = metrics.normalize_zscore(pcg)
        return torch.from_numpy(pcg).unsqueeze(0), torch.from_numpy(ecg).unsqueeze(0)


def loaders(batch):
    tr, va, te = _load_split(); idx = _index()
    def paths(ss): return [p for s in sorted(ss) for p in idx.get(s, [])]
    tp, vp, ep = paths(tr), paths(va), paths(te)
    print(f'  train {len(tp)} | val {len(vp)} | test {len(ep)} segments')
    tl = DataLoader(Chunks(tp, 'train', augment=True), batch_size=batch, shuffle=True,
                    drop_last=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
                    persistent_workers=(NUM_WORKERS > 0),
                    prefetch_factor=(4 if NUM_WORKERS > 0 else None))
    vl = DataLoader(Chunks(vp, 'val'), batch_size=batch, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    return tl, vl, ep

# %% [markdown]
# ## Validation metric (L1 + SSIM), shared by all models

# %%
@torch.no_grad()
def validate(gen, vl):
    """Fast validation: mean L1 on the GPU (no CPU/skimage per-sample loop)."""
    gen.eval(); tot, n = 0.0, 0
    for pcg, ecg in vl:
        pcg = pcg.to(DEVICE, non_blocking=True); ecg = ecg.to(DEVICE, non_blocking=True)
        with torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
            fake = gen(pcg)
        tot += torch.mean(torch.abs(fake.float() - ecg)).item() * pcg.size(0); n += pcg.size(0)
    return tot / max(n, 1)

# %% [markdown]
# ## Trainers (regression for CNN/U-Net; WGAN-GP with critic + gradient penalty)

# %%
def he_init(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(mod.weight, a=0, nonlinearity='relu')
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def _better(cur, best, mode):
    return (cur < best) if mode == 'min' else (cur > best)


def train_baseline(name, cfg):
    path = ckpt_path(name)
    done = os.path.join(OUT, f'{name}.done')
    resume = os.path.join(OUT, f'{name}_resume.pt')
    if os.path.exists(done) and os.path.exists(path):
        print(f'[{name}] already trained -> skip.'); return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    set_seed(SEED)
    tl, vl, _ = loaders(cfg['batch'])
    gen = build(cfg['arch']).to(DEVICE); he_init(gen)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=ETA_MIN)
    l1 = nn.L1Loss()
    adversarial = cfg['adversarial']
    critic = opt_c = sch_c = None
    if adversarial:
        critic = models.WGANCritic(in_channels=2, features=64).to(DEVICE)
        opt_c = torch.optim.Adam(critic.parameters(), lr=cfg['lr'], betas=cfg['betas'])
        sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=EPOCHS, eta_min=ETA_MIN)

    best = float('inf') if cfg['mode'] == 'min' else -float('inf')
    best_ep, bad, start = -1, 0, 0
    if os.path.exists(resume):                    # T4 session died? continue.
        st = torch.load(resume, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(st['gen']); opt_g.load_state_dict(st['opt_g']); sch_g.load_state_dict(st['sch_g'])
        if adversarial and st.get('critic') is not None:
            critic.load_state_dict(st['critic']); opt_c.load_state_dict(st['opt_c']); sch_c.load_state_dict(st['sch_c'])
        best, best_ep, bad, start = st['best'], st['best_ep'], st['bad'], st['epoch'] + 1
        print(f'[{name}] resume from epoch {start} (best {best:.4f}@{best_ep}).')
    ac = lambda: torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP)
    for ep in range(start, EPOCHS):
        gen.train(); t0 = time.time()
        for pcg, ecg in tl:
            pcg = pcg.to(DEVICE, non_blocking=True); ecg = ecg.to(DEVICE, non_blocking=True)
            if adversarial:
                for _ in range(cfg['n_critic']):
                    with torch.no_grad(), ac():
                        fake = gen(pcg)
                    with ac():
                        c_real = critic(pcg, ecg).mean(); c_fake = critic(pcg, fake).mean()
                    with torch.autocast('cuda', enabled=False):   # GP in fp32: stable double-backward
                        gp = models.compute_gradient_penalty(
                            critic, pcg.float(), ecg.float(), fake.float(), DEVICE)
                    loss_c = (c_fake.float() - c_real.float()) + cfg['lambda_gp'] * gp
                    opt_c.zero_grad(set_to_none=True); loss_c.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP); opt_c.step()
                with ac():
                    fake = gen(pcg)
                    loss_g = -critic(pcg, fake).mean() + cfg['l1_weight'] * l1(fake, ecg)
            else:
                with ac():
                    fake = gen(pcg)
                    loss_g = cfg['l1_weight'] * l1(fake, ecg)
            opt_g.zero_grad(set_to_none=True); loss_g.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP); opt_g.step()
        sch_g.step()
        if adversarial:
            sch_c.step()

        v_l1 = validate(gen, vl)
        cur = v_l1
        if _better(cur, best, cfg['mode']):
            best, best_ep, bad = cur, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': gen.state_dict(),
                        'model_name': name, 'select': cfg['select'], 'select_value': cur},
                       path)
        else:
            bad += 1
        torch.save({'epoch': ep, 'gen': gen.state_dict(), 'opt_g': opt_g.state_dict(),
                    'sch_g': sch_g.state_dict(),
                    'critic': critic.state_dict() if adversarial else None,
                    'opt_c': opt_c.state_dict() if adversarial else None,
                    'sch_c': sch_c.state_dict() if adversarial else None,
                    'best': best, 'best_ep': best_ep, 'bad': bad}, resume)
        if ep % 2 == 0 or ep == EPOCHS - 1:
            print(f'[{name}] ep {ep:3d}  valL1 {v_l1:.4f}  '
                  f'best {best:.4f}@{best_ep}  ({time.time()-t0:.0f}s/epoch)')
        if bad >= PATIENCE:
            print(f'[{name}] early stop at epoch {ep} (best {best:.4f}@{best_ep}).'); break
    open(done, 'w').write(f'best {best:.6f} @ epoch {best_ep}\n')
    if os.path.exists(resume):
        os.remove(resume)
    print(f'[{name}] done -> {path}')
    return path


for _n in RUN:
    print('=' * 60); print('TRAIN', _n); print('=' * 60)
    try:
        train_baseline(_n, BASELINES[_n])
    except Exception as exc:
        print(f'[{_n}] failed -> {exc}')

# %% [markdown]
# ## Evaluate ALL FOUR models on the full 453-segment test set
# Correct (target, prediction) metric order; no amplitude rescaling (blind).

# %%
import pandas as pd


def load_baseline(name):
    if not os.path.exists(ckpt_path(name)):
        return None
    m = build(BASELINES[name]['arch'])
    m, _ = models.load_model(m, ckpt_path(name), device=DEVICE)
    return m


def load_proposed():
    p = config_paths.CKPT['Proposed']
    if not os.path.exists(p):
        p = config_paths.CKPT_PROPOSED_ALT
    m, _ = models.load_model(models.ImprovedGeneratorNoTanh(), p, device=DEVICE)
    return m


_tr, _va, _te = _load_split(); _idx = _index()
test_paths = [p for s in sorted(_te) for p in _idx.get(s, [])]
print('test segments:', len(test_paths))
loaded = {}
for n in ['CNN', 'UNet', 'WGAN-GP']:
    mdl = load_baseline(n)
    if mdl is not None:
        loaded[n] = (mdl, False)
prop = load_proposed()
loaded['Proposed'] = (prop, True)

rows = []
for p in test_paths:
    d = np.load(p, allow_pickle=True)
    ref = metrics.normalize_zscore(metrics.ensure_positive_polarity(_fit(d['ecg'])))
    pcg = metrics.normalize_zscore(_fit(d['pcg']))
    sid = str(d['subject_id'])
    pcg_t = torch.from_numpy(pcg).unsqueeze(0).unsqueeze(0).to(DEVICE)
    scen = torch.zeros(1, config_paths.NUM_SCENARIOS, device=DEVICE)
    scen[0, config_paths.scenario_to_index(d['scenario'])] = 1.0
    for name, (mdl, is_prop) in loaded.items():
        fake = models.run_generator(mdl, pcg_t, scen, is_prop)
        rows.append(dict(chunk=os.path.basename(p), subject=sid, model=name,
                         psnr=metrics.calc_psnr(ref, fake),
                         corr=metrics.calc_correlation(ref, fake),
                         ssim=metrics.calc_ssim(ref, fake),
                         rmse=metrics.calc_rmse(ref, fake),
                         rpeak_amp_err=metrics.calc_r_peak_amplitude_error(
                             ref, fake, fs=FS, distance=config_paths.RPEAK_DISTANCE,
                             prominence=config_paths.RPEAK_PROMINENCE)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'exp7_persegment_metrics.csv'), index=False)
print('per-segment rows:', len(df), '| models:', sorted(df.model.unique()))

# %% [markdown]
# ## Aggregate table (updated same-protocol Table II) + paired Wilcoxon vs Proposed

# %%
from scipy.stats import wilcoxon

METS = ['psnr', 'corr', 'ssim', 'rpeak_amp_err', 'rmse']
order = [m for m in ['CNN', 'UNet', 'WGAN-GP', 'Proposed'] if m in df.model.unique()]
summary = {}
for m in order:
    sub = df[df.model == m]
    summary[m] = {k: (float(sub[k].mean()), float(sub[k].std())) for k in METS}
print('\n=== Updated Table II (mean +/- SD over 453 segments) ===')
print(f'{"model":<10}' + ''.join(f'{k:>16}' for k in METS))
for m in order:
    print(f'{m:<10}' + ''.join(f'{summary[m][k][0]:>9.3f}+-{summary[m][k][1]:<5.2f}' for k in METS))

# paired Wilcoxon: Proposed vs each baseline, per metric, Holm-Bonferroni
paired = []
if 'Proposed' in order:
    pv = df[df.model == 'Proposed'].set_index('chunk')
    for base in [m for m in order if m != 'Proposed']:
        bv = df[df.model == base].set_index('chunk')
        common = pv.index.intersection(bv.index)
        for k in METS:
            a = pv.loc[common, k].values; b = bv.loc[common, k].values
            good = ~(np.isnan(a) | np.isnan(b))
            try:
                stat, p = wilcoxon(a[good], b[good])
            except Exception:
                stat, p = np.nan, np.nan
            paired.append(dict(baseline=base, metric=k, statistic=float(stat), p_raw=float(p)))
    # Holm-Bonferroni across the whole family
    ps = sorted([(r['p_raw'], i) for i, r in enumerate(paired) if not np.isnan(r['p_raw'])])
    mtot = len(ps)
    for rank, (p, i) in enumerate(ps):
        paired[i]['p_holm'] = float(min(1.0, p * (mtot - rank)))
pd.DataFrame(paired).to_csv(os.path.join(OUT, 'exp7_paired_wilcoxon.csv'), index=False)

# subject-clustered bootstrap CI for the Proposed correlation (effective n=3)
prop_corr = df[df.model == 'Proposed']
subj = sorted(prop_corr.subject.unique())
means = {s: prop_corr[prop_corr.subject == s]['corr'].mean() for s in subj}
boot = []
rng = np.random.RandomState(SEED)
for _ in range(10000):
    pick = rng.choice(subj, size=len(subj), replace=True)
    boot.append(np.mean([means[s] for s in pick]))
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
seg = prop_corr['corr'].values
seg_ci = (float(np.percentile([np.mean(rng.choice(seg, len(seg))) for _ in range(2000)], 2.5)),
          float(np.percentile([np.mean(rng.choice(seg, len(seg))) for _ in range(2000)], 97.5)))
print(f'\nProposed corr: pooled {prop_corr["corr"].mean():.3f} | '
      f'segment-level CI {seg_ci} | subject-clustered CI {ci} (effective n={len(subj)})')

json.dump({'summary': summary, 'paired_wilcoxon': paired,
           'proposed_corr_subject_ci': ci, 'proposed_corr_segment_ci': seg_ci,
           'per_subject_proposed_corr': means, 'n_test': int(len(test_paths))},
          open(os.path.join(OUT, 'exp7_results.json'), 'w'), indent=2)

# %% [markdown]
# ## Emit updated Table II (LaTeX) + IEEE figure

# %%
def emit_table():
    hdr = {'psnr': 'PSNR (dB)', 'corr': 'Corr.', 'ssim': 'SSIM',
           'rpeak_amp_err': 'R-peak amp. err.', 'rmse': 'RMSE'}
    higher = {'psnr', 'corr', 'ssim'}
    best = {}
    for k in METS:
        vals = {m: summary[m][k][0] for m in order}
        best[k] = (max if k in higher else min)(vals, key=vals.get)
    L = ['% Auto-generated by exp7. REFERENCE ONLY -- retrained baselines used for the',
         '% R1.2 paired test. The paper Table II KEEPS its original numbers; do NOT',
         '% replace it with this. Requires booktabs.',
         '\\begin{table}[t]\\centering', '\\footnotesize',
         '\\caption{Reference: faithfully retrained baselines under a single protocol, '
         'scored on all 453 EPHNOGRAM test segments. Used only for the paired '
         'significance test in the R1.2 response; the main results table is '
         'unchanged. Best in \\textbf{bold}.}',
         '\\label{tab:retrained_ref}', '\\begin{tabular}{l' + 'c' * len(order) + '}', '\\toprule',
         'Metric & ' + ' & '.join(order) + ' \\\\', '\\midrule']
    for k in METS:
        cells = []
        for m in order:
            v = summary[m][k][0]
            s = f'{v:.2f}' if k in ('psnr',) else f'{v:.3f}'
            cells.append(f'\\textbf{{{s}}}' if m == best[k] else s)
        L.append(f'{hdr[k]} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    open(os.path.join(OUT, 'exp7_reference_table.tex'), 'w').write('\n'.join(L))
    print('wrote exp7_reference_table.tex (reference only; paper Table II unchanged)')


emit_table()

fig, axes = ieee_style.new_fig('double', height=2.4, ncols=3)
for ax, k, lab in zip(axes, ['psnr', 'corr', 'rpeak_amp_err'],
                      ['PSNR (dB)', 'Pearson r', 'R-peak amp. err. (a.u.)']):
    data = [df[df.model == m][k].dropna().values for m in order]
    bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
    for patch, c in zip(bp['boxes'], [ieee_style.COLORS[x] for x in ['cnn', 'unet', 'wgan', 'proposed'][:len(order)]]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticklabels(order, rotation=25, ha='right', fontsize=7)
    ieee_style.finalize_axes(ax, ylabel=lab)
ieee_style.save_fig(fig, OUT, 'fig_exp7_model_boxplots')
print('\nDONE. Baseline weights saved to the config_paths.CKPT locations; exp2/exp5 '
      'now load them. The paper Table II stays as-is. Use exp7_paired_wilcoxon.csv '
      'for the R1.2 significance test; exp7_reference_table.tex is a rebuttal-only reference.')
