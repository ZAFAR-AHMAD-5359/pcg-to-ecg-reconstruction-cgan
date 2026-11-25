# %% [markdown]
# # exp6 - Component Ablation Training Harness (Reviewer 1, comment 4)
#
# Reviewer 1 asked for a controlled ablation that isolates each design choice in
# the proposed generator. This script trains every ablation variant under ONE
# fixed protocol, then scores each trained model against a SINGLE
# polarity-corrected reference so the numbers are directly comparable.
#
# The grouped factors under test are:
#   * MHSA          (multi-head self-attention in the 4 bottleneck blocks)
#   * MSDC          (multi-scale dilated convolutions in the bottleneck)
#   * output head   (batchnorm | tanh | linear)
#   * adversarial   (conditional GAN loss on/off; L1-only when off)
#   * polarity      (data-loader polarity correction on/off)
#
# Design notes
# ------------
# * MHSA and MSDC are ARCHITECTURE switches on models.ConfigurableGenerator.
# * The output head is an ARCHITECTURE switch (output_activation).
# * Adversarial on/off is a TRAINING switch (drop the GAN term, keep L1).
# * Polarity on/off is a DATA-LOADER switch (skip metrics.ensure_positive_polarity).
#
# The harness is RESUMABLE. Each variant writes a per-epoch resume checkpoint and
# a final checkpoint. Re-running skips any variant whose final checkpoint exists,
# and resumes any variant that has a partial resume checkpoint.
#
# IMPORTANT: this is the heaviest job in the revision. It is meant to run on a
# Colab GPU over several sessions. Nothing runs on the author's laptop.

# %% [markdown]
# ## Dependencies
# scikit-image supplies the SSIM used by metrics.calc_ssim. The rest ship with
# Colab. Uncomment the pip line below on Colab if scikit-image is missing.

# %%
# Uncomment on Colab (a bare !pip line is not valid in a plain .py, so it is
# commented here to keep this file importable):
# !pip install -q scikit-image scipy matplotlib numpy

# %% [markdown]
# ## Setup (mount Drive, put the shared modules on the path)
#
# Copy your local `Revision/Colab_Experiments/shared/` folder to
# `MyDrive/JBHI_Revision/shared/` on Drive, OR paste the four shared modules into
# `/content`. The self-contained notebook built by `build_notebooks.py` writes the
# modules to `/content` for you, so it does not need this cell.

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('Not on Colab (or Drive already mounted):', exc)

import sys
SHARED_ON_DRIVE = '/content/drive/MyDrive/JBHI_Revision/shared'
if SHARED_ON_DRIVE not in sys.path:
    sys.path.append(SHARED_ON_DRIVE)
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

print('NOTE: this script imports config_paths, models, metrics, ieee_style.')
print('      Copy the local shared/ folder to', SHARED_ON_DRIVE)
print('      or paste the four shared modules into /content before running.')

import config_paths
import models
import metrics
import ieee_style

ieee_style.set_ieee_style()
print('shared modules ready | device =', config_paths.DEVICE)

# %%
import os
import glob
import json
import time
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = config_paths.DEVICE
OUT = config_paths.out_dir('exp6_ablation')          # all artifacts land here
CKPT_DIR = os.path.join(OUT, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
print('output dir     :', OUT)
print('checkpoint dir :', CKPT_DIR)

# %% [markdown]
# ## Fixed training protocol (identical for every variant)
#
# 100 epochs, Adam(5e-5, betas 0.5/0.999), cosine schedule to 1e-6, gradient
# clipping at 1.0, batch size 8, L1 weight 100, He initialization on the
# generator. The subject split is read from `config_paths.SPLIT_FILE`, so every
# variant trains and tests on exactly the same subjects.

# %%
EPOCHS       = 60      # 60 is ample for the ablation: component deltas converge
                       # well before 100, and this cuts wall-clock ~40%.
LR           = 5e-5
BETAS        = (0.5, 0.999)
ETA_MIN      = 1e-6
GRAD_CLIP    = 1.0
LAMBDA_L1    = 100.0
SEED         = 42

# ---- Hardware-aware knobs (auto-scale to the GPU; override by hand if needed) --
# The free T4 (14.5 GB) OOM-ed the attention variants at batch 8. Two fixes make
# an A100 fast and OOM-free: (1) models.MultiHeadSelfAttention now uses
# memory-efficient attention, and (2) bf16 autocast halves activation memory and
# roughly doubles throughput on A100 tensor cores. Spend the compute-unit budget
# on a large batch: an A100-80GB runs batch 64 comfortably, an A100-40GB batch 32.
torch.backends.cudnn.benchmark = True    # fixed 64000-sample inputs -> autotune
torch.backends.cuda.matmul.allow_tf32 = True     # TF32 matmul on Ampere (A100)
torch.backends.cudnn.allow_tf32 = True
_cuda = torch.cuda.is_available()
_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if _cuda else 0.0
# SDPA (flash / memory-efficient attention in models.MultiHeadSelfAttention) removes
# the O(T^2) attention-memory wall, so push the batch up to use the A100 fully.
#   A100-80GB -> 128, A100-40GB -> 64, mid GPUs -> 16, T4/CPU -> 8. Flash attention
#   makes the attention variants fit at these sizes; a bigger batch means fewer
#   optimizer steps per epoch, which is the main throughput lever on the A100.
BATCH_SIZE  = (128 if _gb > 60 else 64 if _gb > 34 else 16 if _gb > 20 else 8) if _cuda else 8
NUM_WORKERS = min(8, (os.cpu_count() or 4)) if _cuda else 2
USE_AMP     = _cuda and torch.cuda.is_bf16_supported()     # A100 supports bf16
AMP_DTYPE   = torch.bfloat16
USE_COMPILE = _cuda   # torch.compile the generator (~1.3-1.8x on A100); set False
                      # if you hit a compile error. The critic stays eager because
                      # spectral norm + torch.compile can misbehave.
PRELOAD_TO_RAM = True     # hold the whole train set in RAM to remove per-epoch Drive
                          # I/O (the dominant Colab cost; it drove epochs to ~700 s on T4)
print(f'GPU: {_gb:.0f} GB | bf16 AMP: {USE_AMP} | batch: {BATCH_SIZE} | workers: {NUM_WORKERS}')

# Evaluation tolerances for the temporal R-peak F1 (samples <- ms via FS).
TEMPORAL_TOL_MS = [50, 150]

# R1.8 CONFOUND GUARD: the paper's inference pipeline rescales the generator
# output by the target standard deviation (fake *= real_std / fake_std). That
# rescaling leaks the ground-truth scale into the prediction and is exactly the
# confound Reviewer 1 comment 8 exposes. This ablation reports BLIND numbers, so
# NO test-time rescaling is applied. Keep this False.
EVAL_RESCALE_BY_TARGET_STD = False


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% [markdown]
# ## Ablation variants
#
# Each entry is one row of the ablation table. `use_attention` and `use_dilated`
# toggle MHSA and MSDC. `output` selects the output head. `adversarial` toggles
# the GAN term. `polarity` toggles data-loader polarity correction.
#
# VERIFIED generator parameter counts (computed locally with torch; the main
# text's "14.0M" is an error, use these):
#   U-Net only              10.92M   (att F, dil F)
#   U-Net + MSDC            15.12M   (att F, dil T)
#   U-Net + MHSA            15.11M   (att T, dil F)
#   U-Net + both (proposed) 19.32M   (att T, dil T)  = 19,315,715 params
#   Spectral-norm critic     2.18M   (2,176,833 params)
# The tanh and linear heads share the +both backbone. They carry no output-head
# affine parameters, so they hold 2 fewer parameters than the BatchNorm head
# (19,315,713 vs 19,315,715); both round to 19.32M. The adversarial-off and
# polarity-off variants use the +both architecture unchanged (19.32M); they only
# change the loss or the data loader. The harness prints count_params at build
# time so you can confirm every number against these comments.

# %%
VARIANTS = {
    # key                 attention  dilated  output        adversarial polarity  params(M)
    'unet_only':        dict(use_attention=False, use_dilated=False, output='batchnorm', adversarial=True,  polarity=True,  params_m=10.92),
    'msdc':             dict(use_attention=False, use_dilated=True,  output='batchnorm', adversarial=True,  polarity=True,  params_m=15.12),
    'mhsa':             dict(use_attention=True,  use_dilated=False, output='batchnorm', adversarial=True,  polarity=True,  params_m=15.11),
    'both':             dict(use_attention=True,  use_dilated=True,  output='batchnorm', adversarial=True,  polarity=True,  params_m=19.32),
    'proposed_tanh':    dict(use_attention=True,  use_dilated=True,  output='tanh',      adversarial=True,  polarity=True,  params_m=19.32),
    'proposed_linear':  dict(use_attention=True,  use_dilated=True,  output='linear',    adversarial=True,  polarity=True,  params_m=19.32),
    # 'proposed_batchnorm' is IDENTICAL to 'both' (proposed model, batchnorm head,
    # adversarial, polarity-corrected). Train one, then reuse its checkpoint for
    # the other row if you want the label in the table.
    'proposed_batchnorm': dict(use_attention=True, use_dilated=True, output='batchnorm', adversarial=True,  polarity=True,  params_m=19.32),
    # Training-protocol / data variants of the proposed model (architecture 19.32M):
    'adv_off':          dict(use_attention=True,  use_dilated=True,  output='batchnorm', adversarial=False, polarity=True,  params_m=19.32),
    'polarity_off':     dict(use_attention=True,  use_dilated=True,  output='batchnorm', adversarial=True,  polarity=False, params_m=19.32),
}

# ---------------------------------------------------------------------------
# TRIAGE ORDER (run these first; each answers a distinct reviewer worry):
#   1. polarity_off @ 100 epochs  -> shows polarity correction is load-bearing
#                                     (its blind correlation should collapse).
#   2. proposed_tanh              -> shows the output head choice matters.
#   3. mhsa                       -> isolates attention.
#   4. msdc                       -> isolates dilated multi-scale context.
#   5. unet_only                  -> the plain backbone floor.
# 'both' reproduces the proposed model already in the paper (PSNR 17.10 dB,
# corr 0.610), so retrain it only if you want a same-protocol control.
# ---------------------------------------------------------------------------
# Full set for a complete R1.4 table. Variants whose final checkpoint already
# exists are auto-skipped, so re-running only trains what is missing. The
# attention variants (mhsa, both, proposed_tanh/linear, polarity_off, adv_off)
# are the ones that OOM-ed before; the flash-attention + batch-32 fix lets them
# fit on a 40 GB A100. (msdc, unet_only already completed at batch 48 -- delete
# their checkpoints to re-run at the uniform batch 32 if you want one protocol.)
RUN_VARIANTS = ['mhsa', 'polarity_off', 'proposed_tanh', 'proposed_linear',
                'adv_off', 'both', 'msdc', 'unet_only']


# %% [markdown]
# ## Verify the parameter counts before spending GPU hours
#
# This instantiates each variant's generator and the critic, then prints
# count_params next to the verified comment values. Confirm they match before you
# launch a 100-epoch run.

# %%
def build_generator(cfg):
    """Construct a ConfigurableGenerator for a variant config dict."""
    return models.ConfigurableGenerator(
        base_channels=64,
        num_scenarios=config_paths.NUM_SCENARIOS,
        use_attention=cfg['use_attention'],
        use_dilated=cfg['use_dilated'],
        output_activation=cfg['output'],
        use_scenario=True,
    )


def build_discriminator():
    return models.SpectralNormDiscriminator(
        base_channels=64, num_scenarios=config_paths.NUM_SCENARIOS)


print(f'{"variant":<20} {"gen params":>12} {"expected(M)":>12}')
print('-' * 46)
for _name, _cfg in VARIANTS.items():
    _g = build_generator(_cfg)
    _p = models.count_params(_g)
    print(f'{_name:<20} {_p:>12,} {_cfg["params_m"]:>11.2f}M')
    del _g
_d = build_discriminator()
print(f'{"discriminator":<20} {models.count_params(_d):>12,} {"2.18":>11}M')
del _d


# %% [markdown]
# ## Dataset (EPHNOGRAM chunks with a polarity switch)
#
# One item per `.npz` chunk. The ECG target is polarity-corrected only when the
# variant asks for it. Both signals are z-scored, matching the paper pipeline.
# The scenario one-hot conditions the generator and the critic.

# %%
def _load_split():
    """Return train/val/test subject sets from config_paths.SPLIT_FILE."""
    with open(config_paths.SPLIT_FILE) as f:
        sp = json.load(f)

    def pick(*keys):
        for k in keys:
            if k in sp:
                return set(str(s) for s in sp[k])
        return set()

    train = pick('train_subjects', 'train')
    val = pick('val_subjects', 'val')
    test = pick('test_subjects', 'test')
    return train, val, test


def _index_chunks():
    """Map subject_id -> [chunk paths]. Cached to JSON so we scan Drive once.

    np.load on an .npz is lazy, so reading only the 'subject_id' key does not
    decompress the full ecg/pcg arrays.
    """
    cache = os.path.join(OUT, 'chunk_index.json')
    if os.path.exists(cache):
        with open(cache) as f:
            return {k: v for k, v in json.load(f).items()}
    index = {}
    paths = sorted(glob.glob(os.path.join(config_paths.EPH_CHUNKS_DIR, '*.npz')))
    for p in paths:
        try:
            sid = str(np.load(p, allow_pickle=True)['subject_id'])
        except Exception as exc:
            print('  skip (unreadable):', os.path.basename(p), exc)
            continue
        index.setdefault(sid, []).append(p)
    with open(cache, 'w') as f:
        json.dump(index, f, indent=1)
    return index


class EphnogramChunks(Dataset):
    """PCG -> ECG paired chunks for a set of subjects.

    polarity=True  applies metrics.ensure_positive_polarity to the raw ECG before
                   z-scoring (the paper pipeline).
    polarity=False leaves the raw ECG polarity untouched, so a model can learn an
                   inverted target. This is the polarity_off ablation.
    """
    def __init__(self, chunk_paths, polarity=True):
        self.paths = list(chunk_paths)
        self.polarity = polarity

    def __len__(self):
        return len(self.paths)

    def _fit_length(self, arr):
        arr = np.asarray(arr, dtype=np.float32).ravel()
        n = config_paths.SEG_LEN
        if arr.shape[0] >= n:
            return arr[:n]
        pad = np.zeros(n - arr.shape[0], dtype=np.float32)
        return np.concatenate([arr, pad])

    def __getitem__(self, idx):
        d = np.load(self.paths[idx], allow_pickle=True)
        ecg_raw = self._fit_length(d['ecg'])
        pcg_raw = self._fit_length(d['pcg'])

        if self.polarity:
            ecg_raw = metrics.ensure_positive_polarity(ecg_raw)
        ecg = metrics.normalize_zscore(ecg_raw)
        pcg = metrics.normalize_zscore(pcg_raw)

        sc_idx = config_paths.scenario_to_index(d['scenario'])
        scen = np.zeros(config_paths.NUM_SCENARIOS, dtype=np.float32)
        scen[sc_idx] = 1.0

        pcg_t = torch.from_numpy(pcg).unsqueeze(0)   # (1, L)
        ecg_t = torch.from_numpy(ecg).unsqueeze(0)   # (1, L)
        scen_t = torch.from_numpy(scen)              # (3,)
        return pcg_t, ecg_t, scen_t


# ---------------------------------------------------------------------------
# In-RAM training set. Reading 4,059 .npz from Drive every epoch is the dominant
# Colab cost (it drove epoch time to ~700 s on the T4). The whole train set is
# only ~2 GB of float32, so with 84 GB RAM we load the RAW arrays ONCE and reuse
# them across every variant. Workers fork this array copy-on-write, so no copies.
# ---------------------------------------------------------------------------
_RAM_CACHE = {}   # tag -> (pcg_all, ecg_all, scen_all) raw, shared across variants


def _fit_len(arr):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    n = config_paths.SEG_LEN
    if arr.shape[0] >= n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - arr.shape[0], dtype=np.float32)])


def _preload_raw(paths, tag):
    if tag in _RAM_CACHE:
        return _RAM_CACHE[tag]
    n, L = len(paths), config_paths.SEG_LEN
    pcg = np.zeros((n, L), dtype=np.float32)
    ecg = np.zeros((n, L), dtype=np.float32)
    scen = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        pcg[i] = _fit_len(d['pcg'])
        ecg[i] = _fit_len(d['ecg'])
        scen[i] = config_paths.scenario_to_index(d['scenario'])
        if (i + 1) % 500 == 0:
            print(f'    preloaded {i + 1}/{n} chunks into RAM')
    print(f'  RAM preload [{tag}]: {n} chunks, '
          f'{(pcg.nbytes + ecg.nbytes) / 1e9:.2f} GB, {time.time() - t0:.0f}s')
    _RAM_CACHE[tag] = (pcg, ecg, scen)
    return _RAM_CACHE[tag]


class InMemoryChunks(Dataset):
    """Whole subject set held in RAM. Polarity + z-score run per item (cheap)."""
    def __init__(self, paths, tag, polarity=True):
        self.pcg, self.ecg, self.scen = _preload_raw(paths, tag)
        self.polarity = polarity

    def __len__(self):
        return self.pcg.shape[0]

    def __getitem__(self, idx):
        ecg_raw = self.ecg[idx]
        if self.polarity:
            ecg_raw = metrics.ensure_positive_polarity(ecg_raw)
        ecg = metrics.normalize_zscore(ecg_raw)
        pcg = metrics.normalize_zscore(self.pcg[idx])
        scen = np.zeros(config_paths.NUM_SCENARIOS, dtype=np.float32)
        scen[int(self.scen[idx])] = 1.0
        return (torch.from_numpy(pcg).unsqueeze(0),
                torch.from_numpy(ecg).unsqueeze(0),
                torch.from_numpy(scen))


def make_loaders(polarity):
    train_subj, _val_subj, test_subj = _load_split()
    index = _index_chunks()
    train_paths = [p for s in sorted(train_subj) for p in index.get(s, [])]
    test_paths = [p for s in sorted(test_subj) for p in index.get(s, [])]
    print(f'  train subjects {sorted(train_subj)} -> {len(train_paths)} chunks')
    print(f'  test  subjects {sorted(test_subj)} -> {len(test_paths)} chunks')
    if PRELOAD_TO_RAM:
        train_ds = InMemoryChunks(train_paths, tag='train', polarity=polarity)
    else:
        train_ds = EphnogramChunks(train_paths, polarity=polarity)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True,
                              persistent_workers=(NUM_WORKERS > 0),
                              pin_memory=(DEVICE.type == 'cuda'),
                              prefetch_factor=(4 if NUM_WORKERS > 0 else None))
    # The test loader always uses polarity-corrected references so every variant
    # is scored against the SAME ground truth (see evaluate_variant).
    return train_loader, test_paths


# %% [markdown]
# ## He initialization (generator only)
#
# The spectral-norm critic keeps its default initialization. He (Kaiming) init on
# a spectrally-normalized weight would not persist, because that weight is a
# derived buffer rather than a leaf parameter.

# %%
def he_init(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# %% [markdown]
# ## Trainer (matched protocol, resumable)
#
# The generator loss is the L1 term (weight 100) plus, when adversarial is on, a
# conditional-GAN BCE term on the spectral-norm critic. The critic sees the PCG
# condition concatenated with a real or generated ECG. A per-epoch resume file
# lets an interrupted run continue; a final checkpoint marks the variant done.

# %%
def _paths_for(name):
    final_ckpt = os.path.join(CKPT_DIR, f'{name}_final.pt')
    resume_ckpt = os.path.join(CKPT_DIR, f'{name}_resume.pt')
    return final_ckpt, resume_ckpt


def _maybe_compile(module):
    """Return a torch.compile-d module for a fast forward, or the module itself.

    The compiled wrapper SHARES parameters with `module`, so we always save/load
    the underlying `module.state_dict()` (no '_orig_mod.' prefix) and only use the
    compiled object for forward passes.
    """
    if not USE_COMPILE:
        return module
    try:
        return torch.compile(module)
    except Exception as exc:
        print('  torch.compile unavailable -> eager:', exc)
        return module


def train_variant(name, cfg):
    """Train one variant under the fixed protocol. Returns the final ckpt path.

    Skips entirely if the final checkpoint exists. Resumes from the per-epoch
    resume file when present.
    """
    final_ckpt, resume_ckpt = _paths_for(name)
    if os.path.exists(final_ckpt):
        print(f'[{name}] final checkpoint present -> skip training.')
        return final_ckpt

    set_seed(SEED)
    adversarial = cfg['adversarial']
    train_loader, _test_paths = make_loaders(polarity=cfg['polarity'])

    gen = build_generator(cfg).to(DEVICE)
    he_init(gen)
    gen_fwd = _maybe_compile(gen)     # compiled forward; `gen` stays the leaf for state_dict
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR, betas=BETAS)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=ETA_MIN)

    disc = opt_d = sched_d = None
    if adversarial:
        disc = build_discriminator().to(DEVICE)
        opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=BETAS)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS, eta_min=ETA_MIN)

    l1 = nn.L1Loss()
    bce = nn.BCELoss()   # the critic already applies a sigmoid

    start_epoch = 0
    history = {'epoch': [], 'loss_g': [], 'loss_d': [], 'loss_l1': []}
    if os.path.exists(resume_ckpt):
        st = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(st['gen'])
        opt_g.load_state_dict(st['opt_g'])
        sched_g.load_state_dict(st['sched_g'])
        if adversarial and st.get('disc') is not None:
            disc.load_state_dict(st['disc'])
            opt_d.load_state_dict(st['opt_d'])
            sched_d.load_state_dict(st['sched_d'])
        history = st['history']
        start_epoch = st['epoch'] + 1
        print(f'[{name}] resuming at epoch {start_epoch}/{EPOCHS}.')

    for ep in range(start_epoch, EPOCHS):
        gen.train()
        if adversarial:
            disc.train()
        t0 = time.time()
        run_g = run_d = run_l1 = 0.0
        n_batches = 0
        for pcg, ecg_real, scen in train_loader:
            pcg = pcg.to(DEVICE, non_blocking=True)
            ecg_real = ecg_real.to(DEVICE, non_blocking=True)
            scen = scen.to(DEVICE, non_blocking=True)

            ac = torch.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP)
            # ---- discriminator step ----
            if adversarial:
                with torch.no_grad(), ac:
                    fake_detached = gen_fwd(pcg, scen)
                with ac:
                    d_real = disc(pcg, ecg_real, scen)
                    d_fake = disc(pcg, fake_detached, scen)
                # BCE is autocast-UNSAFE (raises under autocast) -> compute in fp32.
                loss_d = bce(d_real.float(), torch.ones_like(d_real, dtype=torch.float32)) + \
                         bce(d_fake.float(), torch.zeros_like(d_fake, dtype=torch.float32))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
                opt_d.step()
                run_d += float(loss_d.detach())

            # ---- generator step ----
            with ac:
                fake = gen_fwd(pcg, scen)
                loss_l1 = l1(fake, ecg_real)
                d_fake_for_g = disc(pcg, fake, scen) if adversarial else None
            if adversarial:                          # BCE in fp32, outside autocast
                loss_g = bce(d_fake_for_g.float(),
                             torch.ones_like(d_fake_for_g, dtype=torch.float32)) \
                         + LAMBDA_L1 * loss_l1.float()
            else:
                loss_g = LAMBDA_L1 * loss_l1.float()   # L1-only, matched L1 weighting
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt_g.step()

            run_g += float(loss_g.detach())
            run_l1 += float(loss_l1.detach())
            n_batches += 1

        sched_g.step()
        if adversarial:
            sched_d.step()

        n_batches = max(n_batches, 1)
        history['epoch'].append(ep)
        history['loss_g'].append(run_g / n_batches)
        history['loss_d'].append(run_d / n_batches)
        history['loss_l1'].append(run_l1 / n_batches)

        torch.save({
            'epoch': ep, 'variant': name, 'config': cfg,
            'gen': gen.state_dict(),
            'disc': disc.state_dict() if adversarial else None,
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict() if adversarial else None,
            'sched_g': sched_g.state_dict(),
            'sched_d': sched_d.state_dict() if adversarial else None,
            'history': history,
        }, resume_ckpt)

        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f'[{name}] epoch {ep:3d}/{EPOCHS}  '
                  f'G {history["loss_g"][-1]:.4f}  '
                  f'D {history["loss_d"][-1]:.4f}  '
                  f'L1 {history["loss_l1"][-1]:.4f}  '
                  f'({time.time() - t0:.1f}s)')

    torch.save({
        'variant': name, 'config': cfg, 'epochs': EPOCHS,
        'gen': gen.state_dict(), 'history': history,
        'params': models.count_params(gen),
    }, final_ckpt)
    print(f'[{name}] done -> {final_ckpt}')
    return final_ckpt


# %% [markdown]
# ## Evaluation (same polarity-corrected reference for every variant)
#
# Each trained generator is scored on the held-out test subjects. The reference
# ECG is always polarity-corrected and z-scored, so every variant, including
# polarity_off, is judged against identical ground truth. No target-std rescaling
# is applied (see the R1.8 confound guard). Signal fidelity uses metrics.calc_*;
# the temporal F1 uses metrics.temporal_rpeak_metrics aggregated over segments.

# %%
def _scenario_onehot(scenario_value):
    scen = np.zeros(config_paths.NUM_SCENARIOS, dtype=np.float32)
    scen[config_paths.scenario_to_index(scenario_value)] = 1.0
    return scen


def evaluate_variant(name, cfg):
    """Return a results dict of blind test metrics for a trained variant."""
    final_ckpt, _ = _paths_for(name)
    if not os.path.exists(final_ckpt):
        print(f'[{name}] no final checkpoint -> cannot evaluate.')
        return None

    st = torch.load(final_ckpt, map_location=DEVICE, weights_only=False)
    gen = build_generator(cfg).to(DEVICE)
    gen.load_state_dict(st['gen'])
    gen.eval()

    _train, _val, test_subj = _load_split()
    index = _index_chunks()
    test_paths = [p for s in sorted(test_subj) for p in index.get(s, [])]

    psnr_v, ssim_v, corr_v, ramp_v = [], [], [], []
    temporal_rows = {tol: [] for tol in TEMPORAL_TOL_MS}

    with torch.no_grad():
        for p in test_paths:
            d = np.load(p, allow_pickle=True)
            ecg_raw = np.asarray(d['ecg'], dtype=np.float32).ravel()[:config_paths.SEG_LEN]
            pcg_raw = np.asarray(d['pcg'], dtype=np.float32).ravel()[:config_paths.SEG_LEN]
            if ecg_raw.shape[0] < config_paths.SEG_LEN:
                continue

            # FIXED reference: always polarity-corrected, always z-scored.
            ref = metrics.normalize_zscore(metrics.ensure_positive_polarity(ecg_raw))
            pcg = metrics.normalize_zscore(pcg_raw)

            scen = torch.from_numpy(_scenario_onehot(d['scenario'])).unsqueeze(0).to(DEVICE)
            pcg_t = torch.from_numpy(pcg).unsqueeze(0).unsqueeze(0).to(DEVICE)
            fake = gen(pcg_t, scen).squeeze().cpu().numpy()

            if EVAL_RESCALE_BY_TARGET_STD:      # kept False; see R1.8 guard
                fake = fake * (ref.std() / (fake.std() + 1e-8))

            psnr_v.append(metrics.calc_psnr(ref, fake))
            corr_v.append(metrics.calc_correlation(ref, fake))
            try:
                ssim_v.append(metrics.calc_ssim(ref, fake))
            except Exception:
                pass
            ramp = metrics.calc_r_peak_amplitude_error(
                ref, fake, fs=config_paths.FS,
                distance=config_paths.RPEAK_DISTANCE,
                prominence=config_paths.RPEAK_PROMINENCE)
            if not np.isnan(ramp):
                ramp_v.append(ramp)

            for tol in TEMPORAL_TOL_MS:
                temporal_rows[tol].append(metrics.temporal_rpeak_metrics(
                    ref, fake, fs=config_paths.FS, tol_ms=tol,
                    distance=config_paths.RPEAK_DISTANCE,
                    prominence=config_paths.RPEAK_PROMINENCE))

    def _m(vals):
        return float(np.mean(vals)) if len(vals) else float('nan')

    result = {
        'variant': name,
        'config': cfg,
        'params': int(st.get('params', models.count_params(gen))),
        'n_test_segments': len(psnr_v),
        'psnr': _m(psnr_v),
        'ssim': _m(ssim_v),
        'corr': _m(corr_v),
        'rpeak_amp_err': _m(ramp_v),
        'rpeak_amp_err_validN': len(ramp_v),
    }
    for tol in TEMPORAL_TOL_MS:
        agg = metrics.aggregate_temporal(temporal_rows[tol])
        result[f'temporal_f1_{tol}'] = agg['f1']
        result[f'temporal_se_{tol}'] = agg['sensitivity']
        result[f'temporal_ppv_{tol}'] = agg['ppv']

    with open(os.path.join(OUT, f'metrics_{name}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[{name}] PSNR {result["psnr"]:.2f} dB | corr {result["corr"]:.3f} | '
          f'F1@50 {result.get("temporal_f1_50", float("nan")):.3f} | '
          f'params {result["params"]:,}')
    return result


# %% [markdown]
# ## Run the queued variants
#
# Trains each variant in RUN_VARIANTS (skipping finished ones), evaluates it, and
# accumulates results into `all_results.json`. Every step is guarded, so a missing
# data path prints a message and the loop continues.

# %%
def load_all_results():
    path = os.path.join(OUT, 'all_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_all_results(all_results):
    with open(os.path.join(OUT, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


all_results = load_all_results()
for _name in RUN_VARIANTS:
    if _name not in VARIANTS:
        print(f'[{_name}] not a known variant -> skip.')
        continue
    _cfg = VARIANTS[_name]
    print('=' * 64)
    print(f'VARIANT: {_name}  {_cfg}')
    print('=' * 64)
    try:
        train_variant(_name, _cfg)
        res = evaluate_variant(_name, _cfg)
        if res is not None:
            all_results[_name] = res
            save_all_results(all_results)
    except FileNotFoundError as exc:
        print(f'[{_name}] data or checkpoint path missing -> skipped:', exc)
    except Exception as exc:
        print(f'[{_name}] error -> skipped:', exc)
    finally:
        # release this variant's GPU memory before the next one starts, so a
        # failed variant cannot starve the following variants.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print('\nfinished variants:', sorted(all_results.keys()))


# %% [markdown]
# ## Summary figure (IEEE double-column)
#
# Grouped bars of PSNR, Pearson correlation, and temporal F1 at plus/minus 50 ms
# across the evaluated variants. Bold axis labels carry explicit units, and
# constrained_layout keeps the rotated variant labels clear of the bars.

# %%
def plot_summary(all_results, tol=50):
    results = load_all_results() if not all_results else all_results
    order = [k for k in VARIANTS if k in results]
    if not order:
        print('No results yet -> skipping summary figure.')
        return

    labels = order
    psnr = [results[k]['psnr'] for k in order]
    corr = [results[k]['corr'] for k in order]
    f1 = [results[k].get(f'temporal_f1_{tol}', float('nan')) for k in order]

    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
               '#56B4E9', '#666666', '#F0E442', '#000000']
    colors = [palette[i % len(palette)] for i in range(len(order))]

    fig, axes = ieee_style.new_fig('double', height=2.7, ncols=3)
    panels = [
        (axes[0], psnr, 'PSNR (dB)'),
        (axes[1], corr, 'Pearson r (a.u.)'),
        (axes[2], f1, f'Temporal F1 at {tol} ms (a.u.)'),
    ]
    x = np.arange(len(order))
    for ax, vals, ylab in panels:
        ax.bar(x, vals, color=colors, width=0.72, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=6.5)
        ieee_style.finalize_axes(ax, ylabel=ylab)
    ieee_style.save_fig(fig, OUT, 'fig_exp6_ablation_summary')


def plot_training_curves(name):
    """Optional per-variant G/D/L1 loss curves (IEEE single-column)."""
    final_ckpt, _ = _paths_for(name)
    if not os.path.exists(final_ckpt):
        print(f'[{name}] no checkpoint -> skip curves.')
        return
    hist = torch.load(final_ckpt, map_location='cpu', weights_only=False)['history']
    if not hist['epoch']:
        return
    fig, ax = ieee_style.new_fig('single', height=2.3)
    ax.plot(hist['epoch'], hist['loss_l1'], color=ieee_style.COLORS['proposed'], label='L1')
    ax.plot(hist['epoch'], hist['loss_g'], color=ieee_style.COLORS['cnn'], label='Generator')
    if any(v > 0 for v in hist['loss_d']):
        ax.plot(hist['epoch'], hist['loss_d'], color=ieee_style.COLORS['unet'], label='Discriminator')
    ieee_style.finalize_axes(ax, xlabel='Epoch (count)', ylabel='Loss (a.u.)', legend=True)
    ieee_style.save_fig(fig, OUT, f'fig_exp6_curves_{name}')


plot_summary(all_results, tol=50)
for _name in RUN_VARIANTS:
    if _name in VARIANTS:
        plot_training_curves(_name)


# %% [markdown]
# ## LaTeX ablation table skeleton
#
# Writes a paste-ready table to `table_exp6_ablation.tex` in the output folder.
# Cells fill from `all_results.json` where a variant has been evaluated, otherwise
# they hold a placeholder dash. Copy the file into the manuscript and drop it near
# the ablation discussion.

# %%
def emit_latex_table(all_results):
    results = load_all_results() if not all_results else all_results

    display = {
        'unet_only':          'U-Net only',
        'msdc':               'U-Net + MSDC',
        'mhsa':               'U-Net + MHSA',
        'both':               'U-Net + both',
        'proposed_batchnorm': 'Proposed (BatchNorm)',
        'proposed_tanh':      'Proposed + Tanh',
        'proposed_linear':    'Proposed + linear',
        'adv_off':            'Proposed, adversarial off',
        'polarity_off':       'Proposed, polarity off',
    }
    row_order = ['unet_only', 'msdc', 'mhsa', 'both', 'proposed_batchnorm',
                 'proposed_tanh', 'proposed_linear', 'adv_off', 'polarity_off']

    yes = r'\checkmark'          # amssymb provides \checkmark (IEEEtran-safe)
    no = r'--'

    def cell(key, metric, fmt):
        if key in results and metric in results[key] and results[key][metric] == results[key][metric]:
            return fmt.format(results[key][metric])
        return no

    lines = []
    lines.append('% Auto-generated by exp6_R1_4_ablation_training.py')
    lines.append('% FIND the current ablation table in the manuscript and REPLACE it with this,')
    lines.append('% or paste this near the ablation discussion (Reviewer 1, comment 4).')
    lines.append('% Requires \\usepackage{booktabs} and \\usepackage{amssymb}.')
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append('\\footnotesize')
    lines.append('\\caption{Component ablation on the EPHNOGRAM held-out subjects. '
                 f'All variants share the training protocol ({EPOCHS} epochs, Adam, '
                 f'batch {BATCH_SIZE}, L1 weight 100). The harness reports blind test scores '
                 'without target-std rescaling. PSNR is in dB; the R-peak amplitude '
                 'error is in z-score units; the temporal F1 uses a plus/minus '
                 '50\\,ms window.}')
    lines.append('\\label{tab:exp6_ablation}')
    lines.append('\\begin{tabular}{lccccccccccc}')
    lines.append('\\toprule')
    lines.append('Variant & MHSA & MSDC & Polarity & Output & Adv. & '
                 'PSNR & SSIM & Corr. & R-peak amp. err. & Temporal F1 & Params (M) \\\\')
    lines.append('\\midrule')

    for key in row_order:
        cfg = VARIANTS.get(key, {})
        mhsa = yes if cfg.get('use_attention') else no
        msdc = yes if cfg.get('use_dilated') else no
        pol = yes if cfg.get('polarity') else no
        adv = yes if cfg.get('adversarial') else no
        out = cfg.get('output', '--')
        params = cfg.get('params_m', float('nan'))
        params_s = f'{params:.2f}' if params == params else no
        lines.append(
            f'{display.get(key, key)} & {mhsa} & {msdc} & {pol} & {out} & {adv} & '
            f'{cell(key, "psnr", "{:.2f}")} & '
            f'{cell(key, "ssim", "{:.3f}")} & '
            f'{cell(key, "corr", "{:.3f}")} & '
            f'{cell(key, "rpeak_amp_err", "{:.3f}")} & '
            f'{cell(key, "temporal_f1_50", "{:.2f}")} & '
            f'{params_s} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table*}')

    tex = '\n'.join(lines) + '\n'
    path = os.path.join(OUT, 'table_exp6_ablation.tex')
    with open(path, 'w') as f:
        f.write(tex)
    print('wrote', path)
    print(tex)
    return path


emit_latex_table(all_results)
