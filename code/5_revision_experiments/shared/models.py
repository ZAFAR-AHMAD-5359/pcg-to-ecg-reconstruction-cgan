"""
Model architectures for the JBHI revision experiments.

Reproduced VERBATIM from the author's COLAB_IEEE_FIGURES.ipynb (paper Table II
baselines) and models.ipynb (ablation trails), so the checkpoints in
config_paths.CKPT load cleanly.

Which class loads which checkpoint
----------------------------------
    CKPT['CNN']      -> CNNEncoderDecoder()        # no scenario, forward(x)
    CKPT['UNet']     -> UNetGenerator()            # pix2pix, forward(x)
    CKPT['WGAN-GP']  -> UNetGenerator()            # same generator as U-Net
    CKPT['Proposed'] -> ImprovedGeneratorNoTanh()  # scenario, forward(x, s_onehot)

The proposed model's forward takes a scenario one-hot; the three baselines take
only the PCG. Every inference script must branch on this (helper `run_generator`
below does it for you).

For the R1.4 component ablation, use `ConfigurableGenerator`, which exposes the
three grouped factors as switches: use_attention, use_dilated, and
output_activation in {'batchnorm','tanh','linear'}.

NOTE ON PARAM COUNTS: the submitted paper's numbers (CNN 9.8M, U-Net 13.6M,
Proposed 14.0M) do NOT match these classes (which give ~larger counts). Always
report count_params(model_loaded_from_checkpoint) computed here -- see R2.2/R2.5.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Shared bottleneck modules
# ===========================================================================
class MultiHeadSelfAttention(nn.Module):
    """8-head self-attention over the temporal axis at the bottleneck."""
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, T)
        qkv = qkv.permute(1, 0, 2, 4, 3)          # (3, B, heads, T, head_dim)
        # .contiguous() is REQUIRED for SDPA to select the flash / memory-efficient
        # kernel (O(T) memory). Without it the permuted views fall back to the MATH
        # kernel, which materializes the T x T score matrix -- that OOM-ed the
        # attention variants even on a 40 GB A100 at batch 48. Mathematically
        # identical to softmax(QK^T/sqrt(d))V, so existing checkpoints are unaffected.
        q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
        p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        out = out.transpose(2, 3).reshape(B, C, T)
        return self.proj(out) + x


class MultiScaleDilatedBlock(nn.Module):
    """Parallel dilated convs at rates 1, 2, 4 with a 1x1 fusion + residual."""
    def __init__(self, channels):
        super().__init__()
        c1 = channels // 3 + (1 if channels % 3 >= 1 else 0)
        c2 = channels // 3 + (1 if channels % 3 >= 2 else 0)
        c3 = channels // 3
        self.conv_d1 = nn.Conv1d(channels, c1, 3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(channels, c2, 3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(channels, c3, 3, padding=4, dilation=4)
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        cat = torch.cat([self.conv_d1(x), self.conv_d2(x), self.conv_d4(x)], dim=1)
        return self.fusion(cat) + x


class EnhancedResidualBlock(nn.Module):
    """Residual block with optional MHSA and MSDC (Trails 2-4)."""
    def __init__(self, channels, use_attention=False, use_dilated=True):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.use_dilated = use_dilated
        if use_dilated:
            self.dilated_block = MultiScaleDilatedBlock(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadSelfAttention(channels, num_heads=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use_dilated:
            out = self.dilated_block(out)
        out = self.bn2(self.conv2(out))
        if self.use_attention:
            out = self.attention(out)
        return self.relu(out + residual)


# ===========================================================================
# Paper Table II baseline: CNN Encoder-Decoder (8-level, no skips, no scenario)
# ===========================================================================
class ConvBlock(nn.Module):
    def __init__(self, i, o, k=4, s=2, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(i, o, k, s, p, bias=False),
            nn.BatchNorm1d(o), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x): return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, i, o, k=4, s=2, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(i, o, k, s, p, bias=False),
            nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class CNNEncoderDecoder(nn.Module):
    """Paper CNN baseline -> loads CKPT['CNN']. forward(x)."""
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        bf = base_filters
        self.enc1 = ConvBlock(in_channels, bf)
        self.enc2 = ConvBlock(bf, bf * 2)
        self.enc3 = ConvBlock(bf * 2, bf * 4)
        self.enc4 = ConvBlock(bf * 4, bf * 8)
        self.enc5 = ConvBlock(bf * 8, bf * 8)
        self.enc6 = ConvBlock(bf * 8, bf * 8)
        self.enc7 = ConvBlock(bf * 8, bf * 8)
        self.enc8 = ConvBlock(bf * 8, bf * 8)
        self.dec1 = DeconvBlock(bf * 8, bf * 8)
        self.dec2 = DeconvBlock(bf * 8, bf * 8)
        self.dec3 = DeconvBlock(bf * 8, bf * 8)
        self.dec4 = DeconvBlock(bf * 8, bf * 8)
        self.dec5 = DeconvBlock(bf * 8, bf * 4)
        self.dec6 = DeconvBlock(bf * 4, bf * 2)
        self.dec7 = DeconvBlock(bf * 2, bf)
        self.final = nn.ConvTranspose1d(bf, out_channels, 4, 2, 1)

    def forward(self, x):
        n = x.size(2)
        e = self.enc1(x); e = self.enc2(e); e = self.enc3(e); e = self.enc4(e)
        e = self.enc5(e); e = self.enc6(e); e = self.enc7(e); e = self.enc8(e)
        d = self.dec1(e); d = self.dec2(d); d = self.dec3(d); d = self.dec4(d)
        d = self.dec5(d); d = self.dec6(d); d = self.dec7(d)
        out = self.final(d)
        if out.size(2) != n:
            diff = n - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))
        return out


# ===========================================================================
# Paper Table II baseline: pix2pix U-Net (shared by U-Net and WGAN-GP)
# ===========================================================================
class UNetBlock(nn.Module):
    def __init__(self, i, o, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv1d(i, o, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm1d(o))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, i, o, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose1d(i, o, 4, 2, 1, bias=False),
                  nn.BatchNorm1d(o), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.block(x)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """Paper U-Net / WGAN-GP generator -> loads CKPT['UNet'] or CKPT['WGAN-GP']. forward(x)."""
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        f = features
        self.e1 = UNetBlock(in_channels, f, normalize=False)
        self.e2 = UNetBlock(f, f * 2); self.e3 = UNetBlock(f * 2, f * 4)
        self.e4 = UNetBlock(f * 4, f * 8); self.e5 = UNetBlock(f * 8, f * 8)
        self.e6 = UNetBlock(f * 8, f * 8); self.e7 = UNetBlock(f * 8, f * 8)
        self.e8 = UNetBlock(f * 8, f * 8, normalize=False)
        self.d1 = UNetUpBlock(f * 8, f * 8, dropout=0.5)
        self.d2 = UNetUpBlock(f * 16, f * 8, dropout=0.5)
        self.d3 = UNetUpBlock(f * 16, f * 8, dropout=0.5)
        self.d4 = UNetUpBlock(f * 16, f * 8)
        self.d5 = UNetUpBlock(f * 16, f * 4); self.d6 = UNetUpBlock(f * 8, f * 2)
        self.d7 = UNetUpBlock(f * 4, f)
        self.final = nn.ConvTranspose1d(f * 2, out_channels, 4, 2, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6); e8 = self.e8(e7)
        d = self.d1(e8, e7); d = self.d2(d, e6); d = self.d3(d, e5); d = self.d4(d, e4)
        d = self.d5(d, e3); d = self.d6(d, e2); d = self.d7(d, e1)
        out = self.final(d)
        if out.size(2) != x.size(2):
            diff = x.size(2) - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))
        return out


# ===========================================================================
# Proposed generator (Trail-4): U-Net + MHSA + MSDC + scenario, BatchNorm output
# ===========================================================================
class ImprovedGeneratorNoTanh(nn.Module):
    """Proposed model -> loads CKPT['Proposed']. forward(x, scenario_onehot)."""
    def __init__(self, signal_length=64000, base_channels=64, num_scenarios=3):
        super().__init__()
        bc = base_channels
        self.scenario_embed = nn.Linear(num_scenarios, bc)
        self.enc1 = nn.Sequential(nn.Conv1d(1, bc, 7, 2, 3), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(bc, bc*2, 5, 2, 2), nn.BatchNorm1d(bc*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(bc*2, bc*4, 5, 2, 2), nn.BatchNorm1d(bc*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(bc*4, bc*8, 5, 2, 2), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.enc5 = nn.Sequential(nn.Conv1d(bc*8, bc*8, 5, 2, 2), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.bottleneck = nn.Sequential(
            EnhancedResidualBlock(bc*8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(bc*8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(bc*8, use_attention=True, use_dilated=True),
            EnhancedResidualBlock(bc*8, use_attention=True, use_dilated=True))
        self.dec5 = nn.Sequential(nn.ConvTranspose1d(bc*8, bc*8, 4, 2, 1), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.dec4 = nn.Sequential(nn.Conv1d(bc*16, bc*8, 1), nn.ConvTranspose1d(bc*8, bc*4, 4, 2, 1), nn.BatchNorm1d(bc*4), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.Conv1d(bc*8, bc*4, 1), nn.ConvTranspose1d(bc*4, bc*2, 4, 2, 1), nn.BatchNorm1d(bc*2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv1d(bc*4, bc*2, 1), nn.ConvTranspose1d(bc*2, bc, 4, 2, 1), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv1d(bc*2, bc, 1), nn.ConvTranspose1d(bc, 1, 4, 2, 1), nn.BatchNorm1d(1))

    def forward(self, x, scenario_onehot):
        s = self.scenario_embed(scenario_onehot).unsqueeze(2)
        e1 = self.enc1(x) + s
        e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = torch.cat([self.dec5(b), e4], dim=1)
        d4 = torch.cat([self.dec4(d5), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        return self.dec1(d2)


# ===========================================================================
# Configurable generator for the R1.4 component ablation
# ===========================================================================
class ConfigurableGenerator(nn.Module):
    """Ablation generator: toggle each grouped factor independently.

    use_attention : add MHSA in the 4 bottleneck blocks
    use_dilated   : add MSDC in the 4 bottleneck blocks
    output_activation : 'batchnorm' (proposed) | 'tanh' | 'linear'
    use_scenario  : condition on the scenario one-hot (keep True to match proposed)

    Ablation rows the reviewer asked for map to:
        U-Net only          -> use_attention=False, use_dilated=False
        U-Net + MSDC        -> use_attention=False, use_dilated=True
        U-Net + MHSA        -> use_attention=True,  use_dilated=False
        U-Net + both        -> use_attention=True,  use_dilated=True
        Proposed + Tanh     -> both True, output_activation='tanh'
        Proposed + linear   -> both True, output_activation='linear'
        Proposed (BatchNorm)-> both True, output_activation='batchnorm'
    Polarity-on/off is a DATA-loader switch; adversarial-on/off is a TRAINING
    switch -- neither is an architecture change, so control them in the trainer.
    """
    def __init__(self, base_channels=64, num_scenarios=3,
                 use_attention=True, use_dilated=True,
                 output_activation="batchnorm", use_scenario=True):
        super().__init__()
        bc = base_channels
        self.use_scenario = use_scenario
        if use_scenario:
            self.scenario_embed = nn.Linear(num_scenarios, bc)
        self.enc1 = nn.Sequential(nn.Conv1d(1, bc, 7, 2, 3), nn.BatchNorm1d(bc), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv1d(bc, bc*2, 5, 2, 2), nn.BatchNorm1d(bc*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv1d(bc*2, bc*4, 5, 2, 2), nn.BatchNorm1d(bc*4), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv1d(bc*4, bc*8, 5, 2, 2), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.enc5 = nn.Sequential(nn.Conv1d(bc*8, bc*8, 5, 2, 2), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.bottleneck = nn.Sequential(*[
            EnhancedResidualBlock(bc*8, use_attention=use_attention, use_dilated=use_dilated)
            for _ in range(4)])
        self.dec5 = nn.Sequential(nn.ConvTranspose1d(bc*8, bc*8, 4, 2, 1), nn.BatchNorm1d(bc*8), nn.ReLU(True))
        self.dec4 = nn.Sequential(nn.Conv1d(bc*16, bc*8, 1), nn.ConvTranspose1d(bc*8, bc*4, 4, 2, 1), nn.BatchNorm1d(bc*4), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.Conv1d(bc*8, bc*4, 1), nn.ConvTranspose1d(bc*4, bc*2, 4, 2, 1), nn.BatchNorm1d(bc*2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv1d(bc*4, bc*2, 1), nn.ConvTranspose1d(bc*2, bc, 4, 2, 1), nn.BatchNorm1d(bc), nn.ReLU(True))
        head = [nn.Conv1d(bc*2, bc, 1), nn.ConvTranspose1d(bc, 1, 4, 2, 1)]
        if output_activation == "batchnorm":
            head.append(nn.BatchNorm1d(1))
        elif output_activation == "tanh":
            head.append(nn.Tanh())
        elif output_activation == "linear":
            pass
        else:
            raise ValueError(f"unknown output_activation {output_activation}")
        self.dec1 = nn.Sequential(*head)

    def forward(self, x, scenario_onehot=None):
        e1 = self.enc1(x)
        if self.use_scenario and scenario_onehot is not None:
            e1 = e1 + self.scenario_embed(scenario_onehot).unsqueeze(2)
        e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = torch.cat([self.dec5(b), e4], dim=1)
        d4 = torch.cat([self.dec4(d5), e3], dim=1)
        d3 = torch.cat([self.dec3(d4), e2], dim=1)
        d2 = torch.cat([self.dec2(d3), e1], dim=1)
        return self.dec1(d2)


# ===========================================================================
# Discriminator (for R1.4 adversarial retraining)
# ===========================================================================
class SpectralNormDiscriminator(nn.Module):
    """Spectrally-normalized PatchGAN critic with scenario conditioning."""
    def __init__(self, base_channels=64, num_scenarios=3):
        super().__init__()
        bc = base_channels
        self.scenario_embed = nn.Linear(num_scenarios, bc)
        sn = nn.utils.spectral_norm
        self.conv1 = sn(nn.Conv1d(2, bc, 7, 2, 3))
        self.conv2 = sn(nn.Conv1d(bc, bc*2, 5, 2, 2)); self.bn2 = nn.BatchNorm1d(bc*2)
        self.conv3 = sn(nn.Conv1d(bc*2, bc*4, 5, 2, 2)); self.bn3 = nn.BatchNorm1d(bc*4)
        self.conv4 = sn(nn.Conv1d(bc*4, bc*8, 5, 2, 2)); self.bn4 = nn.BatchNorm1d(bc*8)
        self.conv5 = sn(nn.Conv1d(bc*8, bc*8, 5, 2, 2)); self.bn5 = nn.BatchNorm1d(bc*8)
        self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(bc*8, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, pcg, ecg, scenario_onehot):
        x = torch.cat([pcg, ecg], dim=1)
        s = self.scenario_embed(scenario_onehot).unsqueeze(2)
        x = self.lrelu(self.conv1(x)) + s
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        return torch.sigmoid(self.fc(self.pool(x).squeeze(2)))


# ===========================================================================
# WGAN-GP critic (for retraining the WGAN-GP baseline; extracted verbatim from
# the original 05_GAN_WGAN_GP notebook). No BatchNorm, no sigmoid -> a valid
# WGAN critic. The generator is the plain UNetGenerator above.
# ===========================================================================
class WGANCritic(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features, features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features * 2, features * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features * 4, features * 8, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features * 8, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


def compute_gradient_penalty(critic, pcg, real_ecg, fake_ecg, device):
    """WGAN-GP gradient penalty. Interpolates the ECG only; PCG conditions."""
    bs = real_ecg.size(0)
    alpha = torch.rand(bs, 1, 1, device=device)
    interp = (alpha * real_ecg + (1 - alpha) * fake_ecg).requires_grad_(True)
    c_interp = critic(pcg, interp)
    grads = torch.autograd.grad(
        outputs=c_interp, inputs=interp,
        grad_outputs=torch.ones_like(c_interp),
        create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ===========================================================================
# Loading + utility helpers
# ===========================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model, checkpoint_path, device="cpu", strict=False):
    """Load a checkpoint into `model`, handling common save formats."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("generator_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]; break
    state = {k.replace("module.", ""): v for k, v in ckpt.items()} \
            if isinstance(ckpt, dict) else ckpt
    result = model.load_state_dict(state, strict=strict)
    # Loud guard: a large key mismatch means the checkpoint does not fit this
    # architecture (e.g. a Trail-2/trial2 file loaded into the Trail-4 model).
    n_model = max(len(model.state_dict()), 1)
    n_missing, n_unexpected = len(result.missing_keys), len(result.unexpected_keys)
    tag = os.path.basename(checkpoint_path)
    if n_missing or n_unexpected:
        flag = "  <-- LIKELY WRONG CHECKPOINT FOR THIS MODEL" if n_missing > 0.10 * n_model else ""
        print(f"[load_model] {tag}: {n_missing}/{n_model} params missing, "
              f"{n_unexpected} unexpected{flag}")
    else:
        print(f"[load_model] {tag}: exact match ({n_model} tensors).")
    model.to(device).eval()
    return model, result


def build_paper_models(ckpt_dict, device="cpu"):
    """Load all four paper models. Returns {name: model}."""
    models = {}
    models["CNN"], _     = load_model(CNNEncoderDecoder(), ckpt_dict["CNN"], device)
    models["UNet"], _    = load_model(UNetGenerator(), ckpt_dict["UNet"], device)
    models["WGAN-GP"], _ = load_model(UNetGenerator(), ckpt_dict["WGAN-GP"], device)
    models["Proposed"], _= load_model(ImprovedGeneratorNoTanh(), ckpt_dict["Proposed"], device)
    return models


def run_generator(model, pcg_tensor, scenario_tensor, is_proposed):
    """Uniform inference call. `is_proposed` selects the (x, s) signature."""
    with torch.no_grad():
        if is_proposed:
            out = model(pcg_tensor, scenario_tensor)
        else:
            out = model(pcg_tensor)
    return out.squeeze().cpu().numpy()
