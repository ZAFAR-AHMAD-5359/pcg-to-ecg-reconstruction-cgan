# Trained weights

The files are attached to the GitHub release
[v1.0.0](https://github.com/ZAFAR-AHMAD-5359/pcg-to-ecg-reconstruction-cgan/releases/tag/v1.0.0).
Download them into this folder. Check a file with `sha256sum <file>` (Linux, macOS) or
`certutil -hashfile <file> SHA256` (Windows).

| File | Model | Paper item | MB | SHA-256 |
|---|---|---|---|---|
| `trail4_best.pt` | Proposed generator (19.3 M parameters) and discriminator | Table II, proposed; all exp1 to exp5 and exp7 results | 86.2 | `284a5b43e5f0da53b28f80a1f13452eea7e16b52390d195eb2f9b6df829140ab` |
| `trial2_warm_start_generator_epoch150.pt` | Trial-2 generator after epoch 150 (Tanh output), the starting point of the proposed model | Training history | 77.4 | `7b04862a62d9c20e80ffeca23018764c23cb76c482bdf56424bbecc009676559` |
| `baseline_retrained_CNN_EncoderDecoder.pt` | CNN encoder-decoder, retrained | Table II lower block, paired tests | 39.2 | `35b599ae78ebd2096aeb676aeef4546c6147eb3d3d5c1c97e95099cf695d995d` |
| `baseline_retrained_UNet.pt` | U-Net, retrained | Table II lower block, paired tests | 54.5 | `2c87320fe612d370d8cccbca9e04c94e25d9534fb6c79603f28b38002d0e7606` |
| `baseline_retrained_WGAN_GP.pt` | WGAN-GP generator, retrained | Table II lower block, paired tests | 54.5 | `98ba8f321721868b259a5cf128a66bb960274f9661ee5303e35a2f6424ea4938` |
| `ablation_unet_only.pt` | U-Net only | Table V | 43.8 | `108ab018bc773c085dd1ee4acc6da9b5c2366717cdaa5a0a903b1d25fd91bd67` |
| `ablation_msdc.pt` | U-Net + MSDC | Table V | 60.6 | `7cd6a35a5452bceebe6b0a94fd383152793f41a18925a5f92a13f55481574cfd` |
| `ablation_mhsa.pt` | U-Net + MHSA | Table V | 60.6 | `a315aa87c4b1bb5e108582bf98dc9f6348f9844d19a041017b21542af7c87023` |
| `ablation_both.pt` | U-Net + MHSA + MSDC (proposed configuration) | Table V (see note) | 77.4 | `c1ccf1b93f24ba70f14a4c242801170fdd7396b123723ab42e9bfad5d321aead` |
| `ablation_proposed_tanh.pt` | Proposed, Tanh output | Table V | 77.4 | `dc5003be9b2d7650044d1a0ca29b9762d856f9c5d07420f172a2315792d1d5a1` |
| `ablation_proposed_linear.pt` | Proposed, linear output | Table V | 77.4 | `2d784b05a9e5e8f8d35dca3434b40f2cd4afcd47da543ac2ba06d1ac720873e8` |
| `ablation_adv_off.pt` | Proposed, no adversarial loss | Table V | 77.4 | `156e15c1d734fcfdb3ffc9022e87a568f8463f7b680bd6f45b6f59e5280966d2` |
| `ablation_polarity_off.pt` | Proposed, no polarity correction | Table V | 77.4 | `e7eec23c2c849233d138cef39179df71906a021fd25ef67694691f5f3dd71079` |
| `proposed_test_reconstructions.npz` | Raw output of `trail4_best.pt` on the 453 test segments (float32, 64,000 samples each, keyed by segment name) | exp1, exp3 | 106.7 | `975ee80528908d7a001c2d564bdf22ab39b7eb3b0fe366da8a517f27cb3aaf5e` |

Note on `ablation_both.pt`: this variant was trained twice. The file is from the run in
`results/exp6_ablation/metrics_both.json` (17.59 dB, correlation 0.552 over 453 segments, checked
again on a CPU). Table V reports the other run (17.58 dB, correlation 0.544), whose weights were
overwritten.

## Checks done before release (25 Nov 2026, CPU)

- `trail4_best.pt` reproduces `proposed_test_reconstructions.npz` (largest difference 3e-5).
- `trail4_best.pt` and the three retrained baselines reproduce the per-segment PSNR and correlation
  in `results/exp7_baseline_retrain/exp7_persegment_metrics.csv` to within 2e-3 (GPU vs CPU
  rounding).
- Every ablation file loads into its configuration with `strict=True`.

## Loading

```python
import sys, torch
sys.path.insert(0, "code/5_revision_experiments/shared")
import models

# Proposed model and retrained baselines
gen, _ = models.load_model(models.ImprovedGeneratorNoTanh(), "weights/trail4_best.pt")
cnn, _ = models.load_model(models.CNNEncoderDecoder(), "weights/baseline_retrained_CNN_EncoderDecoder.pt")
unet, _ = models.load_model(models.UNetGenerator(), "weights/baseline_retrained_UNet.pt")
wgan, _ = models.load_model(models.UNetGenerator(), "weights/baseline_retrained_WGAN_GP.pt")

# Ablation variants: the generator is stored under "gen", its settings under "config"
ck = torch.load("weights/ablation_both.pt", map_location="cpu", weights_only=False)
c = ck["config"]
abl = models.ConfigurableGenerator(base_channels=64, num_scenarios=3,
                                   use_attention=c["use_attention"], use_dilated=c["use_dilated"],
                                   output_activation=c["output"], use_scenario=True)
abl.load_state_dict(ck["gen"], strict=True)
abl.eval()
```

Input: one 8 s PCG segment at 8 kHz (64,000 samples), z-scored, shape `(1, 1, 64000)`, plus a
scenario one-hot of length 3 (0 rest, 1 exercise, 2 other) for the proposed and ablation
generators: `models.run_generator(gen, pcg, scenario, is_proposed=True)`. The baselines take the
PCG only (`is_proposed=False`).
