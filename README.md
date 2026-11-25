# PCG-to-ECG Signal Reconstruction Using Conditional GAN with Multi-Scale Attention

Code, trained weights, data split, and outputs for the paper

> Z. Ahmad, N. Perwaiz, M. S. Khan, S. Sultana, F. Abdullah, H. Anjum, and I. Haq,
> "PCG-to-ECG Signal Reconstruction Using Conditional GAN with Multi-Scale Attention,"
> *IEEE Journal of Biomedical and Health Informatics*, 2026 (accepted, manuscript JBHI-01557-2026).

A conditional GAN maps an 8 s phonocardiogram (PCG, heart sound) segment to the simultaneous
single-lead ECG. The generator is a U-Net with multi-head self-attention and multi-scale dilated
convolutions, with a spectrally normalized PatchGAN discriminator. On 453 test segments from 3
held-out EPHNOGRAM subjects, the model reaches 17.10 dB PSNR, 0.903 SSIM, and 0.610 correlation,
and R-peak detection on the reconstructed ECG reaches F1 0.694 at ±50 ms, against 0.404 for
PCG-only timing. The output gives approximate cardiac-cycle timing on matched recording hardware.
It is not a diagnostic ECG, and it does not transfer to new hardware without retraining.

## Contents

| Folder | What it holds |
|---|---|
| `code/0_quickstart/` | `evaluate_checkpoint.py`: scores a released checkpoint on the test segments and compares each one with the paper's run |
| `code/1_preprocessing/` | Notebook (as run) that cuts the PhysioNet EPHNOGRAM records into 8 s segments and writes the subject-level split |
| `code/2_warm_start_trial2/` | Notebook (as run) of the Trial-2 development run, whose final generator initializes the proposed model |
| `code/3_proposed_model/` | Training (`TRAIL4_COMPLETE.ipynb`) and test evaluation (`TRAIL4_TEST_EVALUATION.ipynb`) of the proposed model |
| `code/4_original_baselines/` | Notebooks (as run) for the CNN, U-Net, and WGAN-GP rows in the upper block of Table II |
| `code/5_revision_experiments/` | Experiments exp1 to exp7: Python sources (`src/`), Colab notebooks built from them (`notebooks/`), and the shared model, metric, and path modules (`shared/`) |
| `code/6_supplementary_analyses/` | Preprocessing, metric, model, Tanh-gradient, and cross-dataset notebooks of the supplementary material |
| `code/7_paper_figures/` | Script that draws Fig. 4 from the saved signals |
| `splits/` | `train_val_test_split.json` (subjects), `test_segments.csv` (the 453 test segments), `segment_processing_log.csv` (record, subject, scenario, and segment count) |
| `results/` | Outputs of every run behind the paper: per-segment metrics, summaries, LaTeX tables, figures, training histories, and the executed notebooks (`notebook_as_run.ipynb`) |
| `weights/` | List of the trained checkpoints with SHA-256 hashes and loading code; the files are attached to the [v1.0.0 release](https://github.com/ZAFAR-AHMAD-5359/pcg-to-ecg-reconstruction-cgan/releases/tag/v1.0.0) |

## Data

The recordings are not redistributed here. Download them from PhysioNet:

- EPHNOGRAM (training and in-domain test, 8 kHz): https://physionet.org/content/ephnogram/1.0.0/
- SensSmart-Tech (blind cross-dataset test, 1 kHz): https://physionet.org/content/senssmarttech/1.0.0/

`code/1_preprocessing/01_preprocessing_clean_chunks.ipynb` writes 5,193 segments of 64,000 samples
as `.npz` files with keys `ecg`, `pcg`, `subject_id`, and `scenario`. The paper's code reads them
from a folder named `Refine-Chunked`. The split is by subject:

| Set | Subjects | Segments |
|---|---|---|
| Train | S001, S005, S006, S007, S008, S010, S011, S012, S013, S014, S015, S016, S017, S020 | 4,059 |
| Validation | S002, S003, S018 | 681 |
| Test | S009, S019, S023 | 453 |

## Quick check

```bash
pip install -r requirements.txt
# download trail4_best.pt from the v1.0.0 release into weights/
python code/0_quickstart/evaluate_checkpoint.py --model proposed \
    --checkpoint weights/trail4_best.pt --chunks path/to/Refine-Chunked
# Proposed: 453 segments | PSNR 17.097 dB | correlation 0.6101
```

`--model cnn|unet|wgan_gp` scores the retrained baselines the same way. Without the data,
`proposed_test_reconstructions.npz` in the release holds the model's raw output for all 453 test
segments.

## Where each result comes from

| Paper item | Code | Outputs |
|---|---|---|
| Table II, proposed model | `code/3_proposed_model/` | `results/table2_proposed_trail4/` |
| Table II, upper block (CNN, U-Net, WGAN-GP) | `code/4_original_baselines/` | `results/table2_original_baselines/` |
| Table II, lower block, and the paired Wilcoxon tests | `exp7_baseline_retrain` | `results/exp7_baseline_retrain/` |
| Temporal R-peak detection (F1, sensitivity, PPV) | `exp1_R1_1_rpeak_timing` | `results/exp1_rpeak_timing/` |
| Per-subject correlation and the subject-clustered bootstrap | `exp2_R1_2_baseline_stats` | `results/exp2_baseline_stats/` |
| Cardiac-cycle boundary error | `exp3_R1_7_boundary_segmentation` | `results/exp3_boundary/` |
| Blind cross-dataset test (SensSmart-Tech) | `exp4_R1_8_blind_crossdataset` | `results/exp4_blind_crossdataset/` |
| Parameters, MACs, and latency | `exp5_R2_2_complexity_latency` | `results/exp5_complexity/` |
| Component ablation (Table V) | `exp6_R1_4_ablation_training` | `results/exp6_ablation/` |
| Fig. 4 | `code/7_paper_figures/regenerate_fig4.py` | |

The `exp*` notebooks are in `code/5_revision_experiments/notebooks/`. Set the Google Drive paths in
`shared/config_paths.py` before running them on Colab. exp4 has no saved executed notebook; its
outputs are in `results/exp4_blind_crossdataset/`.

## Training history and caveats

- **Warm start.** The proposed model was not trained from a random initialization. Its training
  notebook (`TRAIL4_COMPLETE.ipynb`) loads the final generator of the Trial-2 development run,
  which was trained for 150 epochs on the same subject split with a Tanh output layer. It then
  fine-tunes this generator with the BatchNorm output head and polarity correction, and keeps the
  checkpoint with the best validation PSNR. That checkpoint is from the first fine-tuning epoch,
  as stored in `trail4_best.pt` (`epoch: 1`) and in `results/table2_proposed_trail4/trail4_training_history.json`.
  The release includes the Trial-2 generator weights (`trial2_warm_start_generator_epoch150.pt`).
  The ablation of Table V trains every variant, including the proposed configuration, from a
  random initialization; the proposed configuration reaches 17.58 dB there.
- **Original baselines.** The checkpoints behind the upper block of Table II were not kept. The
  U-Net values in that block come from a 10-segment test run (`test_samples: 10` in
  `results/table2_original_baselines/UNet_Baseline/test_results.json`); the CNN and WGAN-GP values
  use all 453 segments. The retrained baselines in the release are scored on all 453 segments.
- **Ablation, proposed row.** The variant with both attention and dilated convolutions was trained
  twice. Table V reports the run in `results/exp6_ablation/all_results.json` (17.58 dB, correlation
  0.544). The released `ablation_both.pt` is from the other run (`metrics_both.json`: 17.59 dB,
  correlation 0.552); the weights of the Table V run were overwritten.

## Metric conventions

- PSNR, SNR, Pearson correlation, RMSE, and MAE use the z-score-normalized reconstruction and
  reference, with no rescaling of the reconstruction. PSNR = 20 log10(max|reference| / RMSE).
- SSIM normalizes each signal to [0, 1] by its own range first.
- In Table II, the R-peak amplitude error and the QRS-width error rescale the reconstruction to the
  reference mean and standard deviation, read the amplitude at the ground-truth R-peaks, and average
  over the 301 segments with detected R-peaks. The ablation and exp7 tables report the R-peak
  amplitude error in z-score units, so values from different tables are not comparable.
- The ECG target is polarity-corrected (R-peaks made positive) before normalization. The PCG input
  needs only per-segment z-scoring.

## Environment

The models were trained on Google Colab GPUs (T4, A100) with Python 3.10 and PyTorch 2.x. The
evaluation code also runs on a CPU. Package list: `requirements.txt`.

## License

Code: MIT License (`LICENSE`). Trained weights and result files: CC BY 4.0. The weights were
trained on EPHNOGRAM; please also cite the dataset:
A. Kazemnejad, P. Gordany, and R. Sameni, "EPHNOGRAM: A simultaneous electrocardiogram and
phonocardiogram database," PhysioNet, 2021.

## Citation

See `CITATION.cff`. The DOI will be added when the article is published.
