"""Score a released checkpoint on the EPHNOGRAM test segments and compare with the paper.

Example (proposed model, all 453 test segments):
    python code/0_quickstart/evaluate_checkpoint.py --model proposed \
        --checkpoint weights/trail4_best.pt --chunks path/to/Refine-Chunked

--chunks is the folder of 8 s .npz segments written by
code/1_preprocessing/01_preprocessing_clean_chunks.ipynb from the PhysioNet EPHNOGRAM files.
Each segment is compared with the per-segment scores in
results/exp7_baseline_retrain/exp7_persegment_metrics.csv (same metric code as the paper).
Runs on a CPU in about 1.5 s per segment; use --limit for a quick check.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code", "5_revision_experiments", "shared"))
import config_paths  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

BUILD = {
    "proposed": (models.ImprovedGeneratorNoTanh, "Proposed"),
    "cnn": (models.CNNEncoderDecoder, "CNN"),
    "unet": (models.UNetGenerator, "UNet"),
    "wgan_gp": (models.UNetGenerator, "WGAN-GP"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(BUILD), default="proposed")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chunks", required=True, help="folder with ECGPCGxxxx_chunkNNN.npz segments")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N test segments")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    cls, label = BUILD[a.model]
    model, _ = models.load_model(cls(), a.checkpoint, device="cpu")
    is_prop = a.model == "proposed"

    with open(os.path.join(ROOT, "splits", "test_segments.csv")) as f:
        segments = [r["segment"] for r in csv.DictReader(f)]
    if a.limit:
        segments = segments[: a.limit]
    ref_rows = {}
    with open(os.path.join(ROOT, "results", "exp7_baseline_retrain", "exp7_persegment_metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["model"] == label:
                ref_rows[r["chunk"].replace(".npz", "")] = (float(r["psnr"]), float(r["corr"]))

    got, diffs = [], []
    for s in segments:
        d = np.load(os.path.join(a.chunks, s + ".npz"), allow_pickle=True)
        ref = metrics.normalize_zscore(metrics.ensure_positive_polarity(d["ecg"].astype(np.float32)))
        pcg = metrics.normalize_zscore(d["pcg"].astype(np.float32))
        scen = torch.zeros(1, config_paths.NUM_SCENARIOS)
        scen[0, config_paths.scenario_to_index(d["scenario"])] = 1.0
        fake = models.run_generator(model, torch.from_numpy(pcg)[None, None], scen, is_prop)
        p, c = float(metrics.calc_psnr(ref, fake)), float(metrics.calc_correlation(ref, fake))
        got.append((p, c))
        if s in ref_rows:
            diffs.append(max(abs(p - ref_rows[s][0]), abs(c - ref_rows[s][1])))
    got = np.array(got)
    print(f"{label}: {len(got)} segments | PSNR {got[:, 0].mean():.3f} dB | correlation {got[:, 1].mean():.4f}")
    if diffs:
        print(f"largest per-segment difference from the paper's run: {max(diffs):.1e} "
              "(GPU vs CPU rounding is about 1e-3)")


if __name__ == "__main__":
    main()
