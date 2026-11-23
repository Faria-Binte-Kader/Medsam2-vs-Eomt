"""Learning curve of fine-tuned EoMT (test split, validation-chosen checkpoints) against the MedSAM2 reference.

  python analyze_eomt.py --out-dir figures

Reads results/eomt/sweep_*/eval_test/volume_metrics.csv and the committed MedSAM2 test run, and writes
  eomt_learning_curve.csv       per training size: per-patient Dice mean, bootstrap 95% CI, runs pooled, spread across orderings
  eomt_by_size_bin.csv          Dice by predeclared tumor-size bin (mL)
  eomt_vs_medsam2_per_patient.csv  per-patient test Dice, EoMT (all-data run) vs MedSAM2 exact/mild/strong
  fig_learning_curve.png        Dice vs number of training patients; MedSAM2 exact/mild/strong as reference bands
Sizes with several orderings (1 and 2) report the mean over orderings (each ordering's mean first), plus the min-max range.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import metrics as M

SIZE_ORDER = ["1", "2", "5", "10", "20", "all"]


def load_runs(eomt_dir):
    frames = []
    for d in sorted(Path(eomt_dir).glob("sweep_*")):
        f = d / "eval_test" / "volume_metrics.csv"
        if f.exists():
            v = pd.read_csv(f)
            v["run"] = d.name
            v["n_train"] = str(v["n_train_patients"].iloc[0])
            if "order_seed" not in v:
                v["order_seed"] = 0
            frames.append(v)
    if not frames:
        raise SystemExit(f"no test results under {eomt_dir}/sweep_*/eval_test")
    return pd.concat(frames, ignore_index=True)


def curve(vol):
    rows = []
    for n in SIZE_ORDER:
        g = vol[vol.n_train == n]
        if g.empty:
            continue
        per_run = g.groupby("run").dice.mean()
        pooled = g.groupby("patient_id").dice.mean()  # mean over orderings, then over patients
        lo, hi = M.bootstrap_ci(pooled.values, C.BOOTSTRAP["n"], C.BOOTSTRAP["seed"])
        rows.append({"n_train_patients": n, "n_runs": g.run.nunique(), "n_test_patients": pooled.size,
                     "dice_mean": pooled.mean(), "dice_ci95_lo": lo, "dice_ci95_hi": hi,
                     "run_min": per_run.min(), "run_max": per_run.max(), "iou_mean": g.groupby("patient_id").iou.mean().mean()})
    return pd.DataFrame(rows)


def by_bin(vol):
    rows = []
    for n in SIZE_ORDER:
        g = vol[vol.n_train == n]
        for b, gb in g.groupby("size_bin"):
            p = gb.groupby("patient_id").dice.mean()
            rows.append({"n_train_patients": n, "size_bin": b, "n_patients": p.size, "dice_mean": p.mean()})
    return pd.DataFrame(rows)


def medsam2_reference(path):
    m = pd.read_csv(path)
    out = {}
    for c, g in m.groupby("condition"):
        p = g.groupby("patient_id").dice.mean()
        lo, hi = M.bootstrap_ci(p.values, C.BOOTSTRAP["n"], C.BOOTSTRAP["seed"])
        out[c] = (p.mean(), lo, hi)
    return m, out


def figure(cv, ref, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(cv))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(x, cv.dice_mean, yerr=[cv.dice_mean - cv.dice_ci95_lo, cv.dice_ci95_hi - cv.dice_mean], marker="o", capsize=3,
                color="#1f77b4", label="EoMT fine-tuned (automatic), mean and bootstrap 95% CI")
    multi = cv[cv.n_runs > 1]
    ax.scatter(multi.index, multi.run_min, marker="v", color="#1f77b4", alpha=0.5, label="EoMT, worst / best ordering")
    ax.scatter(multi.index, multi.run_max, marker="^", color="#1f77b4", alpha=0.5)
    for (c, (mu, lo, hi)), col in zip(ref.items(), ["#2ca02c", "#ff7f0e", "#d62728"]):
        ax.axhline(mu, color=col, ls="--", lw=1.2, label=f"MedSAM2 zero-shot, {c} box")
        ax.axhspan(lo, hi, color=col, alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels([("68" if n == "all" else n) for n in cv.n_train_patients])
    ax.set_xlabel("labeled training patients for EoMT")
    ax.set_ylabel("test 3D Dice (per-patient mean)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Automatic EoMT vs prompted MedSAM2 (14 test patients)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eomt-dir", default=str(C.OUTPUT_DIR / "eomt"))
    ap.add_argument("--medsam2-csv", default="reports/medsam2/test-unbounded/volume_metrics.csv")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vol = load_runs(a.eomt_dir)
    cv, bb = curve(vol), by_bin(vol)
    msam, ref = medsam2_reference(a.medsam2_csv)
    cv.to_csv(out / "eomt_learning_curve.csv", index=False)
    bb.to_csv(out / "eomt_by_size_bin.csv", index=False)
    full = vol[(vol.n_train == "all")].groupby("patient_id").dice.mean().rename("eomt_all_data")
    ms = msam.pivot_table(index="patient_id", columns="condition", values="dice")
    pd.concat([full, ms], axis=1).round(4).to_csv(out / "eomt_vs_medsam2_per_patient.csv")
    figure(cv, ref, out / "fig_learning_curve.png")
    pd.set_option("display.width", 200)
    print(cv.round(3).to_string(index=False))
    print("\nMedSAM2 reference (same test patients):", {c: tuple(round(v, 3) for v in t) for c, t in ref.items()})
    print("\nBy tumor size (EoMT Dice):")
    print(bb.pivot(index="n_train_patients", columns="size_bin", values="dice_mean").reindex(SIZE_ORDER).round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
