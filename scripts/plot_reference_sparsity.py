"""Plot saved reference hidden-layer spike sparsity; no model evaluation."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    for dataset, label, color, marker in [
        ("SHD", "SHD* (3 seeds)", "#2563eb", "o"),
        ("SSC", "SSC (3 seeds)", "#d97706", "s"),
        ("ECG", "ECG (3 seeds)", "#15803d", "^")
    ]:
        rows = []
        for seed in [42, 123, 456]:
            path = ROOT / f"results/valid/reference/{dataset}/seed_{seed}/COST_RESULTS.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            totals = record["measurements"]["totals"]
            row = []
            for layer in ["l1", "l2"]:
                positions = totals["sequence_sample_timesteps"] * record["saved_config"][layer]["n_neurons"]
                row.append(100 * (1 - totals[f"{layer}_spikes_sequence"] / positions))
            rows.append(row)
        means = [statistics.mean(values) for values in zip(*rows)]
        errors = [statistics.stdev(values) for values in zip(*rows)] if len(rows) > 1 else None
        ax.errorbar([1, 2], means, yerr=errors, label=label, color=color,
                    marker=marker, linewidth=2, markersize=7, capsize=5)
        for x, y in zip([1, 2], means):
            ax.annotate(f"{y:.2f}%", (x, y), xytext=(9 if x == 1 else -9,
                                    -18 if (dataset == "SSC" and x == 1) or (dataset == "SHD" and x == 2) else 9),
                        textcoords="offset points", ha="left" if x == 1 else "right",
                        color=color, fontsize=10)
    ax.set(title="Reference model: hidden-layer spike sparsity",
           xlabel="Layer number", ylabel="Spike sparsity (%)", xlim=(0.8, 2.2), ylim=(0, 100))
    ax.set_xticks([1, 2])
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.supxlabel("Error bars: sample SD across three seeds.\nStored sequence includes minimum padding and burn-in; batch-added padding excluded.",
                  fontsize=8)
    for extension in ["png", "svg"]:
        target = output / f"reference-layer-sparsity.{extension}"
        fig.savefig(target, dpi=200)
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                              target.read_text(encoding="utf-8").splitlines()) + "\n",
                              encoding="utf-8")
    plt.close(fig)

if __name__ == "__main__":
    main()
