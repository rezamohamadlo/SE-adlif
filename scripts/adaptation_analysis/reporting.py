"""Descriptive paired validation reports for the frozen-checkpoint pilot.

Training seeds and perturbation replicates remain separate. These reports do
not estimate a final-test improvement or select an architecture.
"""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


FEATURES = (
    ("mean_rate_hz", "Mean input rate (Hz)"),
    ("fast_slow_contrast", "Fast/slow rate contrast"),
    ("autocorr_tau_ms", "Input ACF crossing lag (ms)"),
    ("active_channel_fraction", "Active channel fraction"),
    ("duration_ms", "Physical frame support (ms)"),
)

SEED_METRICS = (
    "parent_mean_loss", "variant_mean_loss", "mean_loss_benefit",
    "parent_accuracy", "variant_accuracy", "accuracy_difference_pp",
)


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pearson(rows, feature):
    pairs = [(float(row[feature]), float(row["benefit"])) for row in rows
             if _finite(row.get(feature)) and _finite(row.get("benefit"))]
    result = {"feature": feature, "n_finite_pairs": len(pairs),
              "pearson_r": None, "undefined_reason": None}
    if len(pairs) < 3:
        result["undefined_reason"] = "fewer_than_three_finite_pairs"
        return result
    values = np.asarray(pairs, dtype=np.float64)
    if np.ptp(values[:, 0]) == 0 or np.ptp(values[:, 1]) == 0:
        result["undefined_reason"] = "constant_feature_or_benefit"
        return result
    result["pearson_r"] = float(np.corrcoef(values.T)[0, 1])
    return result


def _summarize(rows, pc, metadata):
    """Pure reductions; no pooling over seeds or perturbation replicates."""
    if not rows:
        raise ValueError("Cannot report an empty paired validation result")
    grouped = defaultdict(list)
    clean = defaultdict(list)
    seen = set()
    required = ("seed", "condition", "replicate", "sample_id", "parent_loss",
                "variant_loss", "benefit", "parent_correct", "variant_correct",
                "correctness_difference")
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Missing paired-result fields: {missing}")
        key = (row["seed"], row["condition"], row["replicate"])
        identity = key + (row["sample_id"],)
        if identity in seen:
            raise ValueError(f"Duplicate paired sample within seed/condition/replicate: {identity}")
        seen.add(identity)
        for name in ("parent_loss", "variant_loss", "benefit"):
            if not _finite(row[name]):
                raise ValueError(f"Nonfinite {name} in paired validation record")
        benefit = float(row["parent_loss"]) - float(row["variant_loss"])
        if not math.isclose(float(row["benefit"]), benefit, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Loss benefit must be parent_loss minus variant_loss")
        if row["parent_correct"] not in (0, 1) or row["variant_correct"] not in (0, 1):
            raise ValueError("Correctness must be a Boolean or zero/one")
        difference = int(row["variant_correct"]) - int(row["parent_correct"])
        if row["correctness_difference"] != difference:
            raise ValueError("Correctness difference must be variant minus parent")
        grouped[key].append(row)
        if str(row["condition"]).lower() == "clean":
            clean[row["seed"]].append(row)

    summaries = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]), str(item[2]))):
        group = grouped[key]
        parent_loss = float(np.mean([row["parent_loss"] for row in group]))
        variant_loss = float(np.mean([row["variant_loss"] for row in group]))
        parent_acc = float(np.mean([row["parent_correct"] for row in group]))
        variant_acc = float(np.mean([row["variant_correct"] for row in group]))
        summaries.append(dict(seed=key[0], condition=key[1], replicate=key[2],
                              n_validation_samples=len(group), parent_mean_loss=parent_loss,
                              variant_mean_loss=variant_loss,
                              mean_loss_benefit=parent_loss-variant_loss,
                              parent_accuracy=parent_acc, variant_accuracy=variant_acc,
                              accuracy_difference_pp=100*(variant_acc-parent_acc)))

    # Every paired training seed has equal weight. Perturbation replicates
    # remain distinct groups, never additional training seeds.
    present_cohort = {group["seed"] for group in summaries}
    cohort = set(metadata.get("seeds", present_cohort))
    if not present_cohort.issubset(cohort):
        raise ValueError("Reported training seeds differ from the metadata seed cohort")
    across_seeds = defaultdict(list)
    for group in summaries:
        across_seeds[(group["condition"], group["replicate"])].append(group)
    aggregates = []
    for key in sorted(across_seeds, key=lambda item: (str(item[0]), str(item[1]))):
        seed_groups = across_seeds[key]
        present = {group["seed"] for group in seed_groups}
        n_seeds = len(present)
        metrics = {}
        for metric in SEED_METRICS:
            values = np.asarray([group[metric] for group in seed_groups], dtype=np.float64)
            metrics[metric] = dict(mean=float(np.mean(values)),
                                   sample_sd=float(np.std(values, ddof=1)) if n_seeds > 1 else None)
        aggregates.append(dict(condition=key[0], replicate=key[1],
                               training_seeds=sorted(present, key=str),
                               expected_training_seeds=sorted(cohort, key=str),
                               missing_training_seeds=sorted(cohort-present, key=str),
                               complete_training_seed_cohort=present == cohort,
                               n_training_seeds=n_seeds, metrics=metrics))

    associations = []
    for seed in sorted(clean, key=str):
        group = clean[seed]
        if len({row["sample_id"] for row in group}) != len(group):
            raise ValueError("Clean feature associations require one record per sample and seed")
        for feature, _ in FEATURES:
            associations.append(dict(seed=seed, condition="clean", **_pearson(group, feature)))

    return _native(dict(
        report_version="adaptation_validation_pilot_v1", pc=pc,
        scope="validation_only", ordinary_inference=True, test_time_augmentation=False,
        final_test_evidence=False, architecture_selection=False,
        grouping=["seed", "condition", "replicate"],
        benefit_definition="parent_loss - variant_loss; positive favors variant",
        accuracy_difference_definition="100 * (variant_accuracy - parent_accuracy), percentage points",
        accuracy_units="fraction", n_unique_validation_samples=len({row["sample_id"] for row in rows}),
        n_paired_records=len(rows), groups=summaries,
        across_seed_aggregates=aggregates,
        aggregate_scope="equal-weight per-training-seed means within each condition/replicate; sample SD (ddof=1), undefined for one seed; SD is not a confidence interval; missing members of the reported seed cohort are explicit",
        feature_associations=associations,
        association_scope="descriptive Pearson correlation on finite clean-input pairs, separately per seed; no p-values",
        metadata=metadata,
    ))


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mean_sd(metric, scale=1, precision=6, signed=False):
    sign = "+" if signed else ""
    mean = format(scale*metric["mean"], f"{sign}.{precision}f")
    sd = ("undefined" if metric["sample_sd"] is None
          else format(scale*metric["sample_sd"], f".{precision}f"))
    return f"{mean} ± {sd}"


def _markdown(summary):
    lines = [
        "# SHD* paired validation pilot", "",
        "Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test "
        "results and do not select an architecture. No test-time augmentation is used.", "",
        "Loss benefit is parent loss minus variant loss; positive values favor the variant. "
        "Accuracy differences are variant minus parent in percentage points (pp). "
        "Each seed, condition and perturbation replicate is summarized separately; repeated "
        "samples and seeds are not pooled as independent observations.", "",
        f"Unique validation samples: {summary['n_unique_validation_samples']}; "
        f"paired records across all groups: {summary['n_paired_records']}.", "",
        "| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {_cell(group['seed'])} | {_cell(group['condition'])} | {_cell(group['replicate'])} | "
            f"{group['n_validation_samples']} | {group['parent_mean_loss']:.6f} | "
            f"{group['variant_mean_loss']:.6f} | {group['mean_loss_benefit']:+.6f} | "
            f"{100*group['parent_accuracy']:.2f}% | {100*group['variant_accuracy']:.2f}% | "
            f"{group['accuracy_difference_pp']:+.2f} pp |")
    lines += ["", "## Across matching training seeds", "",
              "Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, "
              "with equal weight per training seed. Conditions and perturbation replicates "
              "remain separate. SD describes variation across training seeds, not a confidence "
              "interval; it is undefined when n=1. Accuracy means are percentages and their "
              "SDs are percentage points; accuracy differences and their SDs are also in pp. "
              "A partial cohort lists missing seeds and is not a complete-cohort comparison.", "",
              "| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for group in summary["across_seed_aggregates"]:
        metrics = group["metrics"]
        cohort_text = ("complete" if group["complete_training_seed_cohort"] else
                       "partial; missing " + ", ".join(str(seed) for seed in group["missing_training_seeds"]))
        lines.append(
            f"| {_cell(group['condition'])} | {_cell(group['replicate'])} | {group['n_training_seeds']} | "
            f"{cohort_text} | {_mean_sd(metrics['parent_mean_loss'])} | {_mean_sd(metrics['variant_mean_loss'])} | "
            f"{_mean_sd(metrics['mean_loss_benefit'], signed=True)} | "
            f"{_mean_sd(metrics['parent_accuracy'], scale=100, precision=2)} | "
            f"{_mean_sd(metrics['variant_accuracy'], scale=100, precision=2)} | "
            f"{_mean_sd(metrics['accuracy_difference_pp'], precision=2, signed=True)} |")
    lines += ["", "## Descriptive clean-input feature associations", "",
              "Pearson r is calculated separately for each seed using finite pairs only, "
              "with at least three pairs and nonconstant feature and benefit. No p-values "
              "or causal claims are made. Undefined values are retained explicitly.", "",
              "| Seed | Feature | Finite pairs | Pearson r | Undefined reason |",
              "|---|---|---:|---:|---|"]
    for item in summary["feature_associations"]:
        correlation = "undefined" if item["pearson_r"] is None else f"{item['pearson_r']:+.4f}"
        lines.append(f"| {_cell(item['seed'])} | `{item['feature']}` | {item['n_finite_pairs']} | "
                     f"{correlation} | {item['undefined_reason'] or '-'} |")
    if not summary["feature_associations"]:
        lines.append("| - | No clean-input records | 0 | undefined | no_clean_condition |")
    lines += ["", "Figures are descriptive. Conditions are categorical; connecting lines "
              "do not imply a continuous dose response. Checkpoint, split and feature "
              "provenance are retained in the accompanying `MANIFEST.json`; "
              "group and environment metadata are retained in `SUMMARY.json`.", ""]
    return "\n".join(lines)


def _plot(rows, summary, pc):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    if pc == "pc1":
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        clean = [row for row in rows if str(row["condition"]).lower() == "clean"]
        seeds = sorted({row["seed"] for row in clean}, key=str)
        for axis, (feature, label) in zip(axes.flat, FEATURES):
            for seed in seeds:
                pairs = [(float(row[feature]), float(row["benefit"])) for row in clean
                         if row["seed"] == seed and _finite(row.get(feature))]
                if pairs:
                    x, y = np.asarray(pairs).T
                    axis.scatter(x, y, s=10, alpha=.55, label=f"seed {seed}")
            axis.axhline(0, color="grey", linewidth=.8)
            axis.set_xlabel(label)
            axis.set_ylabel("Parent loss - variant loss")
            axis.grid(alpha=.2)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
        axes.flat[-1].axis("off")
        figure.suptitle("SHD* clean validation: input features and paired loss benefit")
    else:
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        conditions = sorted({str(group["condition"]) for group in summary["groups"]},
                            key=lambda value: (value.lower() != "clean", value))
        positions = {condition: index for index, condition in enumerate(conditions)}
        curves = defaultdict(list)
        for group in summary["groups"]:
            curves[(group["seed"], group["replicate"])].append(group)
        for key in sorted(curves, key=str):
            groups = sorted(curves[key], key=lambda group: positions[str(group["condition"])])
            x = [positions[str(group["condition"])] for group in groups]
            label = f"seed {key[0]}, replicate {key[1]}"
            axes[0].plot(x, [group["mean_loss_benefit"] for group in groups], ".-", label=label)
            axes[1].plot(x, [group["accuracy_difference_pp"] for group in groups], ".-", label=label)
        axes[0].set_ylabel("Mean parent loss - variant loss")
        axes[1].set_ylabel("Variant - parent accuracy (pp)")
        for axis in axes:
            axis.axhline(0, color="grey", linewidth=.8)
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[1].set_xticks(list(positions.values()), list(positions), rotation=25, ha="right")
        axes[1].set_xlabel("Validation condition (categorical)")
        figure.suptitle("SHD* paired validation benefit by condition; seeds and replicates separate")
    figure.tight_layout()
    return figure, plt


def write_report(output_dir, paired_rows, pc, metadata):
    """Write JSON, Markdown, PNG and SVG; return their output Paths."""
    pc = str(pc).lower()
    if pc in ("1", "2", "3"):
        pc = "pc" + pc
    if pc not in ("pc1", "pc2", "pc3"):
        raise ValueError("Report PC must be pc1, pc2 or pc3")
    rows = [_native(dict(row)) for row in paired_rows]
    summary = _summarize(rows, pc, metadata)
    # Strict serialization happens before writing any output, including plots.
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "SUMMARY.json", output / "SUMMARY.md"]
    paths[0].write_text(encoded, encoding="utf-8")
    paths[1].write_text(_markdown(summary), encoding="utf-8")
    figure, pyplot = _plot(rows, summary, pc)
    stem = "feature-benefit" if pc == "pc1" else "condition-benefit"
    try:
        for extension in ("png", "svg"):
            path = output / f"{stem}.{extension}"
            figure.savefig(path, dpi=150, metadata={"Creator": "SHD adaptation validation pilot"})
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                                encoding="utf-8")
            paths.append(path)
    finally:
        pyplot.close(figure)
    return paths
