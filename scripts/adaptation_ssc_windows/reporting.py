"""Sample-weighted paired descriptors; no windows counted as independent seeds."""
import numpy as np

from scripts.adaptation_ssc_suite.common import require
from scripts.adaptation_ssc_windows.metrics import METRICS


def summarize(values, seeds, widths):
    groups, differences, across = [], [], []
    models = ('MT_SE_adLIF', 'MT_TC_SE_adLIF')
    for seed in seeds:
        parent, variant = [values[m, seed] for m in models]
        require(parent.shape == variant.shape and parent.shape[0] == len(widths) and parent.shape[2] == len(METRICS),
                'Window comparison arrays have different shape')
        for w, width in enumerate(widths):
            for model, array in [(models[0], parent), (models[1], variant)]:
                groups.append(dict(seed=seed, model=model, window_frames=width,
                    metrics={name: dict(n_finite_records=int(np.isfinite(array[w, :, k]).sum()),
                        mean=float(np.mean(array[w, np.isfinite(array[w, :, k]), k])) if np.isfinite(array[w, :, k]).any() else None)
                        for k, name in enumerate(METRICS)}))
            for k, name in enumerate(METRICS):
                keep = np.isfinite(parent[w, :, k]) & np.isfinite(variant[w, :, k])
                differences.append(dict(seed=seed, window_frames=width, metric=name, n_finite_pairs=int(keep.sum()),
                    mean_variant_minus_parent=float(np.mean(variant[w, keep, k]-parent[w, keep, k])) if keep.any() else None))
    for width in widths:
        for name in METRICS:
            data = [r['mean_variant_minus_parent'] for r in differences
                    if r['window_frames'] == width and r['metric'] == name and r['mean_variant_minus_parent'] is not None]
            across.append(dict(window_frames=width, metric=name, n_training_seeds=len(data),
                mean=float(np.mean(data)) if data else None, sample_sd=float(np.std(data, ddof=1)) if len(data) > 1 else None))
    return dict(groups=groups, paired_per_seed_differences=differences, across_seed_differences=across,
        weighting='equal record weight within each model/seed/window width; then three training-seed values',
        limitations=['Descriptive validation only; windows, speakers and records are not extra independent training seeds.',
            'Lower adjacent JS or fewer class switches may describe consistently wrong or low-confidence outputs.',
            'Inspect true-class probability, window accuracy and window CE together; smoother is not automatically better.',
            'Window CE uses mean raw membrane potentials; probability metrics use mean frame-softmax probabilities.',
            'Diagnostic windows exclude burn-in/minimum/batch padding, whereas ordinary anchor loss retains its original mask.',
            'Window widths and training JS overlap/stride are different contracts; JS is never added to this prediction loss.',
            'No p-values, bootstrap, automatic confidence/benefit regression, TTA or independent-test/causal claim.'])


def markdown(summary):
    text = ['# SSC MT/MT-TC clean output-window diagnostics', '',
        'Full official validation: 9981 records, training seeds 42/123/456, six saved checkpoints.',
        'Nonoverlapping windows of 5/10/20 frames (20/40/80 ms), after ten burn-in frames and before synthetic padding.',
        'Positive differences mean MT-TC minus MT. JS and CE use nats; other metrics use fractions (.01 accuracy = 1 pp).',
        'Probability-window accuracy and mean-membrane window accuracy are separate metrics.', '',
        '| Window (frames) | Metric | Seeds | Mean MT-TC minus MT | Sample SD |', '|---:|---|---:|---:|---:|']
    for row in summary['across_seed_differences']:
        mean = 'undefined' if row['mean'] is None else f"{row['mean']:+.8f}"
        sd = 'undefined' if row['sample_sd'] is None else f"{row['sample_sd']:.8f}"
        text.append(f"| {row['window_frames']} | {row['metric']} | {row['n_training_seeds']} | {mean} | {sd} |")
    text += ['', '## Per-seed paired differences', '',
        '| Seed | Frames | Metric | Finite paired records | MT-TC minus MT |', '|---:|---:|---|---:|---:|']
    for row in summary['paired_per_seed_differences']:
        value = 'undefined' if row['mean_variant_minus_parent'] is None else f"{row['mean_variant_minus_parent']:+.8f}"
        text.append(f"| {row['seed']} | {row['window_frames']} | {row['metric']} | {row['n_finite_pairs']} | {value} |")
    text += ['', '## Limits', '']+['- '+value for value in summary['limitations']]
    text += ['', 'All ordinary clean anchors must pass before these summaries are published. '
             'Raw per-batch window arrays and scalar CSVs retain explicit undefined reasons; no record is silently discarded.', '']
    return '\n'.join(text)
