"""Analysis-named reports; reuse established reductions without rewriting pilot code."""
from collections import defaultdict
import statistics

from scripts.adaptation_analysis import reporting
from scripts.adaptation_analysis.pilot import write_json


def write_reports(root, paired, metadata):
    outputs = []
    for analysis in metadata['analyses']:
        rows = [r for r in paired if r['condition'] == 'clean' or
                (analysis == 'time_scaling' and r['condition'].startswith('time_scale_')) or
                (analysis == 'spike_deletion' and r['condition'].startswith('delete_'))]
        directory = root / 'analyses' / analysis
        directory.mkdir(parents=True, exist_ok=True)
        summary = reporting._summarize(rows, None, dict(metadata, analysis=analysis))
        summary.pop('pc')
        summary['report_version'] = 'adaptation_ssc_pilot_v1'
        summary['scope'] = 'ssc_validation_pilot'
        summary['analysis'] = analysis
        clean = {g['seed']: g['mean_loss_benefit'] for g in summary['groups'] if g['condition'] == 'clean'}
        by_seed = defaultdict(list)
        for g in summary['groups']:
            g['benefit_change_from_clean'] = g['mean_loss_benefit'] - clean[g['seed']]
            by_seed[g['condition'], g['seed']].append(g)
        reduced = []
        for (condition, seed), groups in sorted(by_seed.items()):
            reduced.append(dict(condition=condition, seed=seed, n_replicates=len(groups),
                                benefit_change=statistics.mean(g['benefit_change_from_clean'] for g in groups),
                                accuracy_gap_pp=statistics.mean(g['accuracy_difference_pp'] for g in groups)))
        summary['within_seed_replicate_means'] = reduced
        aggregates = []
        for condition in sorted({g['condition'] for g in reduced}):
            groups = [g for g in reduced if g['condition'] == condition]
            aggregates.append(dict(condition=condition, n_training_seeds=len(groups), **{
                key: dict(mean=statistics.mean(g[key] for g in groups),
                          sample_sd=statistics.stdev(g[key] for g in groups) if len(groups) > 1 else None)
                for key in ('benefit_change', 'accuracy_gap_pp')}))
        summary['replicate_averaged_across_seed_effects'] = aggregates
        text = reporting._markdown(summary).replace('# SHD* paired validation pilot', '# SSC paired validation pilot: ' + analysis)
        text = text.replace('\n\nFrozen-checkpoint',
            f"\n\n**Pilot subset: {metadata['samples']} fixed validation records.** "
            f"Full clean anchors use {metadata['full_clean_anchor_samples']} records per checkpoint "
            "and are stored separately in `../../VALIDATION_ANCHORS.json`. "
            "These tables and feature associations describe the subset only.\n\nFrozen-checkpoint", 1)
        text += '\n## Change from clean input\n\nPerturbation realizations are averaged within each training seed first. '
        text += 'Across-seed values are mean and sample SD; three seeds are not a confidence interval. '
        text += 'Positive loss-benefit change favors AMT relative to clean input. No causal claim or significance test is made.\n\n'
        text += '| Condition | Training seeds | Benefit change (mean +/- SD) |\n|---|---:|---:|\n'
        for g in aggregates:
            v = g['benefit_change']
            sd = 'undefined' if v['sample_sd'] is None else f"{v['sample_sd']:.8f}"
            text += f"| {g['condition']} | {g['n_training_seeds']} | {v['mean']:+.8f} +/- {sd} |\n"
        write_json(directory / 'SUMMARY.json', summary)
        (directory / 'SUMMARY.md').write_text(text, encoding='utf-8')
        outputs += [directory / 'SUMMARY.json', directory / 'SUMMARY.md']
        # The legacy plot selector is internal only; no PC is persisted in new results.
        figure, plt = reporting._plot(rows, summary, 'pc1' if analysis == 'clean' else 'pc2')
        figure.suptitle('SSC validation pilot: '+analysis)
        figure.tight_layout(rect=(0, 0, 1, .96))
        try:
            for ext in ('png', 'svg'):
                path = directory / ('benefit.' + ext)
                figure.savefig(path, dpi=150, metadata={'Creator': 'SSC validation pilot'})
                outputs.append(path)
        finally:
            plt.close(figure)
    return outputs
