"""Strict record pairing and descriptive summaries; user-launched only."""
import statistics

from scripts.adaptation_ecg.data import require
from scripts.adaptation_ecg.runtime import aggregate


def paired_records(parent, variant, clean_parent, clean_variant):
    groups = (parent, variant, clean_parent, clean_variant)
    ids = [r['sample_id'] for r in parent]
    require(len(ids) == len(set(ids)) and bool(ids), 'ECG sample IDs must be nonempty and unique')
    require(all([r['sample_id'] for r in rows] == ids for rows in groups), 'ECG pair coverage/order differs')
    result = []
    for p, v, cp, cv in zip(*groups):
        for left, right in ((p, v), (cp, cv)):
            require(all(left[k] == right[k] for k in ('sample_id', 'seed', 'condition', 'replicate',
                'input_sha256', 'targets_sha256', 'blocks_sha256', 'valid_steps')), 'ECG records are not paired')
        require(cp['condition'] == cv['condition'] == 'clean' and cp['replicate'] == cv['replicate'] == 0 and
                p['model'] == cp['model'] and v['model'] == cv['model'] and
                all(p[k] == cp[k] == cv[k] for k in ('sample_id', 'seed', 'targets_sha256', 'blocks_sha256', 'valid_steps')),
                'ECG transformed and clean labels/blocks/seed differ')
        benefit, clean_benefit = p['loss']-v['loss'], cp['loss']-cv['loss']
        result.append(dict(sample_id=p['sample_id'], seed=p['seed'], condition=p['condition'], replicate=p['replicate'],
            parent=p['model'], variant=v['model'], input_sha256=p['input_sha256'], targets_sha256=p['targets_sha256'],
            blocks_sha256=p['blocks_sha256'], valid_steps=p['valid_steps'], parent_loss=p['loss'], variant_loss=v['loss'],
            parent_accuracy=p['accuracy'], variant_accuracy=v['accuracy'],
            accuracy_gap_pp=100*(v['accuracy']-p['accuracy']), loss_benefit=benefit,
            clean_loss_benefit=clean_benefit, benefit_change_from_clean=benefit-clean_benefit))
    return result


def describe(rows, seeds):
    """Average realizations inside each seed before between-seed mean/sample SD."""
    summaries = []
    for condition in sorted({r['condition'] for r in rows}):
        per_seed = []
        for seed in seeds:
            selected = [r for r in rows if r['seed'] == seed and r['condition'] == condition]
            require(bool(selected), 'Incomplete ECG descriptive seed cohort')
            per_seed.append(dict(seed=seed,
                **{key: statistics.mean(r[key] for r in selected) for key in
                   ('parent_loss', 'variant_loss', 'parent_accuracy', 'variant_accuracy',
                    'accuracy_gap_pp', 'loss_benefit', 'benefit_change_from_clean')}))
        summaries.append(dict(condition=condition, n_training_seeds=len(seeds), per_seed=per_seed,
            aggregate={key: dict(mean=statistics.mean(r[key] for r in per_seed),
                                sample_sd=statistics.stdev(r[key] for r in per_seed))
                       for key in per_seed[0] if key != 'seed'}))
    return summaries


def write_pair(root, pair, spec, cache, write):
    directory = root/'pairs'/pair['id']
    directory.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], []
    for seed in spec['seeds']:
        cp, cv = [cache[m, seed, 'clean', 0] for m in (pair['parent'], pair['variant'])]
        for condition, rep in spec['conditions']:
            p, v = [cache[m, seed, condition, rep] for m in (pair['parent'], pair['variant'])]
            rows.extend(paired_records(p, v, cp, cv))
            metrics.extend(dict(model=m, seed=seed, condition=condition, replicate=rep, **aggregate(records))
                           for m, records in ((pair['parent'], p), (pair['variant'], v)))
    require(len(rows) == 360, 'Incomplete ECG pair evaluation coverage')
    from scripts.adaptation_analysis import pilot
    pilot.write_csv(directory/'paired_records.csv', rows)
    summary = dict(pair=pair, scope='full fixed validation cohort, clean and signed deletion only',
        status='evaluation_complete', records=30, paired_rows=len(rows), conditions=spec['conditions'],
        metrics=metrics, descriptive=describe(rows, spec['seeds']), full_mechanism_study_complete=False,
        weighting='record means and step-weighted metrics reported separately; equal stored lengths here',
        limitations=['no independent patient IDs', 'no inferential interval or significance claim',
                     'no features/time scaling/regression/bootstrap', 'validation-selected checkpoints; no independent test'])
    write(directory/'SUMMARY.json', summary)
    text = [f"# ECG {pair['variant']} / {pair['parent']}", '',
        'All 30 validation records, seeds 42/123/456. Evaluation only: clean and 10% signed-event deletion.',
        'Positive loss benefit means lower variant CE. Differences in accuracy are percentage points.',
        'Deletion realizations are averaged within each training seed before mean/sample SD over the three seeds.',
        'Original zero-vector argmax-to-class-0 semantics are preserved. No independent-patient or test claim.', '',
        '| Condition | Seed | Parent validation accuracy (%) | Variant validation accuracy (%) | Gap (pp) | CE benefit | Benefit change |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in summary['descriptive']:
        for row in group['per_seed']:
            text.append(f"| {group['condition']} | {row['seed']} | {row['parent_accuracy']*100:.6f} | "
                f"{row['variant_accuracy']*100:.6f} | {row['accuracy_gap_pp']:+.6f} | "
                f"{row['loss_benefit']:+.8f} | {row['benefit_change_from_clean']:+.8f} |")
        a = group['aggregate']
        text.append(f"| {group['condition']} | mean ± sample SD (n=3) | "
            f"{100*a['parent_accuracy']['mean']:.6f} ± {100*a['parent_accuracy']['sample_sd']:.6f} | "
            f"{100*a['variant_accuracy']['mean']:.6f} ± {100*a['variant_accuracy']['sample_sd']:.6f} | "
            f"{a['accuracy_gap_pp']['mean']:+.6f} ± {a['accuracy_gap_pp']['sample_sd']:.6f} | "
            f"{a['loss_benefit']['mean']:+.8f} ± {a['loss_benefit']['sample_sd']:.8f} | "
            f"{a['benefit_change_from_clean']['mean']:+.8f} ± {a['benefit_change_from_clean']['sample_sd']:.8f} |")
    text += ['', 'Features, label-aligned time scaling and inferential statistics remain pending.', '']
    (directory/'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8')
    return [directory/name for name in ('paired_records.csv', 'SUMMARY.json', 'SUMMARY.md')], summary
