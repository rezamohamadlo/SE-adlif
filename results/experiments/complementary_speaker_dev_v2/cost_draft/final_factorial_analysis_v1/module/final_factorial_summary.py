"""Pure accounting of an already authorized final cohort; no test access/ranking.

The caller must first verify actual checkpoints, configs, source, test rows and
the frozen plan. These JSON consistency checks do not replace that gate.
"""
import hashlib
import json
import math
import statistics

from module.paired_seed_statistics import paired_statistics

ARMS = ('Baseline', 'A', 'B', 'A+B')
DATASETS = ('SHD', 'SSC', 'ECG')
SEEDS = (42, 123, 456)
BUDGETS = {'SHD': 300, 'SSC': 40, 'ECG': 400}


def _number(value, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError('Metric must be a finite numeric value in its stated units')
    return float(value)


def summarize_final_factorial(plan, plan_sha256, results, paper=None):
    """Summarize 36 ordinary TEST_RESULTS payloads keyed by exact checkpoint.

plan_sha256 is the caller-verified hash of the actual frozen plan bytes, not
a reserialization. No file is opened here. No model winner is selected.
"""
    if (plan.get('status') != 'frozen' or plan.get('purpose') != 'final'
            or plan.get('test_selects_ideas') is not False
            or plan.get('architecture_and_settings_frozen') is not True
            or set(plan.get('arms', [])) != set(ARMS)
            or set(plan.get('datasets', [])) != set(DATASETS)
            or set(plan.get('training_seeds', [])) != set(SEEDS)):
        raise ValueError('Require the complete frozen final plan; no development aggregation')
    if (not isinstance(plan_sha256, str) or len(plan_sha256) != 64
            or any(c not in '0123456789abcdef' for c in plan_sha256)):
        raise ValueError('Require the verified actual plan SHA256')
    entries = plan.get('checkpoints', {})
    if not isinstance(results, dict) or len(entries) != 36 or set(entries) != set(results):
        raise ValueError('Require every planned checkpoint result exactly once, with no extras')
    identity = plan.get('evaluator_source_identity')
    if not isinstance(identity, dict) or not identity:
        raise ValueError('Require the frozen evaluator identity')
    evaluator_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    values = {}; source_rows = []
    test_sources = {}; checkpoint_hashes = set()
    for checkpoint, payload in results.items():
        entry = entries[checkpoint]
        if not isinstance(entry, dict) or not isinstance(payload, dict):
            raise ValueError('Require JSON result and plan-entry mappings')
        evidence = payload.get('study_evidence', {})
        if not isinstance(evidence, dict) or not isinstance(payload.get('metrics'), dict):
            raise ValueError('Require final study evidence and metrics mappings')
        arm = entry.get('arm'); dataset = evidence.get('dataset'); seed = evidence.get('seed')
        if arm not in ARMS or dataset not in DATASETS or seed not in SEEDS:
            raise ValueError('Unexpected final arm/dataset/seed')
        key = dataset, arm, seed
        if key in values:
            raise ValueError('Repeated training seed is not an independent replicate')
        if (payload.get('checkpoint') != checkpoint or payload.get('ordinary_inference') is not True
                or payload.get('test_time_augmentation') is not False
                or evidence.get('purpose') != 'final' or evidence.get('test_selects_ideas') is not False
                or evidence.get('verified_configurations') != 36 or evidence.get('preliminary') is not False
                or evidence.get('completed_training_epochs') != BUDGETS[dataset]
                or evidence.get('full_budget') != BUDGETS[dataset]
                or evidence.get('evaluation_plan_sha256') != plan_sha256
                or evidence.get('evaluator_identity_sha256') != evaluator_sha
                or evidence.get('checkpoint_sha256') != entry.get('sha256')):
            raise ValueError('Mixed, preliminary, augmented or mismatched final evidence')
        checkpoint_sha = entry.get('sha256')
        if (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
                or any(c not in '0123456789abcdef' for c in checkpoint_sha)
                or checkpoint_sha in checkpoint_hashes):
            raise ValueError('Checkpoint identities must be distinct SHA256 records')
        checkpoint_hashes.add(checkpoint_sha)
        test_sha = evidence.get('test_data_sha256')
        if (not isinstance(test_sha, str) or len(test_sha) != 64
                or any(c not in '0123456789abcdef' for c in test_sha)
                or (dataset in test_sources and test_sources[dataset] != test_sha)):
            raise ValueError('Matched arms/seeds must use the identical test source')
        test_sources[dataset] = test_sha
        accuracy = _number(payload.get('metrics', {}).get('test_acc'), 0., 1.)
        values[key] = 100*accuracy
        source_rows.append(dict(dataset=dataset, arm=arm, seed=seed,
                                checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha))
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
    if set(values) != expected:
        raise ValueError('Missing part of the matched final cohort')
    if paper is not None and (not isinstance(paper, dict) or set(paper)-set(DATASETS)):
        raise ValueError('Unknown paper comparison dataset')
    datasets = {}
    for dataset in DATASETS:
        label = 'SHD*' if dataset == 'SHD' else dataset
        published = (paper or {}).get(dataset)
        baseline = None
        if published and published.get('compatible') is True:
            if published.get('dataset_label') != label or not published.get('source'):
                raise ValueError('Paper comparison requires compatible protocol label and source')
            baseline = _number(published.get('mean_accuracy_percent'), 0., 100.)
        arms = {}
        for arm in ARMS:
            per_seed = [{'seed': seed, 'test_accuracy_percent': values[dataset, arm, seed]}
                        for seed in SEEDS]
            points = [x['test_accuracy_percent'] for x in per_seed]
            mean = statistics.mean(points)
            arms[arm] = dict(n=3, per_seed=per_seed, mean_test_accuracy_percent=mean,
                sample_sd_percent=statistics.stdev(points),
                delta_paper_pp=mean-baseline if baseline is not None else None)
        contrasts = {}
        for name, positive, negative in [('A_minus_Baseline','A','Baseline'),
                ('B_minus_Baseline','B','Baseline'),('AB_minus_A','A+B','A'),('AB_minus_B','A+B','B')]:
            deltas = [values[dataset, positive, seed]-values[dataset, negative, seed] for seed in SEEDS]
            contrasts[name] = dict(per_seed=[dict(seed=s, difference_pp=v) for s,v in zip(SEEDS,deltas)],
                                   **paired_statistics(deltas))
        interaction = [(values[dataset,'A+B',s]-values[dataset,'A',s])
                       -(values[dataset,'B',s]-values[dataset,'Baseline',s]) for s in SEEDS]
        contrasts['interaction'] = dict(per_seed=[dict(seed=s, difference_pp=v) for s,v in zip(SEEDS,interaction)],
                                       **paired_statistics(interaction))
        datasets[dataset] = dict(dataset_label=label, arms=arms, contrasts=contrasts,
            paper=published, comparable_paper_mean_percent=baseline)
    return dict(metric='ordinary_final_test_accuracy', n_training_seeds=3,
        evaluation_plan_sha256=plan_sha256, datasets=datasets, provenance=source_rows,
        uncertainty='Per-seed values, sample SD and paired95%Studentt intervals across matching training seeds. '
        'Assumes independent seeds and approximately normal differences; small n and fixed splits limit inference. '
        'Intervals are unadjusted descriptive summaries, not speaker-population significance or synergy proof.',
        delta_paper_definition='Local unrounded mean minus compatible published mean, in percentage points; no SD subtraction.',
        test_selects_ideas=False)


def render_final_factorial(summary):
    """Return Markdown; caller chooses a results destination after real gates."""
    lines = ['# Frozen four-arm ordinary test comparison', '', summary['uncertainty'], '',
             'Δ paper = '+summary['delta_paper_definition'], '']
    for dataset in DATASETS:
        record = summary['datasets'][dataset]
        comparable = record['comparable_paper_mean_percent'] is not None
        lines += ['## '+record['dataset_label'], '']
        if comparable:
            lines += [f"Published mean: {record['comparable_paper_mean_percent']:.2f}% ({record['paper']['source']}).", '']
        heading = 'Mean test accuracy ± sample SD (%)'+(' (Δ paper)' if comparable else '')
        lines += [f'| Arm | {heading} |', '|---|---:|']
        for arm in ARMS:
            stats = record['arms'][arm]
            text = f"{stats['mean_test_accuracy_percent']:.2f} ± {stats['sample_sd_percent']:.2f}"
            if comparable:
                text += f" ({stats['delta_paper_pp']:+.2f} pp)"
            lines.append(f'| {arm} | {text} |')
        if comparable:
            lines += ['', 'Per-seed Δ paper compares an individual training seed with the published mean; it is not a replicated-mean comparison.']
        seed_headings = [f'Seed {seed}: test accuracy (%)'+(' (Δ paper)' if comparable else '') for seed in SEEDS]
        lines += ['', '| Arm | '+' | '.join(seed_headings)+' |', '|---|---:|---:|---:|']
        for arm in ARMS:
            points=record['arms'][arm]['per_seed']
            cells=[]
            for point in points:
                value=point['test_accuracy_percent'];text=f'{value:.2f}'
                if comparable:
                    text+=f" ({value-record['comparable_paper_mean_percent']:+.2f} pp)"
                cells.append(text)
            lines.append('| '+arm+' | '+' | '.join(cells)+' |')
        lines += ['', '| Paired contrast | Mean (pp) | Sample SD (pp) | Descriptive 95% t interval (pp) |', '|---|---:|---:|---:|']
        for name,stats in record['contrasts'].items():
            low,high=stats['ci95_pp']
            lines.append(f"| {name} | {stats['mean_pp']:+.3f} | {stats['sample_sd_pp']:.3f} | [{low:+.3f}, {high:+.3f}] |")
        lines += ['']
    lines += ['No cross-dataset pooling or test-selected winner. Combined superiority alone does not prove synergy.']
    return '\n'.join(lines)+'\n'
