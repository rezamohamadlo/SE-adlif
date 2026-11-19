"""Present authenticated saved outcomes; no new regressions or resampling."""
import csv
import json
import math
import os
import statistics
from pathlib import Path

from project_paths import local_path
from scripts.adaptation_stage8.evidence import DATASETS, ECG, EXPERIMENTS, PAIRS, SEEDS, SHD, SSC, STAGE7, require

LABEL = {'SHD': 'SHD*', 'SSC': 'SSC', 'ECG': 'ECG'}


def aggregate(rows, key):
    require(sorted(row['seed'] for row in rows) == SEEDS, 'Aggregate requires three distinct matching seeds')
    values = [float(row[key]) for row in rows]
    require(all(math.isfinite(v) for v in values), 'Nonfinite seed aggregate')
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n_training_seeds=3)


def formatted(value, scale=1, digits=4):
    return f"{value['mean']*scale:.{digits}f} ± {value['sample_sd']*scale:.{digits}f}"


def paired(costs, ds, parent, variant, key):
    index = {(r['model'], r['seed']): r for r in costs if r['dataset'] == ds}
    rows = [dict(seed=s, difference=index[variant, s]['measurements'][key] -
                 index[parent, s]['measurements'][key]) for s in SEEDS]
    return aggregate(rows, 'difference')


def feature_values(rows, feature):
    valid, counts = [], dict(total=len(rows), missing_or_nonfinite=0, censored=0, invalid=0)
    for row in rows:
        value = row.get(feature)
        if feature.startswith('autocorr') and row.get('autocorr_censored', False):
            counts['censored'] += 1
        elif feature.startswith('autocorr') and (row.get('autocorr_valid') is False or row.get('autocorr_reason')):
            counts['invalid'] += 1
        elif feature == 'fast_slow_contrast' and (row.get('fast_slow_valid') is False or row.get('fast_slow_reason')):
            counts['invalid'] += 1
        elif value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
            counts['missing_or_nonfinite'] += 1
        else:
            valid.append(float(value))
    counts['finite_uncensored'] = len(valid)
    require(sum(counts[k] for k in ['finite_uncensored', 'missing_or_nonfinite', 'censored', 'invalid']) == len(rows),
            'Feature exclusions overlap')
    return valid, counts


def write_csv(root, name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (root / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def link(root, name, label):
    return '[' + label + '](' + Path(os.path.relpath(local_path(name), root)).as_posix() + ')'


def table(lines, headings, rows):
    lines.extend(['', '| ' + ' | '.join(headings) + ' |', '|' + '|'.join(['---'] * len(headings)) + '|'])
    lines.extend('| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows)
    lines.append('')


def render(prepared):
    # Imports used for presentation only; --check-only never reaches this function.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = prepared['root']
    root.mkdir(parents=True, exist_ok=True)
    (root / 'figures').mkdir(exist_ok=True)
    lines = ['# Supervisor study: final exploratory synthesis', '',
        'Source review: 17 November 2026. This report presents saved evidence and seed-matched arithmetic. '
        'No new model execution, feature extraction, regression, bootstrap, training or official-test access.', '',
        '**Main conclusion:** the four mechanisms show dataset- and seed-dependent behavior. '
        'The registered experiments do not establish a universal architectural winner or a causal explanation '
        'of natural accuracy. Negative, ineffective and inconclusive results are retained.', '',
        'Scope: AMT/MT, LT-DTH/DTH, SA-MT/MT and MT-TC/MT; six unique models, '
        'three matching training seeds (42/123/456), SHD*/SSC/ECG. Shared MT parents and '
        'repeated controls are shared observations, not additional training runs. '
        'All means use sample SD across those three seeds; SD is not a confidence interval. '
        'No numeric paper baseline is mixed with incompatible validation scores. '
        'The existing paper/reference comparison remains separate: ' +
        link(root, 'reports/comparison-summary.md', 'main accuracy and cost comparison') + '.', '',
        '## 1. Ordinary performance', '',
        'Full clean validation uses source-high precision and original supervision: SHD* 1,631 records '
        '(fixed 20% training split, split seed 42), SSC 9,981 official-validation records, ECG 30 validation '
        'sequences. SHD*/SSC accuracy counts recordings; ECG accuracy counts labeled timesteps. '
        'CE benefit = parent loss minus variant loss; accuracy gap = variant minus parent, in percentage points (pp). '
        'MT-TC auxiliary training JS is excluded from evaluation CE. Dataset-specific losses differ '
        '(SHD* temporal-softmax sum, SSC mean membrane, ECG per-block summed membrane); do not pool their CE.']
    validation_groups = []
    for ds in DATASETS:
        for pair in PAIRS:
            rows = [r for r in prepared['validations'] if r['dataset'] == ds and r['pair'] == pair]
            entry = dict(dataset=ds, pair=pair,
                **{key: aggregate(rows, key) for key in ['parent_accuracy', 'variant_accuracy', 'accuracy_gap_pp', 'loss_benefit']})
            validation_groups.append(entry)
    table(lines, ['Dataset', 'Pair', 'Parent validation %', 'Variant validation %', 'Paired gap (pp)', 'CE benefit'],
        [[LABEL[r['dataset']], r['pair'], formatted(r['parent_accuracy'], 100), formatted(r['variant_accuracy'], 100),
          formatted(r['accuracy_gap_pp']), formatted(r['loss_benefit'], digits=6)] for r in validation_groups])
    lines += ['The following ordinary final-test scores were already present in each training history. '
        'The saved cost evaluation independently reproduces them within 1e-6 accuracy. No official test data '
        'is opened here. These historical test results were exposed during earlier development and are '
        'descriptive, not independent confirmation. No TTA or extra-seed extension is included.']
    costs, cost_groups = prepared['costs'], []
    cost_keys = ['test_accuracy', 'forward_ms_per_sample', 'total_parameters', 'trainable_parameters',
                 'spikes_l1_per_sample', 'spikes_l2_per_sample', 'spikes_total_per_sample',
                 'firing_l1_percent', 'firing_l2_percent', 'synops_per_sample',
                 'dense_gflops_test_padded', 'arithmetic_gflops_test_padded', 'logical_gops_test_padded']
    flat = [dict(dataset=r['dataset'], model=r['model'], seed=r['seed'], checkpoint=r['checkpoint'],
        cost_record=r['cost_record'], batch_size=r['batch_size'], test_accuracy=r['test_accuracy'],
        **{k: v for k, v in r['measurements'].items() if isinstance(v, (int, float))},
        sample_count=r['measurements']['totals']['samples'],
        accuracy_targets=r['measurements']['totals']['targets'], environment=r['environment'], scopes=r['scopes']) for r in costs]
    for ds in DATASETS:
        for model in prepared['models']:
            rows = [r for r in flat if r['dataset'] == ds and r['model'] == model]
            cost_groups.append(dict(dataset=ds, model=model, **{key: aggregate(rows, key) for key in cost_keys}))
    table(lines, ['Dataset', 'Model', 'Historical final test (%)', 'Test seeds 42 / 123 / 456 (%)'],
        [[LABEL[r['dataset']], r['model'], formatted(r['test_accuracy'], 100),
          ' / '.join(f"{next(c['test_accuracy'] for c in flat if c['dataset']==r['dataset'] and c['model']==r['model'] and c['seed']==s)*100:.4f}" for s in SEEDS)]
         for r in cost_groups])
    lines += ['The main paper/reference tables document their own provenance and reference ECG stopping-budget '
        'mismatch. The mechanism comparisons here use each variant’s matching trained parent. '
        'All six study models inherit their dataset-specific paper_classification_v1 settings; '
        'MT-TC intentionally changes the training objective. Checkpoint identities and resolved configs are pinned.']
    lines += ['', '## 2. Input features and diagnostic patterns', '',
        'The figures below plot existing clean-input descriptors once per recording, rather than once per '
        'model or seed. They use all 1,631 / 9,981 / 30 validation records. SHD*/SSC feature support excludes '
        'synthetic padding; feature windows are 20/200 ms with 4-ms frames. ECG uses signed-event activity '
        'and 20/200 model-step windows; physical milliseconds have not been established. '
        'Panels use native units and independent axes, not a pooled physical scale. Invalid/censored ACF '
        'values are counted explicitly and excluded from numeric summaries, without imputation. '
        'CV-ISI was not measured and is not claimed.']
    feature_summary = []
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for i, ds in enumerate(DATASETS):
        keys = [('mean_rate_hz', 'Total event rate (events/s)'), ('fast_slow_contrast', 'Fast/slow contrast'),
                ('autocorr_tau_ms', 'ACF crossing lag (ms)')] if ds != 'ECG' else [
                ('events_per_step', 'Nonzero signed events/model step'), ('fast_slow_contrast', 'Fast/slow contrast'),
                ('autocorr_tau_steps', 'ACF crossing lag (model steps)')]
        for j, (key, title) in enumerate(keys):
            values, counts = feature_values(prepared['features'][ds], key)
            quantiles = np.quantile(values, [.25, .5, .75]).tolist() if values else [None] * 3
            feature_summary.append(dict(dataset=ds, feature=key, **counts,
                q25=quantiles[0], median=quantiles[1], q75=quantiles[2]))
            ax = axes[i, j]
            if values:
                ax.hist(values, bins=min(40, max(5, int(math.sqrt(len(values))))), color=['#286f9b', '#bc6a28', '#6d5098'][i])
            else:
                ax.text(.5, .5, 'No eligible finite values', transform=ax.transAxes, ha='center')
            ax.set_title(f"{LABEL[ds]}: {title}\nfinite={len(values)}/{counts['total']}; censored={counts['censored']}")
            ax.set_ylabel('Recordings'); ax.set_xlabel(title)
    fig.savefig(root / 'figures/input_feature_distributions.png', dpi=160); plt.close(fig)
    lines += ['', '![Native-unit clean feature distributions](figures/input_feature_distributions.png)']
    table(lines, ['Dataset', 'Feature', 'Eligible / total', 'Censored / invalid / missing', 'Q25 / median / Q75'],
        [[LABEL[r['dataset']], r['feature'], f"{r['finite_uncensored']} / {r['total']}",
          f"{r['censored']} / {r['invalid']} / {r['missing_or_nonfinite']}",
          ' / '.join('undefined' if r[k] is None else f'{r[k]:.6g}' for k in ['q25', 'median', 'q75'])] for r in feature_summary])
    lines += ['This distribution comparison describes the observed cohorts. It does not identify a dataset '
        'effect independent of duration, representation, class mix or sampling. The saved stage-7 '
        'class/temporal maps and input-optimization examples are available in ' +
        link(root, STAGE7 + '/presentation_v1/SUMMARY.md', 'the comparable diagnostic figure collection') + '. '
        'Those gradient cohorts are only 40 SHD*, 70 SSC and all 30 ECG records per checkpoint; '
        'four fixed records/checkpoint undergo input optimization. Physical-input gradients include burn-in '
        'sensitivity and report padding separately. SLAYER maps are surrogate sensitivities, not causal attention.']
    lines += ['', '## 3. Feature–benefit relationships and mechanism conclusions', '',
        'The next table copies existing adjusted coefficients; no model is refitted. Coefficients are CE benefit '
        'per fixed original-cohort population-SD increase in a predictor. SHD*/SSC jointly control class, '
        'speaker, log duration and activity. Saved SHD intervals use a two-feature family within each pair; '
        'SSC uses an eight-feature/pair family across four pairs. They are approximate, speaker-clustered '
        'exploratory intervals conditional on the observed trained models. They do not give joint coverage '
        'across datasets or robustness over new training seeds. SHD has only ten speaker clusters. '
        'ECG uses record descriptors and label-class proportions, drops constant duration, has no verified '
        'patient identities and supplies no population interval, bootstrap or p-value.']
    table(lines, ['Dataset', 'Pair', 'Feature', 'Coefficient ± seed SD', 'Saved interval', 'Saved direction / consistent seed signs'],
        [[LABEL[r['dataset']], r['pair'], r['feature'], formatted(r, digits=6),
          'none (descriptive)' if r['interval'] is None else '[' + ', '.join(f'{x:+.6f}' for x in r['interval']) + ']',
          r['direction'] + ' / ' + str(r['seed_direction_consistent'])] for r in prepared['associations']])
    lines += ['All exact seed coefficients, complete-case counts, cluster counts and interval-family labels '
        'are exported in ASSOCIATIONS.csv. SSC exclusions affect only joint regression; interventions retain '
        'all validation recordings. Difficulty controls do not remove every confound.']
    table(lines, ['Mechanism / initial hypothesis', 'Bounded verdict', 'Evidence and limitation'], [
        ['AMT: benefit grows with fast/slow contrast', 'Insufficient consistent support',
         'SHD* fast/slow interval includes zero; SHD* ACF interval is positive but seed signs conflict. SSC primary intervals include zero; ECG is descriptive.'],
        ['SA-MT: separate transient/sustained strengths help', 'Insufficient consistent support',
         'SHD* fast/slow interval includes zero and ACF is positive with mixed seed signs. Other datasets and interventions must be considered; surrogate maps do not establish pathway mediation.'],
        ['LT-DTH: learned trace aligns with input timescale', 'Insufficient consistent support',
         'SHD* adjusted ACF interval includes zero; SHD* deletion weakens relative benefit within its fixed-cohort interval. Learned trace constants are model/layer observations, not per-record mediators.'],
        ['MT-TC: stable adjacent evidence helps decisions', 'Mixed; general stability claim unsupported',
         'SHD* adjacent-window JS improves only in seed 42; SSC smoothing/window CE improve in 42/123 and reverse in 456 at every tested width. Lower mean JS alone is not useful class evidence.'],
    ])
    if prepared['traces']:
        lines += ['Saved learned-trace descriptors are summarized below where authenticated records exist. '
            'Only existing pre-aggregated trace tables are reproduced. SSC per-checkpoint trace records '
            'also remain available in ' + link(root, SSC + '/pairs/lt_dth/evaluation/parameters', 'the authenticated parameter archive') +
            '; their absence from this pre-aggregated table is not a zero-valued estimate. '
            'No new checkpoint parameter extraction or trace-to-feature regression is performed.']
        table(lines, ['Dataset', 'Model', 'Layer', 'Trace', 'Saved e-folding time (ms) ± seed SD'],
            [[LABEL[r['dataset']], r['model'], r['layer'], r['trace'],
              f"{r['mean_ms']:.4f} ± {r['sample_sd_ms']:.4f}"] for r in prepared['traces']])
    lines += ['', '## 4. Controlled experiments and input selection', '',
        'Positive changes below favor the variant relative to its own clean CE gap; they do not require '
        'absolute accuracy or CE improvement. Entries copy the three predeclared conditions from saved '
        'studies. Random deletion realizations average within recording and seed before any seed summary. '
        'SHD uses a separate three-intervention interval family within each pair, SSC a twelve-effect family; '
        'ECG is descriptive without intervals.']
    table(lines, ['Dataset', 'Pair', 'Condition', 'Change in CE benefit ± seed SD', 'Saved interval / direction'],
        [[LABEL[r['dataset']], r['pair'], r['condition'], formatted(r, digits=6),
          'none / descriptive' if r['interval'] is None else '[' + ', '.join(f'{x:+.6f}' for x in r['interval']) + '] / ' + r['direction']]
         for r in prepared['interventions']])
    lines += ['Deletion preserves lengths and targets but changes event density and several temporal statistics. '
        'Time rescaling changes duration/rate/structure jointly. ECG nearest-bin-center resampling maps signed '
        'inputs, labels and masks together, rebuilds temporal blocks and preserves the saved all-zero-label '
        'argmax-to-class-0 policy; event totals and class proportions can change. Thus these transformations '
        'do not isolate one latent mechanism. No temporal-window permutation or retraining-on-corrupted-data '
        'experiment is claimed.', '',
        'The direct MT-TC window diagnostics distinguish frame-probability JS/concentration from window '
        'mean-membrane CE/accuracy and exclude burn-in and synthetic padding only from descriptor windows. '
        'Ordinary loss masks remain unchanged. See ' +
        link(root, EXPERIMENTS + SHD['mt_tc'] + '/window_evidence_v1/SUMMARY.md', 'SHD* window evidence') + ' and ' +
        link(root, EXPERIMENTS + 'adaptation_ssc_windows_v1/SUMMARY.md', 'SSC windows at 5/10/20 frames') +
        '. Windows are not independent seeds; no confidence-to-benefit regression is inferred.', '',
        'Stage 7 adds frozen-weight, six-step constrained input optimization: native nonnegative integer '
        'event counts (SHD*/SSC) or signed ternary events (ECG), fixed length, targets and block masks, '
        'preserved per-channel event totals and separately preserved ECG polarity totals. At most 1% of '
        'original events move from occupied to originally empty cells within channel; the budget never '
        'accumulates across steps. Three random controls match actual channel/polarity movement allocations. '
        'Every source input is evaluated identically by all six target models at the matching seed. '
        'Both predeclared optimization origins are required. Random controls average within record; '
        'four fixed diagnostic records average within training seed before the three-seed mean and sample SD.']
    path = local_path(STAGE7 + '/origin_controls_v1/ORIGIN_ROLE_SUMMARY.csv')
    with path.open(encoding='utf-8', newline='') as stream:
        origins = list(csv.DictReader(stream))
    table(lines, ['Dataset', 'Pair', 'Optimization origin', 'Optimized-minus-random parent–variant CE contrast ± seed SD'],
        [[r['dataset'], r['pair'], r['origin_role'],
          f"{float(r['mean_ce_contrast']):+.6f} ± {float(r['sample_sd_training_seeds']):.6f}"] for r in origins])
    review = prepared['stage7_review']
    lines += [f"All 54 checkpoint jobs and 432 repeat checks passed. {review['checks']['accepted_diagnostic_inputs']} "
        'of 216 input-optimization attempts produced accepted nonzero changes; all 45 ineffective attempts '
        'remain recorded. Nine transfer bundles contain 5,184 target evaluations, 864 unique shared '
        'optimized/random inputs and 288 paired record/seed/origin rows. '
        'Parent-optimized mean contrasts are positive in 11/12 dataset/pair cells, while variant-optimized '
        'means are negative in all 12. This is strong input-selection-model dependence, not universal '
        'robustness or natural-accuracy superiority. No zero map was replaced by fabricated guidance. '
        'Fixed labels do not guarantee unchanged semantic class; a failed greedy attempt does not prove '
        'that no admissible perturbation exists. See ' + link(root, STAGE7 + '/POST_RUN_REVIEW.md', 'independent stage-7 review') + '.']
    lines += ['', '## Activity, computational estimates and timing', '',
        'All 54 saved ordinary-inference cost records match the exact study checkpoint SHA-256, selected '
        'epoch and normalized resolved config. They measured historical test data; this report reads only '
        'the saved measurements. Their accuracy checks match one unambiguous final-test row per history. '
        'This section does not measure validation or perturbation-specific cost.', '',
        'Activity means hidden layers L1 and L2, whose outputs are binary spikes. The output layer is a '
        'nonspiking integrator and contributes no output-spike count. Layer firing fraction = 100 × hidden '
        'spikes / (neurons × summed stored sequence steps). Stored extent includes configured minimum '
        'padding and burn-in; extra batch padding is excluded from activity but retained in padded compute. '
        'These are not physical-duration firing rates and are distinct from the physical-support input '
        'features and stage-7 padding-gradient summaries. Raw stored and batch-padded counts are exported.']
    table(lines, ['Dataset', 'Model', 'L1 spikes/sequence', 'L2 spikes/sequence', 'L1 firing %', 'L2 firing %', 'Forward ms/sequence'],
        [[LABEL[r['dataset']], r['model'], formatted(r['spikes_l1_per_sample'], digits=1),
          formatted(r['spikes_l2_per_sample'], digits=1), formatted(r['firing_l1_percent'], digits=3),
          formatted(r['firing_l2_percent'], digits=3), formatted(r['forward_ms_per_sample'], digits=3)] for r in cost_groups])
    lines += ['Timing uses the recorded RTX 4090, float32 high matmul precision, batch 256 for SHD*/SSC '
        'and 64 for ECG, three timing repeats and two warm-up batches. It measures synchronized '
        'MLPSNN.forward wall time excluding loading, transfer, loss and temporal prediction aggregation. '
        'SD above is across training seeds, not across timing repeats. Sessions, evaluator versions and '
        'environment snapshots differ, so paired differences are descriptive and do not establish hardware '
        'speed superiority. Cross-dataset lengths/batches are unequal. Full environment and repeat timings '
        'are retained in COST_RECORDS.json.']
    table(lines, ['Dataset', 'Model', 'Parameters (exact)', 'SynOps M/sequence', 'Dense GFLOPs/sequence', 'Arithmetic GFLOPs/sequence', 'Logical GOps/sequence'],
        [[LABEL[r['dataset']], r['model'], f"{r['total_parameters']['mean']:,.0f}",
          formatted(r['synops_per_sample'], 1e-6, 3), formatted(r['dense_gflops_test_padded'], digits=5),
          formatted(r['arithmetic_gflops_test_padded'], digits=5), formatted(r['logical_gops_test_padded'], digits=5)] for r in cost_groups])
    lines += ['SynOps are ideal dense-fan-out event-addition estimates, not GPU executed instructions or energy. '
        'The historical ECG evaluator excludes its first input projection from SynOps and records that '
        'projection separately as dense MACs under its "analog input" accounting convention. The mechanism '
        'analysis independently verifies signed ternary ECG validation inputs; this report preserves the '
        'historical compute convention rather than reclassifying its counts. Dense MAC=2 FLOPs. Arithmetic '
        'excludes nonlinearities/comparisons; logical operations count each such scalar operation once and '
        'are not hardware FLOPs. No joules or hardware energy were measured. Lower spikes alone does not '
        'prove lower energy or latency.']
    cost_deltas = []
    for ds in DATASETS:
        for pair, (parent, variant) in PAIRS.items():
            cost_deltas.append(dict(dataset=ds, pair=pair, **{key: paired(costs, ds, parent, variant, key)
                for key in ['spikes_total_per_sample', 'forward_ms_per_sample', 'synops_per_sample']}))
    table(lines, ['Dataset', 'Pair', 'Variant–parent hidden spikes/sequence', 'Variant–parent forward ms/sequence'],
        [[LABEL[r['dataset']], r['pair'], formatted(r['spikes_total_per_sample'], digits=1),
          formatted(r['forward_ms_per_sample'], digits=3)] for r in cost_deltas])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        for model in prepared['models']:
            rows = [r for r in flat if r['dataset'] == ds and r['model'] == model]
            ax.scatter([r['forward_ms_per_sample'] for r in rows], [100 * r['test_accuracy'] for r in rows],
                       label=model.replace('_SE_adLIF', ''), s=28)
        ax.set_title(LABEL[ds] + ': saved ordinary test'); ax.set_xlabel('Forward ms/sequence (historical sessions)')
        ax.set_ylabel('Historical final test accuracy (%)'); ax.legend(fontsize=7)
    fig.savefig(root / 'figures/saved_accuracy_and_timing.png', dpi=160); plt.close(fig)
    lines += ['', '![Historical accuracy and timing, individual seeds](figures/saved_accuracy_and_timing.png)', '',
        'The roughly 3 h 45 min stage-7 queue time is end-to-end workflow time; it is not any latency '
        'entry above and is not a new cost measurement.', '', '## Evidence, limitations and completion', '',
        'This synthesis closes only the registered exploratory reporting scope after user execution and '
        'verification. It does not establish independent population confirmation, exact upstream analytical '
        'figure reproduction, new-seed superiority, causal feature mediation or energy savings. '
        'Historical official-test exposure remains explicit. Additional CV-ISI, temporal-permutation '
        'experiments, independent data, learned-trace mediation and new controlled hardware measurements '
        'are possible follow-ups, not silently claimed results. The saved models and source-high baselines '
        'are preserved; highest-precision stage-7 diagnostics remain a separate numerical protocol.', '',
        'The complete source-file hash index, source completion ledgers, per-seed tables, environment '
        'records and machine-readable scope audit accompany this report. Source files are read-only; '
        'changing evidence or implementation requires a new output root.']
    table(lines, ['Dataset / pair', 'Original evidence'],
        [[LABEL[ds] + ' / ' + pair, link(root, name, 'saved evaluation / statistics')]
         for (ds, pair), name in prepared['reports'].items()])
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_csv(root, 'VALIDATION_PER_SEED.csv', prepared['validations'])
    write_csv(root, 'COST_AND_TEST_PER_SEED.csv', flat)
    write_csv(root, 'ASSOCIATIONS.csv', prepared['associations'])
    write_csv(root, 'INTERVENTIONS.csv', prepared['interventions'])
    write_csv(root, 'FEATURE_DISTRIBUTIONS.csv', feature_summary)
    write_csv(root, 'DIAGNOSTIC_ORIGINS.csv', origins)
    write_csv(root, 'PAIRED_COST_DIFFERENCES.csv', cost_deltas)
    raw_counts = [dict(dataset=r['dataset'], model=r['model'], seed=r['seed'],
                      **r['measurements']['totals']) for r in costs]
    write_csv(root, 'ACTIVITY_EXTENTS.csv', raw_counts)
    (root / 'COST_RECORDS.json').write_text(json.dumps(costs, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (root / 'AGGREGATES.json').write_text(json.dumps(dict(validation=validation_groups, costs=cost_groups,
        paired_cost_differences=cost_deltas), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    audit = dict(status='registered_synthesis_scope_complete', stage8_scoped_complete=True,
        ordinary_cost_records=len(costs), ordinary_model_dataset_groups=len(cost_groups),
        validation_pair_dataset_groups=len(validation_groups), primary_saved_associations=len(prepared['associations']),
        saved_intervention_effects=len(prepared['interventions']), diagnostic_origin_groups=len(origins),
        feature_records={ds: len(rows) for ds, rows in prepared['features'].items()},
        new_inference=False, training=False, official_test_access=False, new_fitting_or_resampling=False,
        upstream_analytical_figure_parity='unverified', independent_scientific_confirmation=False,
        limitations=['registered exploratory scope only', 'historical test exposure',
                     'dataset-specific units and losses', 'limited gradient cohorts',
                     'no population or causal guarantee', 'no energy measurement or controlled speed claim'])
    require((len(costs), len(cost_groups), len(validation_groups), len(prepared['associations']),
             len(prepared['interventions']), len(origins)) == (54, 18, 12, 24, 36, 24), 'Incomplete synthesis coverage')
    (root / 'STAGE8_SCOPE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return audit
