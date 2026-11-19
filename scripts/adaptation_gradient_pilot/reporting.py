"""Descriptive one-seed pilot reporting; no inference, fitting or confidence intervals."""
import statistics

import numpy as np


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(gradients, cases, models, ids):
    groups = []
    for model in models:
        rows = [gradients[model, i] for i in ids]
        groups.append(dict(model=model, records=len(rows),
            nonzero_physical_gradients=sum(r['physical_abs_gradient_sum'] > 0 for r in rows),
            mean_physical_abs_gradient_sum=mean([r['physical_abs_gradient_sum'] for r in rows]),
            mean_padding_abs_gradient_sum=mean([r['synthetic_padding_abs_gradient_sum'] for r in rows]),
            mean_burn_in_abs_gradient_sum=mean([r['burn_in_input_abs_gradient_sum'] for r in rows]),
            source_batch_prediction_mismatches=sum(r['clean']['prediction'] != r['original_batch_reference']['prediction'] for r in rows),
            max_absolute_source_batch_loss_drift=max(abs(r['source_batch_loss_difference']) for r in rows),
            max_autograd_forward_loss_drift=max(abs(r['autograd_forward_loss_difference']) for r in rows)))
    paired = []
    condition_rows = []
    guided_ids = []
    for case in cases:
        i = case['sample_id']
        if any(v['condition'] == 'mt_abs_gradient_deletion' for v in case['conditions']): guided_ids.append(i)
        for condition in case['conditions']:
            p, v = [next(r for r in condition['evaluations'] if r['model'] == m) for m in models]
            cp, cv = [gradients[m, i]['clean'] for m in models]
            condition_rows.extend(dict(sample_id=i, condition=condition['condition'], replicate=condition['replicate'], **r)
                                  for r in condition['evaluations'])
            benefit = p['loss']-v['loss']
            paired.append(dict(sample_id=i, seed=42, parent_model=models[0], variant_model=models[1],
                target=case['target'], speaker=case['speaker'],
                condition=condition['condition'], replicate=condition['replicate'],
                input_sha256=condition['input_sha256'], deletion_budget=condition['budget'],
                parent_loss=p['loss'], variant_loss=v['loss'], loss_benefit=benefit,
                clean_loss_benefit=cp['loss']-cv['loss'], benefit_change_from_clean=benefit-(cp['loss']-cv['loss']),
                parent_actual_loss_change=p['actual_loss_change'], variant_actual_loss_change=v['actual_loss_change'],
                parent_surrogate_linear_prediction=p['surrogate_linear_prediction'],
                variant_surrogate_linear_prediction=v['surrogate_linear_prediction']))
    effects = []
    for model in models:
        guided, random = [], []
        for i in guided_ids:
            selected = [r for r in condition_rows if r['sample_id'] == i and r['model'] == model]
            guided.append(abs(next(r['actual_loss_change'] for r in selected if r['condition'] == 'mt_abs_gradient_deletion')))
            random.append(mean([abs(r['actual_loss_change']) for r in selected if r['condition'] == 'exact_random_deletion']))
        effects.append(dict(model=model, common_eligible_record_ids=guided_ids, n_records=len(guided_ids),
            mean_abs_loss_change_guided=mean(guided), mean_abs_loss_change_random=mean(random),
            mean_guided_minus_random_abs_change=mean([a-b for a, b in zip(guided, random)]),
            meaning='same MT-guidance-eligible records; random scalar effects averaged over three realizations within record'))
    return dict(status='pilot_executed', n_training_seeds=1, seed=42, samples=40, gradient_records=len(gradients),
        groups=groups, guided_vs_random=effects, guided_eligible_records=len(guided_ids),
        paired_condition_records=len(paired), evaluation_records=len(condition_rows),
        all_zero_physical_gradient=all(r['physical_abs_gradient_sum'] == 0 for r in gradients.values()),
        stage7_complete=False, population_inference=False, input_optimization=False,
        limitations=['one pair, one seed and forty class-balanced validation records; not full stage 7',
            'SLAYER surrogate input derivatives are not true finite differences of hard spikes or causal attention',
            'microbatch references are a declared diagnostic mode, not original-batch validation accuracy',
            'MT-derived selection can couple input perturbations to its loss; inspect both models and random controls',
            'zero event-gradient guidance is explicitly ineligible; no linear/finite-change agreement is required',
            'no weight updates, optimized diagnostic input, significance test, population interval or official test']), paired


def write_figures(root, gradients, summary, models, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = root/'figures'
    directory.mkdir(exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in models:
        profiles = [gradients[model, i]['relative_time_abs_mass'] for i in ids
                    if gradients[model, i]['relative_time_abs_mass'] is not None]
        if profiles:
            ax.plot((np.arange(20)+.5)*5, np.mean(profiles, axis=0), label=f'{model}, n={len(profiles)}')
    ax.set(xlabel='Relative position within physical input (%)', ylabel='Within-record normalized |surrogate gradient| mass',
           title='One-seed SHD* tooling pilot; no input time resampling')
    if ax.lines: ax.legend()
    else: ax.text(.5, .5, 'All physical gradients are zero; normalized profiles undefined', transform=ax.transAxes, ha='center')
    fig.tight_layout();path=directory/'relative_time_surrogate_mass.png';fig.savefig(path, dpi=150);plt.close(fig);paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in models:
        rows = [gradients[model, i] for i in ids]
        ax.scatter(np.arange(len(rows)), [r['source_batch_loss_difference'] for r in rows], s=15, label=model)
    ax.axhline(0., color='black', linewidth=.7)
    ax.set(xlabel='Fixed pilot record order', ylabel='Single-record CE minus original-batch CE', title='Declared batching deviation')
    ax.legend();fig.tight_layout();path=directory/'batch_mode_loss_drift.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    eligible = summary['guided_eligible_records']
    if eligible:
        for j, group in enumerate(summary['guided_vs_random']):
            ax.bar(j-.17, group['mean_abs_loss_change_guided'], width=.34, label='MT-guided deletion' if j==0 else None)
            ax.bar(j+.17, group['mean_abs_loss_change_random'], width=.34, label='Matched random deletion' if j==0 else None)
        ax.set_xticks(range(len(models)), models)
        ax.legend()
    else: ax.text(.5,.5,'No eligible MT event gradients; guided contrast undefined',transform=ax.transAxes,ha='center')
    ax.set(ylabel='Mean absolute CE change (nats)', title=f'Equal event budget; matched eligible records n={eligible}; no population inference')
    fig.tight_layout();path=directory/'guided_vs_random_deletion.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    return paths
