"""Descriptive fixed-cohort associations, never patient-population inference."""
import statistics

import numpy as np

from scripts.adaptation_ecg.data import require

FEATURES = ('fast_slow_contrast', 'autocorr_tau_steps')


def design(features, specification):
    keep = [i for i, r in enumerate(features) if all(r[k] is not None and np.isfinite(r[k]) for k in FEATURES)]
    excluded = [dict(sample_id=r['sample_id'], fast_slow_reason=r['fast_slow_reason'],
                     autocorr_reason=r['autocorr_reason'], autocorr_censored=r['autocorr_censored'])
                for i, r in enumerate(features) if i not in keep]
    metadata = dict(complete_case_ids=[features[i]['sample_id'] for i in keep], exclusions=excluded,
        dropped_controls=[], scales={}, time_unit='model_step', inference=False,
        scope='conditional descriptive association in these validation records and three fixed trained seeds')
    if not keep:
        return None, keep, dict(metadata, status='not_estimable', reason='no_joint_feature_complete_cases')
    rows = [features[i] for i in keep]
    columns, names = [np.ones(len(rows))], ['intercept']
    tol = specification['rank_relative_tolerance']
    for key in FEATURES + ('log1p_events_per_step', 'log1p_stored_steps'):
        values = np.array([np.log1p(r['events_per_step']) if key == 'log1p_events_per_step' else
                           np.log1p(r['stored_steps']) if key == 'log1p_stored_steps' else r[key] for r in rows])
        mean, sd = float(values.mean()), float(values.std(ddof=0))
        if sd <= tol:
            if key in FEATURES:
                return None, keep, dict(metadata, status='not_estimable', reason='constant_primary_feature_'+key)
            metadata['dropped_controls'].append(dict(name=key, reason='constant_on_complete_clean_cohort'))
            continue
        columns.append((values-mean)/sd)
        names.append(key)
        metadata['scales'][key] = dict(mean=mean, population_sd=sd)
    for k in range(1, 6):
        values = np.array([r['class_fractions'][k] for r in rows])
        name = f'class_fraction_{k}'
        if np.ptp(values) <= tol:
            metadata['dropped_controls'].append(dict(name=name, reason='constant_on_complete_clean_cohort'))
        else:
            columns.append(values)
            names.append(name)
    x = np.column_stack(columns)
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.sum(singular > tol*singular[0]))
    metadata.update(columns=names, n_records=len(rows), n_columns=x.shape[1], rank=rank,
                    singular_values=singular.tolist(),
                    condition_number=None if singular[-1] == 0 else float(singular[0]/singular[-1]),
                    residual_degrees_of_freedom=len(rows)-x.shape[1])
    if rank != x.shape[1] or len(rows)-x.shape[1] < specification['minimum_residual_df']:
        return None, keep, dict(metadata, status='not_estimable', reason='rank_or_residual_df_guard')
    return x, keep, dict(metadata, status='eligible')


def associations(features, pair_rows, seeds, specification):
    x, keep, meta = design(features, specification)
    if x is None:
        return dict(design=meta, status='not_estimable', inference=False, intervals=None,
                    meaning='diagnostic analysis completed without inventing an unidentified coefficient')
    ys = []
    for seed in seeds:
        clean = {r['sample_id']: r for r in pair_rows if r['seed'] == seed and r['condition'] == 'clean'}
        require(set(clean) == {r['sample_id'] for r in features}, 'Clean outcome/features coverage differs')
        ys.append([clean[features[i]['sample_id']]['loss_benefit'] for i in keep])
    y = np.asarray(ys).T
    tol = specification['rank_relative_tolerance']
    beta = np.linalg.lstsq(x, y, rcond=tol)[0]
    per_seed = [dict(seed=s, coefficients=dict(zip(meta['columns'], beta[:, j].tolist()))) for j, s in enumerate(seeds)]
    summary = {k: dict(mean=statistics.mean(r['coefficients'][k] for r in per_seed),
        sample_sd_across_training_seeds=statistics.stdev(r['coefficients'][k] for r in per_seed)) for k in FEATURES}
    fitted = x @ beta
    residual = y-fitted
    loo = []
    for omitted in range(len(x)):
        retain = np.arange(len(x)) != omitted
        singular = np.linalg.svd(x[retain], compute_uv=False)
        rank = int(np.sum(singular > tol*singular[0]))
        coefficients = None if rank != x.shape[1] else np.linalg.lstsq(x[retain], y[retain], rcond=tol)[0][1:3].tolist()
        loo.append(dict(omitted_record=features[keep[omitted]]['sample_id'], rank=rank,
                        feature_coefficients_by_seed=coefficients))
    return dict(status='descriptive_analysis_complete', design=meta, per_seed=per_seed,
        feature_summary=summary, residual_sum_of_squares_per_seed=(residual**2).sum(0).tolist(),
        leverage=np.sum(x*np.linalg.pinv(x, rcond=tol).T, axis=1).tolist(),
        leave_one_record_out=loo, leave_one_out_scaling='full complete-case design held fixed',
        n_training_seeds=len(seeds), inference=False, intervals=None, p_values=None,
        limitation='Patient/record dependence is unknown; row influence is not independent-patient validation. '
                   'No bootstrap, population confidence interval, significance or causal claim.')


def pair_records(parent, variant, clean_parent, clean_variant, features):
    ids = [r['sample_id'] for r in parent]
    require(bool(ids) and len(set(ids)) == len(ids) and
            all([r['sample_id'] for r in rows] == ids for rows in (variant, clean_parent, clean_variant)),
            'Paired ECG record coverage/order differs')
    result = []
    for p, v, cp, cv in zip(parent, variant, clean_parent, clean_variant):
        for a, b in ((p, v), (cp, cv)):
            require(all(a[k] == b[k] for k in ('sample_id', 'seed', 'condition', 'replicate', 'input_sha256',
                'targets_sha256', 'blocks_sha256', 'mask_sha256', 'time_map_sha256', 'source_targets_sha256', 'valid_steps')),
                'Parents/variants do not share an identical mapped input/label/mask contract')
        require(cp['condition'] == cv['condition'] == 'clean' and cp['replicate'] == cv['replicate'] == 0 and
                all(p[k] == cp[k] == cv[k] for k in ('sample_id', 'seed', 'source_targets_sha256')) and
                p['model'] == cp['model'] and v['model'] == cv['model'], 'Mapped and clean records have different sources')
        f = features[p['condition'], p['replicate']][p['sample_id']]
        require(all(p[k] == f[k] for k in ('input_sha256', 'targets_sha256', 'mask_sha256', 'time_map_sha256')),
                'Feature/prediction identities differ')
        benefit, clean = p['loss']-v['loss'], cp['loss']-cv['loss']
        result.append(dict(sample_id=p['sample_id'], seed=p['seed'], condition=p['condition'], replicate=p['replicate'],
            parent=p['model'], variant=v['model'], input_sha256=p['input_sha256'], targets_sha256=p['targets_sha256'],
            blocks_sha256=p['blocks_sha256'], source_targets_sha256=p['source_targets_sha256'], mask_sha256=p['mask_sha256'],
            time_map_sha256=p['time_map_sha256'], valid_steps=p['valid_steps'], parent_loss=p['loss'], variant_loss=v['loss'],
            parent_accuracy=p['accuracy'], variant_accuracy=v['accuracy'], accuracy_gap_pp=100*(v['accuracy']-p['accuracy']),
            loss_benefit=benefit, clean_loss_benefit=clean, benefit_change_from_clean=benefit-clean,
            total_events=f['total_events'], source_total_events=f['source_total_events'],
            event_count_change=f['event_count_change'], fast_slow_contrast=f['fast_slow_contrast'],
            autocorr_tau_steps=f['autocorr_tau_steps'], autocorr_censored=f['autocorr_censored']))
    return result
