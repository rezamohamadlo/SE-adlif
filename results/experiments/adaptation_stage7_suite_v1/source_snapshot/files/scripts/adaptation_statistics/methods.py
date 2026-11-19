"""Speaker-clustered exploratory regression and sensitivity calculations."""
import numpy as np


def within_speaker(values, speakers):
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    for group in np.unique(speakers):
        mask = speakers == group
        result[mask] -= values[mask].mean(axis=0)
    return result


def make_design(features, duration, rate, classes, speakers):
    continuous = np.column_stack((features, np.log1p(duration), np.log1p(rate)))
    if not np.isfinite(continuous).all() or np.any(duration <= 0) or np.any(rate < 0):
        raise ValueError('Invalid feature, duration or rate')
    mean, scale = continuous.mean(axis=0), continuous.std(axis=0, ddof=0)
    if np.any(scale == 0):
        raise ValueError('Constant feature/control cannot be estimated')
    standardized = (continuous - mean) / scale
    levels = np.unique(classes)
    dummies = np.column_stack([classes == label for label in levels[1:]])
    design = within_speaker(np.column_stack((standardized, dummies)), speakers)
    return design, dict(means=mean.tolist(), population_sd=scale.tolist(),
                        class_levels=levels.tolist(), dropped_reference_class=int(levels[0]),
                        speaker_effects='within-speaker demeaning of design and outcome')


def sufficient_statistics(design, outcomes, speakers, levels):
    gram, cross = [], []
    for level in levels:
        x, y = design[speakers == level], outcomes[speakers == level]
        gram.append(x.T @ x)
        cross.append(x.T @ y)
    return np.asarray(gram), np.asarray(cross)


def solve_clusters(gram, cross, weights, tolerance):
    matrix = np.einsum('g,gij->ij', weights, gram)
    rhs = np.einsum('g,gij->ij', weights, cross)
    eig = np.linalg.eigvalsh(matrix)
    if eig[-1] <= 0 or eig[0] <= tolerance * eig[-1]:
        return None
    return np.linalg.solve(matrix, rhs)


def cluster_analysis(design, outcomes, speakers, effects, all_speakers, specification):
    """Shared speaker draws preserve pairing over seeds and perturbation outcomes.

    Design/outcome are already demeaned per speaker. Class/continuous slopes
    remain estimable without retaining dummy columns for absent bootstrapped
    speakers. Scaling is frozen before resampling. effects uses all input rows.
    """
    if specification['draws'] < 100 or not 0 < specification['minimum_valid_fraction'] <= 1:
        raise ValueError('Invalid bootstrap budget/validity fraction')
    if (design.shape[0] != len(speakers) or outcomes.shape[0] != len(speakers) or
            effects.shape[0] != len(all_speakers) or outcomes.shape[1] != effects.shape[2]):
        raise ValueError('Mismatched analysis dimensions')
    if not all(np.isfinite(x).all() for x in (design, outcomes, effects)):
        raise ValueError('Nonfinite statistical inputs')
    levels = np.unique(all_speakers)
    if len(levels) < specification['minimum_speakers']:
        raise ValueError('Too few speakers for the specified analysis')
    gram, cross = sufficient_statistics(design, outcomes, speakers, levels)
    sums = np.asarray([effects[all_speakers == group].sum(axis=0) for group in levels])
    sizes = np.asarray([np.sum(all_speakers == group) for group in levels])
    tolerance = specification['rank_relative_tolerance']
    beta = solve_clusters(gram, cross, np.ones(len(levels)), tolerance)
    if beta is None:
        raise ValueError('Full design is rank deficient or ill conditioned')
    rng = np.random.default_rng(specification['seed'])
    samples, effect_samples, loo = [], [], []
    weights_record = []
    rejected = 0
    for _ in range(specification['draws']):
        weights = np.bincount(rng.integers(0, len(levels), size=len(levels)), minlength=len(levels))
        estimate = solve_clusters(gram, cross, weights, tolerance)
        if estimate is None:
            rejected += 1
            continue
        samples.append(estimate[:2])
        effect_samples.append(np.einsum('g,gij->ij', weights, sums) / (weights @ sizes))
        weights_record.append(weights)
    if len(samples) < specification['minimum_valid_fraction'] * specification['draws']:
        raise ValueError('Too many rank-deficient bootstrap draws; no intervals reported')
    for group in range(len(levels)):
        weights = np.ones(len(levels)); weights[group] = 0
        estimate = solve_clusters(gram, cross, weights, tolerance)
        loo.append(dict(omitted_speaker=int(levels[group]),
                        feature_coefficients=None if estimate is None else estimate[:2].tolist()))
    inverse = np.linalg.inv(gram.sum(axis=0))
    leverage = np.einsum('ij,jk,ik->i', design, inverse, design)
    leverage += np.asarray([1 / np.sum(speakers == group) for group in speakers])
    fitted = design @ beta
    residual = outcomes - fitted
    total = np.sum(outcomes**2, axis=0)
    ratio = np.full(outcomes.shape[1], np.nan)
    np.divide(np.sum(residual**2,axis=0), total, out=ratio, where=total > 0)
    r_squared = 1 - ratio
    return dict(beta=beta, bootstrap=np.asarray(samples), leverage=leverage, r_squared=r_squared,
                effects=effects.mean(axis=0), effect_bootstrap=np.asarray(effect_samples),
                cluster_weights=np.asarray(weights_record), loo=loo, rejected=rejected,
                design_condition_number=float(np.linalg.cond(design)))


def interval(draws, alpha, family_size):
    """Bonferroni-adjusted, approximate percentile interval (no exact guarantee)."""
    if family_size < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid interval family')
    tail = alpha / (2 * family_size)
    return np.quantile(draws, [tail, 1-tail], axis=0, method='linear')


def direction(low, high):
    if low > 0:
        return 'positive_within_fixed_cohort'
    if high < 0:
        return 'negative_within_fixed_cohort'
    return 'inconclusive'
