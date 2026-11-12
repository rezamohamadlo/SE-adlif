"""Draft pure configuration guard; no file loading, data access or training."""
from copy import deepcopy
import json

ARMS = {'Baseline', 'A', 'B', 'A+B'}
DATASETS = {'SHD', 'SSC', 'ECG'}
SEEDS = {42, 123, 456}
# These identify the run rather than its numerical training configuration.
RUN_METADATA = {
    'random_seed', 'logdir', 'exp_name', 'launcher_logdir', 'launcher_script',
    'run_mode', 'study_hypothesis', 'study_stage', 'ckpt_path',
}
# Declared mechanism differences cannot waive these matched-condition checks.
PROTECTED = {
    'dataset', 'datadir', 'lr', 'batch_size', 'n_epochs', 'max_epochs',
    'precision', 'device', 'optimizer', 'weight_decay', 'betas', 'scheduler',
    'scheduler_min_lr', 'factor', 'patience', 'gradient_clip_val',
    'dropout', 'alpha', 'dt', 'accumulate_grad_batches', 'gradient_clip_algorithm',
    'early_stopping', 'early_stopping_patience', 'early_stopping_min_delta',
    'loss_agg', 'paper_baseline', 'paper_baseline_version',
    'paper_baseline_dataset', 'evaluation_protocol', 'tracking_metric',
    'tracking_mode', 'stage_ceiling',
}


def _stable(config):
    if not isinstance(config, dict):
        raise ValueError('Expected a resolved JSON configuration mapping')
    return {k: deepcopy(v) for k, v in config.items() if k not in RUN_METADATA}


def _comparison_protocol(config):
    """Ignore only the exact objective annotation for shared-condition checks.

    Each full per-arm protocol string is still checked against its frozen
    config below. The underlying split/optimization protocol cannot be waived.
    """
    protocol = config.get('evaluation_protocol')
    objective = config.get('training_objective', 'classification')
    if (isinstance(protocol, str) and isinstance(objective, str)
            and objective not in ('', 'classification')
            and protocol.endswith('_' + objective)):
        return protocol[:-len(objective)-1]
    return protocol


def _remove_path(config, path):
    parts = path.split('.')
    node = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def validate_frozen_configurations(plan, cohort):
    """Validate all 36 checkpoint configs against pre-frozen arm specifications.

    cohort contains (arm, dataset, training_seed, resolved_checkpoint_cfg).
    The caller must verify checkpoint hashes, source identities and receipts,
    and pass portable resolved configs read from those exact checkpoints.
    This helper does not replace those independent provenance checks.
    """
    frozen = plan.get('frozen_configurations')
    differences = plan.get('mechanism_difference_paths')
    if (not isinstance(frozen, dict) or set(frozen) != DATASETS
            or not isinstance(differences, list) or not differences
            or any(not isinstance(p, str) for p in differences)
            or len(set(differences)) != len(differences)):
        raise ValueError('Final testing requires frozen per-arm configs and explicit mechanism paths')
    for path in differences:
        if (not isinstance(path, str) or not path or any(not p for p in path.split('.'))
                or path.split('.')[0] in PROTECTED | RUN_METADATA):
            raise ValueError('Mechanism paths cannot waive matched training conditions')
    for dataset in DATASETS:
        if not isinstance(frozen[dataset], dict) or set(frozen[dataset]) != ARMS:
            raise ValueError('Frozen configurations must specify every arm for every dataset')
        signatures = set()
        shared = None
        for arm in ARMS:
            spec = _stable(frozen[dataset][arm])
            computational = deepcopy(spec)
            # Factory selection must also have explicit mechanism/architecture
            # settings; renaming a baseline or its deviation note is not an arm.
            computational.pop('model_variant', None)
            computational.pop('paper_baseline_deviations', None)
            signatures.add(json.dumps(computational, sort_keys=True))
            common = deepcopy(spec)
            if 'evaluation_protocol' in common:
                common['evaluation_protocol'] = _comparison_protocol(spec)
            # Deviations remain verified against each exact frozen arm config.
            common.pop('paper_baseline_deviations', None)
            for path in differences:
                _remove_path(common, path)
            if shared is None:
                shared = common
            elif common != shared:
                raise ValueError('Frozen arms differ outside their declared mechanisms')
        if len(signatures) != 4:
            raise ValueError('Four arm labels must describe four distinct configurations')
    seen = set()
    for arm, dataset, seed, config in cohort:
        identity = (arm, dataset, seed)
        if arm not in ARMS or dataset not in DATASETS or seed not in SEEDS or identity in seen:
            raise ValueError('Unexpected or duplicate final cohort identity')
        seen.add(identity)
        if config.get('random_seed') != seed or config.get('paper_baseline_dataset') != dataset:
            raise ValueError('Checkpoint dataset/training seed differs from cohort identity')
        if config.get('init_ckpt_path'):
            raise ValueError('Final fresh-training comparison cannot use weights-only initialization')
        if dataset == 'SHD':
            data = config.get('dataset', {})
            if (data.get('validate_on') != 0.2 or data.get('random_seed') != 42
                    or data.get('speaker_split_manifest') or data.get('speaker_split_fingerprint')):
                raise ValueError('Canonical SHD* requires its own random20percent split, seed42')
        if _stable(config) != _stable(frozen[dataset][arm]):
            raise ValueError('Checkpoint settings differ from the frozen arm configuration')
    expected = {(a, d, s) for a in ARMS for d in DATASETS for s in SEEDS}
    if seen != expected:
        raise ValueError('Final configuration cohort is incomplete')
    return {'verified_configurations': len(seen), 'matched_shared_conditions': True}
