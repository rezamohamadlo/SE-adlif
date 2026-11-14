"""Validation-only staged-run guards and progress callbacks; no job launcher."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from project_paths import ROOT, local_path, portable_path, portable_config
from extended_seed_study import atomic_json, validation_rows, portable_checkpoint


def load_study(path):
    actual = local_path(path, must_exist=True)
    if not actual.is_relative_to(ROOT / 'config/studies'):
        raise ValueError('Study definition must be under config/studies')
    study = OmegaConf.to_container(OmegaConf.load(actual), resolve=True)
    if study['version'] not in ('complementary_mechanisms_v1','complementary_speaker_dev_v2','dth_complement_v1'):
        raise ValueError('Unknown staged comparison version')
    output = local_path(study['output_root'])
    if not output.is_relative_to(ROOT / 'results/experiments'):
        raise ValueError('Staged comparison outputs must be isolated experiments')
    study['_file'] = portable_path(actual)
    return study


def study_overrides(study, ceiling, model, dataset, seed):
    settings = study['datasets'][dataset]
    if model not in study['models'] or seed != study['training_seed']:
        raise ValueError('Discovery study requires its declared model and training seed')
    if ceiling not in settings['ceilings']:
        raise ValueError(f'Select a declared {dataset} epoch ceiling: {settings["ceilings"]}')
    if dataset != 'SHD':
        calibration_root=local_path(study['output_root']).parent
        for suffix in ('TRAINING_CALIBRATION','VALIDATION_CALIBRATION'):
            path=calibration_root/f'{dataset}_{suffix}.json'
            if not path.exists():
                raise ValueError(f'Transfer requires verified data and separate calibration: {path.name}')
            record=json.loads(path.read_text())
            expected=settings['batch_size'] if suffix=='TRAINING_CALIBRATION' else settings.get('validation_batch_size',settings['batch_size'])
            if record.get('selected_batch') != expected or record.get('status') != 'verified':
                raise ValueError('Transfer calibration is not verified for the frozen batch')
        if 'num_workers' in settings:
            path=calibration_root/f'{dataset}_WORKER_CALIBRATION.json'
            record=json.loads(path.read_text())
            if (record.get('status')!='verified'
                    or record.get('selected_num_workers')!=settings['num_workers']
                    or bool(record.get('spawn_safe_transforms'))!=bool(settings.get('spawn_safe_transforms'))):
                raise ValueError('Data loading requires its matched verified calibration')
    overrides=[f'++complementary_study={study["version"]}', f'++study_file={study["_file"]}',
            f'++study_idea={study["idea"]}', f'++study_stage={settings["stage"]}',
            f'++study_hypothesis={json.dumps(study.get("hypotheses",{}).get(model,study["hypothesis"]))}', f'++stage_ceiling={ceiling}',
            f'batch_size={settings["batch_size"]}', f'dataset.batch_size={settings["batch_size"]}',
            f'++dataset.validation_batch_size={settings.get("validation_batch_size", settings["batch_size"])}',
            '++dataset.validation_only_data=true', 'run_test=false', '++validation_only=true']
    if 'num_workers' in settings:
        overrides.append(f'dataset.num_workers={settings["num_workers"]}')
    if settings.get('spawn_safe_transforms'):
        overrides.append('++dataset.spawn_safe_transforms=true')
    if 'record_initial_state' in study:
        overrides.append(f'++study_record_initial_state={str(study["record_initial_state"]).lower()}')
    if 'training_objectives' in study:
        overrides += [f'++training_objective={study["training_objectives"][model]}',
                      f'++label_smoothing_epsilon={study["label_smoothing_epsilon"][model]}']
    if 'dropout_consistency_alpha' in study:
        overrides.append(f'++dropout_consistency_alpha={study["dropout_consistency_alpha"][model]}')
    if study.get('protocol_id') and dataset != 'SHD':
        overrides.append(f'++evaluation_protocol={study["protocol_id"]}_{dataset}')
    if study['version']=='complementary_speaker_dev_v2' and dataset=='SHD':
        from datasets.shd_speaker_split import load_manifest, DEVELOPMENT_PROTOCOL
        split=load_manifest(study['split_manifest'],'data',study['split_fingerprint'])
        if split['validation_speaker_ids']!=study['validation_speakers'] or split['role']!=study['role']:
            raise ValueError('Study pair differs from its predeclared frozen manifest')
        if study['role']=='pair_sensitivity':
            path=local_path(study['shortlist'])
            if not path.exists():
                raise ValueError('Sensitivity checks require a frozen primary-validation shortlist first')
            shortlist=json.loads(path.read_text())
            if shortlist.get('selected_using')!='clean_primary_speaker_validation' or (model!='SE_adLIF' and model not in shortlist['models']):
                raise ValueError('Do not evaluate unshortlisted methods or search for a favorable pair')
        overrides += ['dataset.validate_on=speaker_holdout',f'dataset.cache_namespace={DEVELOPMENT_PROTOCOL}',
                      f'++dataset.speaker_split_manifest={study["split_manifest"]}',
                      f'++dataset.speaker_split_fingerprint={study["split_fingerprint"]}',
                      f'++evaluation_protocol={study.get("protocol_id",study["version"])}_{study["pair_id"]}',
                      f'++study_arm={json.dumps(study["arm_labels"][model])}']
    return overrides


def validate_config(cfg):
    baseline = OmegaConf.to_container(OmegaConf.load(ROOT / 'config/paper_baseline.yaml'))['paper_baseline']
    dataset = cfg['paper_baseline_dataset']
    spec = load_study(cfg['study_file'])
    s = spec['datasets'][dataset]
    variant=cfg.get('model_variant')
    if (type(spec.get('record_initial_state', False)) is not bool
            or cfg.get('study_record_initial_state', False) is not spec.get('record_initial_state', False)):
        raise ValueError('Initial-state recording differs from the declared study')
    if 'training_objectives' in spec and variant not in spec['training_objectives']:
        raise ValueError('Declared objective study requires a known model_variant')
    expected_objective=spec.get('training_objectives',{}).get(variant,'classification')
    if cfg.get('training_objective','classification') != expected_objective:
        raise ValueError('Training objective differs from the declared arm')
    if 'label_smoothing_epsilon' in spec:
        if cfg.get('label_smoothing_epsilon',0) != spec['label_smoothing_epsilon'][cfg['model_variant']]:
            raise ValueError('Label smoothing epsilon differs from the declared arm')
    if 'dropout_consistency_alpha' in spec:
        if cfg.get('dropout_consistency_alpha') != spec['dropout_consistency_alpha'][variant]:
            raise ValueError('Dropout consistency alpha differs from the declared arm')
    if 'hidden_cells' in spec:
        expected_cell = spec['hidden_cells'][variant]
        for layer in ('l1', 'l2'):
            if cfg[layer]['cell'] != expected_cell:
                raise ValueError('Hidden neuron differs from the declared arm')
            for key, expected in spec.get('hidden_cell_settings', {}).get(expected_cell, {}).items():
                if cfg[layer].get(key) != expected:
                    raise ValueError(f'Hidden neuron setting {layer}.{key} differs from the declared arm')
    if spec['version'] in ('complementary_speaker_dev_v2','dth_complement_v1'):
        expected_probability=spec['spike_deletion'].get(cfg['model_variant'],0)
        if cfg.get('spike_delete_probability',0)!=expected_probability:
            raise ValueError('Training event deletion differs from the declared factorial arm')
        if 'input_fir_modes' in spec:
            expected_mode=spec['input_fir_modes'][cfg['model_variant']]
            actual=cfg.get('input_fir')
            if ((expected_mode is None and actual)
                    or (expected_mode is not None and
                        (not actual or actual.get('mode','causal')!=expected_mode))):
                raise ValueError('Input filter mode differs from the declared mechanism control')
    if (cfg['paper_baseline'] != baseline or cfg['paper_baseline_version'] != baseline['version']
            or cfg['run_test'] is not False or cfg.get('validation_only') is not True
            or cfg['random_seed'] != spec['training_seed'] or cfg['early_stopping']
            or cfg.get('init_ckpt_path') or not cfg['dataset'].get('validation_only_data')):
        raise ValueError('Unsafe staged validation-only configuration')
    for key in ('n_epochs', 'lr'):
        if cfg[key] != baseline['datasets'][dataset][key]:
            raise ValueError(f'Staging must retain the full-budget {key}')
    for key in ('dropout', 'gradient_clip_val'):
        if cfg[key] != baseline['common'][key]:
            raise ValueError(f'Changed shared {key}')
    for key in ('scheduler', 'patience', 'factor'):
        if cfg[key] != baseline['reproduction_policy'][key]:
            raise ValueError(f'Changed full-run schedule {key}')
    if cfg['batch_size'] != s['batch_size'] or cfg['dataset']['batch_size'] != cfg['batch_size']:
        raise ValueError('Training batch differs from frozen matched study')
    if cfg['dataset'].get('validation_batch_size') != s.get('validation_batch_size',s['batch_size']):
        raise ValueError('Validation batch differs from calibrated matched study')
    if cfg['dataset']['num_workers'] != s.get('num_workers',baseline['reproduction_policy']['num_workers']):
        raise ValueError('Data loading differs from the frozen matched study')
    if bool(cfg['dataset'].get('spawn_safe_transforms',False)) != bool(s.get('spawn_safe_transforms',False)):
        raise ValueError('Preprocessing implementation differs from the frozen study')
    if cfg['stage_ceiling'] not in s['ceilings'] or cfg['stage_ceiling'] > cfg['n_epochs']:
        raise ValueError('Invalid staged ceiling')
    if dataset=='SHD':
        if cfg['dataset']['random_seed']!=42 or cfg['dataset'].get('additional_test_set_validation'):
            raise ValueError('SHD split seed or test isolation changed')
        if spec['version']=='complementary_speaker_dev_v2':
            from datasets.shd_speaker_split import load_manifest,DEVELOPMENT_PROTOCOL
            if (cfg['dataset']['validate_on']!='speaker_holdout' or cfg['dataset']['cache_namespace']!=DEVELOPMENT_PROTOCOL
                    or cfg['dataset']['speaker_split_manifest']!=spec['split_manifest']
                    or cfg['dataset']['speaker_split_fingerprint']!=spec['split_fingerprint']):
                raise ValueError('Speaker development cannot use random-split or another pair weights')
            split=load_manifest(spec['split_manifest'],cfg['dataset']['data_path'],spec['split_fingerprint'])
            if split['validation_speaker_ids']!=spec['validation_speakers']:
                raise ValueError('Held-out speaker pair changed')
        elif cfg['dataset']['validate_on']!=.2:
            raise ValueError('Canonical SHD* split changed')
    if dataset == 'ECG' and (cfg['dataset']['valid_fraction']!=.05 or cfg['dataset']['random_seed']!=42):
        raise ValueError('ECG established row split changed')
    expected_target={'SHD':'datasets.shd.SHDLDM','SSC':'datasets.ssc.SSCLDM','ECG':'datasets.ecg.ECGLDM'}[dataset]
    if cfg['dataset']['_target_']!=expected_target:
        raise ValueError('Dataset implementation differs from the study task')


def canonical(cfg):
    result = portable_config(dict(cfg))
    for key in ('ckpt_path', 'stage_ceiling'):
        result.pop(key, None)
    return result


def forward_identity(dataset='SHD'):
    names = ['models/pl_module.py', 'models/ours/input_fir.py', 'models/alif.py',
             'models/helpers.py', 'models/li.py', 'module/tau_trainers.py',
             {'SHD':'datasets/shd.py','SSC':'datasets/ssc.py','ECG':'datasets/ecg.py'}[dataset],
             'datasets/utils/pad_tensors.py', 'config/paper_baseline.yaml']
    names += ['functional/count_thinning.py','models/ours/count_thinning.py']
    names += ['models/ours/label_smoothing.py', 'models/ours/dth_adlif.py']
    names += ['models/ours/static_threshold_adlif.py']
    names += ['models/ours/current_norm_adlif.py']
    names += ['models/ours/dropout_consistency.py']
    if dataset=='SHD': names += ['datasets/shd_speaker_split.py']
    return {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in names}


def checkpoint(path):
    return torch.load(local_path(path, must_exist=True), map_location='cpu', weights_only=False)


def validate_resume_checkpoint(cfg):
    if cfg.get('ckpt_path'):
        path=local_path(cfg['ckpt_path'],must_exist=True)
        if path!=(local_path(cfg['logdir'])/'ckpt/last.ckpt'):
            raise ValueError('Resume must use this run\'s own last checkpoint, never other split weights')
        state=checkpoint(path)
        previous=OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']),resolve=True)
        if canonical(previous)!=canonical(cfg):
            raise ValueError('Checkpoint saw a different split/settings; fresh training is required')


def inspect_run(directory, cfg):
    validate_config(cfg)
    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()):
        return 'new', None
    manifest = directory/'FULL_RUN_CONFIG.json'
    if not manifest.exists() or canonical(json.loads(manifest.read_text())) != canonical(cfg):
        raise ValueError(f'Changed/missing staged manifest: {directory}')
    identity = directory/'MODEL_SOURCE_IDENTITY.json'
    if identity.exists() and json.loads(identity.read_text()) != forward_identity(cfg['paper_baseline_dataset']):
        raise ValueError('Forward implementation changed; preserve old evidence and use a versioned study')
    if any((directory/name).exists() for name in ['TEST_RESULTS.json', 'completed.txt', 'COST_RESULTS.json']):
        raise ValueError('Test artifacts in staged discovery output')
    rows = validation_rows(directory, cfg)
    last = directory/'ckpt/last.ckpt'
    if not last.exists():
        # A real startup failure is fresh, never weights continuation.
        if any((directory/'ckpt').glob('*.ckpt')):
            raise ValueError('Checkpoints exist without resumable last.ckpt')
        if rows:
            raise ValueError('Metrics without a resumable checkpoint; preserve and diagnose')
        return 'new', None
    state = checkpoint(last)
    saved = OmegaConf.to_container(OmegaConf.create(state['hyper_parameters']['cfg']), resolve=True)
    if canonical(saved) != canonical(cfg):
        raise ValueError('Checkpoint differs from current matched protocol')
    completed = state['epoch']+1
    if rows and rows[-1][0] >= completed:
        raise ValueError('Validation metrics ahead of resumable checkpoint')
    ceiling = cfg['stage_ceiling']
    if completed > ceiling:
        raise ValueError('Cannot resume a completed later stage into an earlier ceiling')
    if completed == ceiling:
        receipt = directory/'stages'/f'{ceiling:03d}'/'RESULT.json'
        if not receipt.exists():
            raise ValueError('Checkpoint reached ceiling but lacks a verified stage receipt; diagnose before continuation')
        record = json.loads(receipt.read_text())
        if record['completed_epochs'] != ceiling or record['status'] != 'stage_complete':
            raise ValueError('Invalid stage receipt')
        for key in ('best_checkpoint', 'last_checkpoint'):
            with local_path(record[key], must_exist=True).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != record[key+'_sha256']:
                    raise ValueError('Stage checkpoint identity mismatch')
        return 'complete', None
    return 'resume', last


def metric(metrics, key):
    value = metrics.get(key)
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value) if value is not None else None


class StudyProgress(pl.Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.times = []
        self.best = -math.inf
        self.best_epoch = None

    def on_fit_start(self, trainer, model):
        self.directory = local_path(self.cfg.logdir)
        self.started = time.perf_counter()
        study=load_study(self.cfg.study_file)
        self.study_root = local_path(study['output_root']).parent
        self.study_plan = study['plan']
        self.protocol=str(self.cfg.get('evaluation_protocol','unspecified'))
        self.pair_id=load_study(self.cfg.study_file).get('pair_id')
        if self.cfg.get('study_record_initial_state', False) and not self.cfg.ckpt_path:
            if trainer.global_step != 0:
                raise ValueError('Fresh initial-state recording requires zero optimizer updates')
            from module.initial_state import record_initial_state
            initial = record_initial_state(self.directory, model, self.cfg, trainer.precision)
            print(f'[INITIAL_STATE] arm={self.cfg.model_variant} seed={self.cfg.random_seed} '
                  f'sha256={initial["model_state_sha256"]} tensors={initial["tensor_count"]}', flush=True)
        # on_fit_start precedes Lightning's loop-state restoration. The saved
        # checkpoint, not trainer.current_epoch at this hook, proves progress.
        restored_epochs = checkpoint(self.cfg.ckpt_path)['epoch']+1 if self.cfg.ckpt_path else 0
        atomic_json(self.study_root/'STATUS.json',dict(status='running',training_active=True,pid=os.getpid(),
                    stage=int(self.cfg.study_stage),idea=self.cfg.study_idea,
                    run_directory=portable_path(self.directory),completed_epochs=restored_epochs,
                    ceiling=int(self.cfg.stage_ceiling),protocol=self.protocol,pair_id=self.pair_id))
        for epoch, value in validation_rows(self.directory, self.cfg):
            if value > self.best:
                self.best, self.best_epoch = value, epoch
        atomic_json(self.directory/'MODEL_SOURCE_IDENTITY.json', forward_identity(self.cfg.paper_baseline_dataset))
        print(f'[STAGE {self.cfg.study_stage}][{self.cfg.study_idea}] '
              f'{self.cfg.study_hypothesis}\n'
              f'dataset={self.cfg.paper_baseline_dataset} arm={self.cfg.model_variant} '
              f'training_seed={self.cfg.random_seed} split_seed={self.cfg.dataset.get("random_seed", "official")} '
              f'batch={self.cfg.batch_size} precision={trainer.precision} lr={self.cfg.lr} '
              f'ceiling={self.cfg.stage_ceiling} full_schedule={self.cfg.n_epochs} '
              f'output={portable_path(self.directory)} mode={"resume" if self.cfg.ckpt_path else "fresh"}', flush=True)
        if self.pair_id:
            spec=load_study(self.cfg.study_file)
            split=json.loads(local_path(spec['split_manifest']).read_text())
            print(f'[PROTOCOL] {self.protocol}; pair={self.pair_id}; '
                  f'train_speakers={split["subsets"]["train"]["speaker_ids"]}; '
                  f'validation_speakers={split["validation_speaker_ids"]}; '
                  f'train/validation={split["subsets"]["train"]["sample_count"]}/{split["subsets"]["validation"]["sample_count"]}; '
                  f'fingerprint={split["fingerprint"]}; validation=clean; official_test=disabled',flush=True)

    def on_train_epoch_start(self, trainer, model):
        self.epoch_started = time.perf_counter()
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(model.device)

    def on_train_epoch_end(self, trainer, model):
        completed = trainer.current_epoch+1
        values = {key: metric(trainer.callback_metrics, key) for key in
                  ['train_acc_epoch', 'train_loss_epoch', 'val_acc', 'val_loss']}
        hard_ce = metric(trainer.callback_metrics, 'train_hard_ce')
        if hard_ce is not None:
            values['train_hard_ce'] = hard_ce
        if values['val_acc'] is not None and values['val_acc'] > self.best:
            self.best, self.best_epoch = values['val_acc'], trainer.current_epoch
        seconds = time.perf_counter()-self.epoch_started
        self.times.append(seconds)
        record = dict(status='running', stage=int(self.cfg.study_stage), idea=self.cfg.study_idea,
                      dataset=self.cfg.paper_baseline_dataset, model=self.cfg.model_variant,
                      seed=int(self.cfg.random_seed), completed_epochs=completed,
                      epoch_index=trainer.current_epoch, ceiling=int(self.cfg.stage_ceiling),
                      full_budget=int(self.cfg.n_epochs), **values, best_validation_accuracy=self.best,
                      best_epoch_index=self.best_epoch, lr=trainer.optimizers[0].param_groups[0]['lr'],
                      batch=int(self.cfg.batch_size), updates=int(trainer.global_step), epoch_seconds=seconds,
                      peak_vram_gib=torch.cuda.max_memory_allocated(model.device)/2**30 if model.device.type=='cuda' else None,
                      estimated_remaining_seconds=max(0,self.cfg.stage_ceiling-completed)*statistics.mean(self.times),
                      elapsed_this_launch_seconds=time.perf_counter()-self.started)
        record.update(protocol=self.protocol,pair_id=self.pair_id,validation_clean=True,
                      training_event_deletion_probability=float(self.cfg.get('spike_delete_probability',0)))
        if self.cfg.get('training_objective'):
            record.update(training_objective=self.cfg.training_objective,
                          label_smoothing_epsilon=float(self.cfg.get('label_smoothing_epsilon',0)),
                          validation_loss_definition='ordinary hard-label cross-entropy')
        if self.cfg.get('training_objective') == 'classification_dropout_consistency_v1':
            record.update(dropout_consistency_alpha=float(self.cfg.dropout_consistency_alpha),
                          training_loss_definition='mean of two supervised losses plus alpha times symmetric KL',
                          training_accuracy_definition='first stochastic view; logged once')
            for name in ('train_supervised_loss', 'train_dropout_symmetric_kl'):
                value = trainer.callback_metrics.get(name)
                record[name] = float(value) if value is not None else None
        atomic_json(self.directory/'STATUS.json', record)
        atomic_json(self.study_root/'STATUS.json',dict(record,training_active=True,pid=os.getpid(),
                    run_directory=portable_path(self.directory),plan=self.study_plan))
        with (self.directory/'EPOCH_PROGRESS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        available = lambda v: f'{v:.6g}' if v is not None else 'unavailable'
        label=('SHD-speaker-dev' if self.cfg.dataset.get('validate_on')=='speaker_holdout' else 'SHD*') if self.cfg.paper_baseline_dataset=='SHD' else self.cfg.paper_baseline_dataset
        print(f'[STAGE {self.cfg.study_stage}][{self.cfg.study_idea}][{label}]'
              f'[seed={self.cfg.random_seed}][completed={completed}/{self.cfg.stage_ceiling}] '
              f'train_acc={available(values["train_acc_epoch"])} train_loss={available(values["train_loss_epoch"])} '
              + (f'train_hard_ce={available(hard_ce)} ' if hard_ce is not None else '') +
              f'val_acc={available(values["val_acc"])} val_loss={available(values["val_loss"])} '
              f'best_val={self.best:.6g} best_epoch_index={self.best_epoch} lr={record["lr"]:.6g} '
              f'batch={record["batch"]} updates={record["updates"]} epoch_seconds={seconds:.2f} '
              f'peak_vram_gib={available(record["peak_vram_gib"])} ETA_est_seconds={record["estimated_remaining_seconds"]:.0f}',flush=True)
        if completed >= self.cfg.stage_ceiling:
            trainer.should_stop = True

    def on_exception(self, trainer, model, exception):
        atomic_json(local_path(self.cfg.logdir)/'STATUS.json',dict(status='technical_failure', exception=repr(exception)))
        print(f'[DECISION] BLOCKED\nEvidence: {exception!r}\nNext action: diagnose the actual failure; preserve checkpoints.',flush=True)


def finish_stage(cfg, trainer):
    validate_config(OmegaConf.to_container(cfg, resolve=True))
    directory = local_path(cfg.logdir)
    last = directory/'ckpt/last.ckpt'
    state = checkpoint(last)
    completed = state['epoch']+1
    if completed != cfg.stage_ceiling:
        raise RuntimeError(f'Stage did not reach its exact ceiling: {completed}/{cfg.stage_ceiling}')
    best = local_path(trainer.checkpoint_callback.best_model_path, must_exist=True)
    selected = checkpoint(best)
    rows = validation_rows(directory, cfg)
    score = float(trainer.checkpoint_callback.best_model_score)
    if len(rows)!=completed or not math.isclose(max(v for _,v in rows),score,abs_tol=1e-7):
        raise ValueError('Stage validation history disagrees with selected checkpoint')
    snapshot = directory/'stages'/f'{completed:03d}'
    snapshot.mkdir(parents=True,exist_ok=True)
    record = dict(status='stage_complete', metric='validation_accuracy', completed_epochs=completed,
                  full_budget=int(cfg.n_epochs), preliminary=completed<cfg.n_epochs,
                  model=cfg.model_variant, dataset=cfg.paper_baseline_dataset, seed=int(cfg.random_seed),
                  best_validation_accuracy=score, selected_epoch_index=selected['epoch'], test_evaluated=False)
    record.update(protocol=str(cfg.evaluation_protocol),split_fingerprint=cfg.dataset.get('speaker_split_fingerprint'),
                  dataset_label='SHD speaker-disjoint development' if cfg.dataset.get('validate_on')=='speaker_holdout' else ('SHD*' if cfg.paper_baseline_dataset=='SHD' else cfg.paper_baseline_dataset))
    if cfg.get('training_objective'):
        record.update(training_objective=cfg.training_objective,
                      label_smoothing_epsilon=float(cfg.get('label_smoothing_epsilon',0)),
                      validation_loss_definition='ordinary hard-label cross-entropy')
    if cfg.get('training_objective') == 'classification_dropout_consistency_v1':
        record.update(dropout_consistency_alpha=float(cfg.dropout_consistency_alpha),
                      training_loss_definition='mean of two supervised losses plus alpha times symmetric KL',
                      training_accuracy_definition='first stochastic view; logged once')
    for name, source in [('best_checkpoint',best),('last_checkpoint',last)]:
        target=snapshot/('best.ckpt' if name=='best_checkpoint' else 'last.ckpt')
        shutil.copy2(source,target)
        record[name]=portable_path(target)
        with target.open('rb') as stream:
            record[name+'_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    atomic_json(snapshot/'RESULT.json',record)
    atomic_json(snapshot/'RESOLVED_CONFIG.json',portable_config(OmegaConf.to_container(cfg,resolve=True)))
    status_path=directory/'STATUS.json'
    status=json.loads(status_path.read_text())
    status.update(status='stage_complete',receipt=portable_path(snapshot/'RESULT.json'))
    atomic_json(status_path,status)
    study_root=local_path(load_study(cfg.study_file)['output_root']).parent
    atomic_json(study_root/'STATUS.json',dict(status,training_active=False,run_directory=portable_path(directory)))
    (directory/'TRAINING_SUMMARY.md').write_text(
        f'# {cfg.model_variant}: staged validation study\n\n'
        f'Protocol: {record["protocol"]}; dataset label: {record["dataset_label"]}.\n\n'
        f'Completed {completed} epochs (indices 0 through {completed-1}); full budget {cfg.n_epochs}. '
        + (f'This is a preliminary screen, not full-budget completion.\n\n' if record['preliminary']
         else 'The full training budget is complete; official-test evaluation remains pending.\n\n') +
        f'Best validation accuracy: {score*100:.6f}%; selected epoch index {selected["epoch"]}. '
        'No official test evaluation.\n\n'
        + ('Training loss is the smoothed objective; `train_hard_ce` is logged separately. '
           'Validation/inference use ordinary hard-label CE.\n\n'
           if cfg.get('training_objective')=='classification_label_smoothing_v1' else '') +
        ('Training loss averages two supervised dropout passes plus weighted symmetric KL; '
         'accuracy uses the first pass only. Hard CE, supervised loss and KL are logged separately. '
         'Validation/inference use one ordinary forward and hard-label CE.\n\n'
         if cfg.get('training_objective')=='classification_dropout_consistency_v1' else '') +
        f'Authoritative receipt: `stages/{completed:03d}/RESULT.json`; '
        'per-epoch evidence: `EPOCH_PROGRESS.jsonl` and `logs/mlp_snn/version_0/metrics.csv`.\n\n'
        'Rerun/verify the same bounded stage (an interrupted run resumes its own last checkpoint):\n\n'
        f'```powershell\n.\\.ven\\Scripts\\python.exe runner.py --models {cfg.model_variant} '
        f'--datasets {cfg.paper_baseline_dataset} --seeds {cfg.random_seed} '
        f'--study-file {cfg.study_file} --stage-ceiling {cfg.stage_ceiling}\n```\n',encoding='utf-8')
    print(f'[STAGE_COMPLETE]\nEvidence: {completed} completed epochs; best validation={score:.8f}, selected epoch index={selected["epoch"]}. '
          f'\nNext action: compare the declared matched control before any extension or transfer.',flush=True)
