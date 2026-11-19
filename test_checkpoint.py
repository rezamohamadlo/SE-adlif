"""Evaluate a trusted local classification checkpoint without training."""
import argparse
import json
import hashlib
import contextlib
import io
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from models.pl_module import MLPSNN, build_training_model
from project_paths import local_path, portable_path, portable_config

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_study_evaluation(plan, checkpoint_path, checkpoint, cfg, output):
    """A frozen receipt authorizes a particular checkpoint, never a test winner."""
    if cfg.get('complementary_study')=='complementary_speaker_dev_v2':
        raise ValueError('Speaker-development weights are not canonical test evidence; train the frozen SHD* comparison separately')
    if (plan.get('status')!='frozen' or plan.get('purpose') not in ('exploratory','final')
            or plan.get('test_selects_ideas') is not False or not output):
        raise ValueError('Study testing requires a frozen evaluation plan and separate output directory')
    directory=local_path(output)
    if not directory.is_relative_to(ROOT/'results/experiments/complementary_mechanisms_v1/evaluations'):
        raise ValueError('Study test evidence must remain in the isolated evaluations directory')
    key=portable_path(checkpoint_path)
    entry=plan.get('checkpoints',{}).get(key)
    if not entry or entry.get('sha256')!=sha256(checkpoint_path):
        raise ValueError('Checkpoint is absent from the frozen plan or its hash differs')
    from module.study_control import forward_identity
    if entry.get('model_source_sha256') != forward_identity(cfg.paper_baseline_dataset):
        raise ValueError('Frozen evaluation model/data implementation differs from its recorded identity')
    receipt=json.loads(local_path(entry['stage_receipt'],must_exist=True).read_text(encoding='utf-8'))
    if (receipt['best_checkpoint']!=key or receipt['best_checkpoint_sha256']!=entry['sha256']
            or receipt['selected_epoch_index']!=checkpoint['epoch']
            or receipt['model']!=cfg.model_variant or receipt['seed']!=cfg.random_seed
            or receipt['dataset']!=cfg.paper_baseline_dataset):
        raise ValueError('Evaluation must use the exact validation-selected stage checkpoint')
    if plan['purpose']=='final':
        arms={'Baseline','A','B','A+B'}
        datasets={'SHD','SSC','ECG'}
        seeds={42,123,456}
        if (set(plan.get('arms',[]))!=arms or set(plan.get('datasets',[]))!=datasets
                or set(plan.get('training_seeds',[]))!=seeds
                or plan.get('architecture_and_settings_frozen') is not True):
            raise ValueError('Final evidence requires all four frozen arms, three datasets and three matching seeds')
        cohorts=set()
        for path,item in plan['checkpoints'].items():
            result=json.loads(local_path(item['stage_receipt'],must_exist=True).read_text(encoding='utf-8'))
            identity=(item.get('arm'),result.get('dataset'),result.get('seed'))
            budget={'SHD':300,'SSC':40,'ECG':400}.get(result.get('dataset'))
            if (identity in cohorts or result.get('status')!='stage_complete' or result.get('preliminary') is not False
                    or result.get('completed_epochs')!=budget or result.get('full_budget')!=budget
                    or result.get('best_checkpoint')!=path or result.get('best_checkpoint_sha256')!=item.get('sha256')
                    or sha256(local_path(path,must_exist=True))!=item['sha256']):
                raise ValueError('Final plan contains incomplete, duplicated or incompatible training evidence')
            cohorts.add(identity)
        if cohorts!={(a,d,s) for a in arms for d in datasets for s in seeds}:
            raise ValueError('Final plan omits part of the four-arm replicated comparison')
    return dict(purpose=plan['purpose'],completed_training_epochs=receipt['completed_epochs'],
                full_budget=receipt['full_budget'],preliminary=receipt['preliminary'],
                checkpoint_sha256=entry['sha256'],selected_epoch_index=checkpoint['epoch'],
                model=cfg.model_variant,dataset=cfg.paper_baseline_dataset,seed=int(cfg.random_seed),
                test_selects_ideas=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        help='Save standalone test evidence separately from training artifacts')
    parser.add_argument('--evaluation-plan',type=Path,
                        help='Required frozen plan for the complementary study; exploratory/final scope is recorded')
    args = parser.parse_args()
    checkpoint_path = local_path(args.checkpoint, must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
        OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
    study_evidence=None
    if cfg.get('complementary_study'):
        if not args.evaluation_plan:
            raise ValueError('Study checkpoints require --evaluation-plan; test access remains gated')
        plan_path=local_path(args.evaluation_plan,must_exist=True)
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        study_evidence=validate_study_evaluation(plan,checkpoint_path,checkpoint,cfg,args.output_dir)
        study_evidence.update(evaluation_plan=portable_path(plan_path),evaluation_plan_sha256=sha256(plan_path))
        test_name={'SHD':'SHDWrapper/shd_test.h5','SSC':'SSCWrapper/ssc_test.h5','ECG':'QTDB/QTDB_test.mat'}[cfg.paper_baseline_dataset]
        test_path=local_path(cfg.dataset.data_path)/test_name
        if not test_path.is_file():
            raise FileNotFoundError(f'Missing exact official test file: {portable_path(test_path)}')
        study_evidence.update(test_data=portable_path(test_path),test_data_sha256=sha256(test_path))
        previous=local_path(args.output_dir)/'TEST_RESULTS.json'
        if previous.exists():
            saved=json.loads(previous.read_text(encoding='utf-8'))
            if saved.get('study_evidence')!=study_evidence or saved.get('checkpoint')!=portable_path(checkpoint_path):
                raise ValueError('Output directory belongs to a different evaluation; preserve it')
            print(f'Already evaluated: {100*saved["metrics"]["test_acc"]:.6f}%');return
        cfg.dataset.validation_only_data=False
    pl.seed_everything(cfg.random_seed, workers=True)
    torch.set_float32_matmul_precision('high')
    # Standalone execution has no Hydra working directory to resolve data paths.
    cfg.dataset.data_path = str(local_path(cfg.dataset.data_path))
    cfg.dataset.num_workers = 0
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_training_model(cfg,default_class=MLPSNN)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    device = torch.device(str(cfg.device))
    trainer = pl.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=[device.index or 0] if device.type == 'cuda' else 1,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=not bool(study_evidence),enable_model_summary=not bool(study_evidence),
    )
    results = trainer.test(model, datamodule=datamodule)[0]
    directory = local_path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent
    directory.mkdir(parents=True, exist_ok=True)
    evidence=dict(checkpoint=portable_path(checkpoint_path),metrics=results)
    if study_evidence is not None:
        evidence.update(study_evidence=study_evidence,ordinary_inference=True,test_time_augmentation=False)
    (directory / 'TEST_RESULTS.json').write_text(
        json.dumps(evidence, indent=2) + '\n',
        encoding='utf-8',
    )
    summary_path = directory / 'TRAINING_SUMMARY.md'
    if summary_path.exists():
        marker = '\n## Final test evaluation\n'
        summary = summary_path.read_text(encoding='utf-8').split(marker, 1)[0].rstrip()
        section = (
            f'{marker}\nThe selected validation checkpoint was evaluated without further training.\n\n'
            f'Checkpoint: `{checkpoint_path.name}`\n\n'
            '| Metric | Value |\n|---|---:|\n'
            f"| Test accuracy | **{100 * results['test_acc']:.2f}%** |\n"
            f"| Test loss | **{results['test_loss']:.4f}** |\n"
        )
        summary_path.write_text(summary + '\n' + section, encoding='utf-8')
    print(f"Test accuracy: {100 * results['test_acc']:.2f}%")
    print(f'Results saved in {directory}')
    if (directory / 'SWEEP_CONFIG.json').exists() and (directory.parent / 'LEADERBOARD.md').exists():
        from tune_shd import report
        report(directory.parent)


if __name__ == '__main__':
    main()
