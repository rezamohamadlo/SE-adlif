"""Evaluate the frozen DTH/SHD* and SA-MT/ECG extension checkpoints; no training."""
import argparse
import hashlib
import json
import statistics

from project_paths import ROOT, local_path, portable_path, portable_config


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--group', choices=['dth_shd', 'sa_mt_ecg'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan_path = local_path(args.plan, must_exist=True)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    rows = [r for r in plan['checkpoints'] if r['group'] == args.group]
    if len(rows) != 7 or len({r['seed'] for r in rows}) != 7:
        raise ValueError('Expected seven distinct frozen seeds')
    output = plan_path.parent / args.group
    # Verify every identity before performing any inference or writing output.
    for row in rows:
        checkpoint = local_path(row['checkpoint'], must_exist=True)
        if digest(checkpoint) != row['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {checkpoint}')
        source = json.loads(local_path(row['validation_record'], must_exist=True).read_text())
        if portable_path(source['best_checkpoint']) != row['checkpoint']:
            raise ValueError('Validation-selected checkpoint changed')
        target = output / f"seed_{row['seed']}" / 'TEST_RESULTS.json'
        if target.exists():
            saved = json.loads(target.read_text())
            if saved.get('checkpoint_sha256') != row['sha256'] or saved.get('checkpoint') != row['checkpoint']:
                raise ValueError(f'Conflicting existing evaluation: {target}')
        print(f"{row['group']} seed {row['seed']}: {'existing' if target.exists() else 'pending'}", flush=True)
    if args.dry_run:
        return
    import hydra
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from models.pl_module import MLPSNN
    for row in rows:
        target = output / f"seed_{row['seed']}" / 'TEST_RESULTS.json'
        if target.exists():
            continue
        checkpoint = torch.load(local_path(row['checkpoint']), map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(portable_config(OmegaConf.to_container(
            OmegaConf.create(checkpoint['hyper_parameters']['cfg']), resolve=True)))
        if cfg.random_seed != row['seed']:
            raise ValueError('Checkpoint training seed mismatch')
        pl.seed_everything(cfg.random_seed, workers=True)
        torch.set_float32_matmul_precision('high')
        cfg.dataset.data_path = str(local_path(cfg.dataset.data_path, must_exist=True))
        cfg.dataset.num_workers = 0
        model = MLPSNN(cfg)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        data = hydra.utils.instantiate(cfg.dataset)
        device = torch.device(str(cfg.device))
        trainer = pl.Trainer(accelerator='gpu' if device.type == 'cuda' else 'cpu',
                             devices=[device.index or 0] if device.type == 'cuda' else 1,
                             logger=False, enable_checkpointing=False)
        metrics = trainer.test(model, datamodule=data)[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(checkpoint=row['checkpoint'], checkpoint_sha256=row['sha256'],
                                         seed=row['seed'], inference='ordinary', metrics=metrics), indent=2) + '\n')
        del trainer, model, data, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results = [dict(seed=r['seed'], accuracy_percent=100 * json.loads(
        (output / f"seed_{r['seed']}" / 'TEST_RESULTS.json').read_text())['metrics']['test_acc']) for r in rows]
    values = [r['accuracy_percent'] for r in results]
    summary = dict(group=args.group, n=len(values), results=results,
                   mean_percent=statistics.mean(values), sample_sd_pp=statistics.stdev(values))
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
