"""Read-only package/CUDA/config checks; optionally write a portable setup report."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = {'SHD': ['data/SHDWrapper/shd_train.h5', 'data/SHDWrapper/shd_test.h5'],
        'SSC': ['data/SSCWrapper/ssc_train.h5', 'data/SSCWrapper/ssc_valid.h5', 'data/SSCWrapper/ssc_test.h5'],
        'ECG': ['data/QTDB/QTDB_train.mat', 'data/QTDB/QTDB_test.mat']}


def package_pins(path):
    pins = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '--')):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if not match:
            raise ValueError(f'Unsupported unpinned requirement: {line}')
        name, version = match.groups()
        if name in pins:
            raise ValueError(f'Duplicate requirement: {name}')
        pins[name] = version
    return pins


def mismatches(pins, installed):
    return {name: dict(expected=version, installed=installed.get(name))
            for name, version in pins.items() if installed.get(name) != version}


def data_inventory(root):
    return {dataset: {name: (root / name).is_file() for name in files}
            for dataset, files in DATA.items()}


def text_hash(path):
    """Ignore Git's Windows line-ending conversion when comparing configs."""
    return hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()


def verify(expected_prefix=None, expected_commit=None):
    errors = []
    if sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8:
        errors.append('Requires 64-bit Python 3.11')
    if sys.prefix == sys.base_prefix:
        errors.append('A project virtual environment is required')
    if expected_prefix and Path(sys.prefix).resolve() != Path(expected_prefix).resolve():
        errors.append('Interpreter is not the selected project environment')
    pins = package_pins(ROOT / 'requirements.txt')
    installed = {}
    for name in pins:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    drift = mismatches(pins, installed)
    if drift:
        errors.append('Package versions differ from requirements.txt; do not silently substitute versions')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if expected_commit and commit != expected_commit:
        errors.append('Git commit does not match the source PC')
    source_changes = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', 'HEAD', '--',
        'requirements.txt', 'run.py', 'runner.py', 'config', 'models', 'datasets', 'functional',
        'module', 'evaluation.yaml', 'evaluation_protocol.py', 'project_paths.py'], text=True).splitlines()
    if source_changes:
        errors.append('Tracked source/config changes differ from the recorded commit; preserve and resolve them before matching PCs')
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                  source_commit=commit, packages=installed, package_mismatches=drift,
                  tracked_source_changes=source_changes, config_hash_convention='UTF-8 text with normalized LF line endings',
                  requirements_sha256=text_hash(ROOT / 'requirements.txt'),
                  data=data_inventory(ROOT), checkpoint_files_present=any((ROOT / 'results').rglob('*.ckpt')),
                  configuration_sha256={p.relative_to(ROOT).as_posix(): text_hash(p)
                                        for p in sorted((ROOT / 'config').rglob('*.yaml'))})
    # Fail before importing an environment with missing/incompatible packages.
    if not errors:
        sys.path.insert(0, str(ROOT))
        try:
            import torch
            import tonic  # noqa: F401 -- compatibility import check
            import pytorch_lightning  # noqa: F401
            from hydra import initialize_config_dir, compose
            from omegaconf import OmegaConf
            from models.pl_module import MLPSNN
            if torch.version.cuda != '12.4' or not torch.cuda.is_available():
                raise RuntimeError('Requires a working NVIDIA GPU with the pinned CUDA 12.4 PyTorch wheel')
            # A real kernel tests driver/GPU support; nvidia-smi alone is insufficient.
            x = torch.ones((32, 32), device='cuda')
            actual = (x @ x).sum().item()
            torch.cuda.synchronize()
            if actual != 32768:
                raise RuntimeError('CUDA matrix smoke test failed')
            report['gpu'] = dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                                 cuda=torch.version.cuda, torch=torch.__version__)
            baseline = OmegaConf.load(ROOT / 'config/paper_baseline.yaml').paper_baseline
            report['paper_baseline_version'] = baseline.version
            report['resolved_preflight_configs'] = {}
            with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
                for ds in ['SHD', 'SSC', 'ECG']:
                    suffix = '_2layer' if ds == 'ECG' else ''
                    cfg = compose(config_name='main', overrides=[f'experiment={ds}_SE_adLIF{suffix}', f'+comparison={ds}'])
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                    report['resolved_preflight_configs'][ds] = resolved
                    expected = baseline.datasets[ds]
                    if (cfg.n_epochs != expected.n_epochs or cfg.dataset.batch_size != expected.batch_size
                            or cfg.lr != expected.lr or cfg.early_stopping != baseline.reproduction_policy.early_stopping):
                        raise RuntimeError(f'{ds} effective configuration differs from paper baseline')
                    # Constructor check only; no forward, training or datamodule/download.
                    import contextlib
                    import io
                    with contextlib.redirect_stdout(io.StringIO()):
                        model = MLPSNN(cfg)
                    del model
        except Exception as error:
            errors.append(f'Import/CUDA/config verification failed: {error}')
    report['errors'] = errors
    report['environment_ready'] = not errors
    report['datasets_ready'] = {ds: all(files.values()) for ds, files in report['data'].items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-prefix')
    parser.add_argument('--expected-commit')
    parser.add_argument('--output', help='Optional setup report inside the repository')
    args = parser.parse_args()
    report = verify(args.expected_prefix, args.expected_commit)
    if args.output:
        path = (ROOT / args.output).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Setup report must be inside this repository')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['environment_ready', 'datasets_ready', 'errors', 'package_mismatches']}, indent=2))
    return 0 if report['environment_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
