"""Portable verification of immutable historical and current analysis records."""
from pathlib import Path

from project_paths import local_path, portable_path
from scripts.adaptation_analysis.pilot import digest, read_json, sha256


def resolve_historical_path(value, index):
    for prefix, replacement in sorted(index.get('path_aliases', {}).items(), key=lambda item: -len(item[0])):
        if value == prefix or value.startswith(prefix + '/'):
            return replacement + value[len(prefix):]
    return value


def experiment_directory(value):
    if '..' in str(value).replace('\\', '/').split('/'):
        raise ValueError('Output traversal is not permitted')
    path = local_path(value).resolve()
    allowed = local_path('results/experiments').resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError('Choose a study directory inside results/experiments')
    if any(path.rglob('*.ckpt')) or any(path.rglob('metrics.csv')):
        raise ValueError('Analysis destination contains training evidence')
    return path


def verify_completion(directory, relocations=None, filename='PILOT_COMPLETE.json'):
    directory = Path(directory)
    complete = read_json(directory / filename)
    manifest = read_json(directory / 'MANIFEST.json')
    if complete['status'] != 'complete' or complete['manifest_sha256'] != digest(manifest):
        raise ValueError('Completion identity differs')
    checked = {}
    for recorded, expected in complete['outputs'].items():
        actual = local_path((relocations or {}).get(recorded, recorded), must_exist=True)
        if sha256(actual) != expected:
            raise ValueError(f'Completed output changed: {portable_path(actual)}')
        checked[recorded] = portable_path(actual)
    return checked


def verify_package(directory):
    directory = experiment_directory(directory)
    index = directory / 'ARTIFACT_INDEX.json'
    if not index.exists():
        return {'full_validation': verify_completion(directory, filename='COMPLETE.json')}
    index = read_json(index)
    checked = {}
    for analysis, record in index['analyses'].items():
        path = local_path(record['directory'], must_exist=True)
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Indexed analysis escapes the package')
        for target in record['relocations'].values():
            if not local_path(target).resolve().is_relative_to(directory):
                raise ValueError('Indexed output escapes the package')
        if sha256(path / 'PILOT_COMPLETE.json') != record['completion_sha256']:
            raise ValueError('Indexed completion changed')
        checked[analysis] = verify_completion(path, record['relocations'])
    for recorded, expected in index.get('review_sha256', {}).items():
        if sha256(local_path(recorded, must_exist=True)) != expected:
            raise ValueError('Historical review changed')
    return checked
