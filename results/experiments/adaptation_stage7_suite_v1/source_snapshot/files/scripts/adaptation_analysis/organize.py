"""Organize an existing local pilot by analysis, preserving immutable records."""
import shutil
from pathlib import Path

from project_paths import local_path, portable_path
from scripts.adaptation_analysis.pilot import read_json, sha256, study_lock, write_json
from scripts.adaptation_analysis.records import experiment_directory, verify_completion, verify_package


PACKAGE = 'results/experiments/adaptation_mechanism_analysis_local_recheck_v1'
ORIGINAL = 'results/experiments/adaptation_mechanism_analysis_v1/pc1'
BACKUP = 'archive/pull-backups/20261112-073929/' + ORIGINAL


def organize():
    root = experiment_directory(PACKAGE)
    aliases = {BACKUP: PACKAGE + '/clean', ORIGINAL: PACKAGE + '/clean',
               PACKAGE + '/pc2': PACKAGE + '/time_scaling',
               PACKAGE + '/pc3': PACKAGE + '/spike_deletion'}
    with study_lock(root / '.analysis.lock'):
        if (root / 'ARTIFACT_INDEX.json').exists():
            index = read_json(root / 'ARTIFACT_INDEX.json')
            index['path_aliases'] = aliases
            write_json(root / 'ARTIFACT_INDEX.json', index)
            return verify_package(root)
        plans = [('clean', local_path(BACKUP), None),
                 ('time_scaling', root / 'pc2', root / 'time_scaling'),
                 ('spike_deletion', root / 'pc3', root / 'spike_deletion')]
        # Validate every source and destination before changing the layout.
        checked = {}
        for name, source, destination in plans:
            if destination and (not source.resolve().is_relative_to(root) or
                                not destination.resolve().is_relative_to(root)):
                raise ValueError('Relocation escapes the intended package')
            if destination and destination.exists():
                raise ValueError('Destination exists; preserve it and inspect the layout')
            if name == 'clean':
                if (root / name).exists():
                    raise ValueError('Clean destination already exists')
                complete = read_json(source / 'PILOT_COMPLETE.json')
                mapping = {p: portable_path(source / Path(p).name)
                           for p in complete['outputs'] if (source / Path(p).name).exists()}
            else:
                mapping = {}
            checked[name] = verify_completion(source, mapping)
        analyses = {}
        for name, source, destination in plans:
            target = root / name
            mapping = {}
            if name == 'clean':
                target.mkdir()
                for recorded, actual in checked[name].items():
                    relative = Path(recorded).relative_to(ORIGINAL)
                    dest = target / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path(actual), dest)
                    mapping[recorded] = portable_path(dest)
                shutil.copy2(source / 'PILOT_COMPLETE.json', target / 'PILOT_COMPLETE.json')
                if (source / 'PROGRESS.json').exists():
                    shutil.copy2(source / 'PROGRESS.json', target / 'PROGRESS.json')
            else:
                source.rename(destination)
                for recorded in checked[name]:
                    relative = Path(recorded).relative_to(portable_path(source))
                    mapping[recorded] = portable_path(target / relative)
            verify_completion(target, mapping)
            analyses[name] = dict(directory=portable_path(target), relocations=mapping,
                                  completion_sha256=sha256(target / 'PILOT_COMPLETE.json'))
        reviews = {portable_path(p): sha256(p) for folder in ('stage3_review', 'stage4_review')
                   for p in (root / folder).glob('*') if p.is_file()}
        write_json(root / 'ARTIFACT_INDEX.json', dict(
            schema='adaptation_analysis_relocation_v1', analyses=analyses, review_sha256=reviews, path_aliases=aliases,
            notes='Original manifests, completion records and reviews are immutable. Resolve historical output paths through this index. Clean metadata and rows are now self-contained; no archive backup is required to verify this package.'))
        (root / 'README.md').write_text(
            '# SHD* validation pilot: portable analysis package\n\n'
            'The 256-sample cohort and training seeds 42, 123, 456 are unchanged.\n\n'
            '- `clean/`: original local clean evaluation, including its matching metadata and detailed rows.\n'
            '- `time_scaling/`: locally regenerated timing evaluation.\n'
            '- `spike_deletion/`: locally regenerated deletion evaluation.\n'
            '- `stage3_review/`, `stage4_review/`: historical audit records, preserved byte for byte.\n\n'
            '`ARTIFACT_INDEX.json` maps historical recorded paths to the current layout. '
            'Original PC labels inside immutable provenance identify historical analysis assignments, not required hardware. '
            'Historical review paths must also be resolved through this index. '
            'No result values, manifests, completion identities or original evaluation files were rewritten.\n\n'
            'Share this ENTIRE directory, including ignored `diagnostic_rows/`, using the same repository-relative layout. '
            'Git alone omits the detailed rows. The archive backup is no longer needed to verify this package.\n\n'
            'Verify from the checkout root:\n\n```powershell\n'
            '.\\.venv\\Scripts\\python.exe scripts/adaptation_analysis/evaluate.py --verify-only '
            '--output-root ' + PACKAGE + '\n```\n', encoding='utf-8')
        return verify_package(root)


if __name__ == '__main__':
    print({name: len(paths) for name, paths in organize().items()})
