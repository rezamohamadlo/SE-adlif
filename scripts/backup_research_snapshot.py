"""Copy source and local research evidence to a new, hash-verified snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import portable_path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists():
        raise ValueError('Use a new snapshot directory; existing backups are preserved')
    paths = set()
    for options in (['--cached'], ['--others', '--exclude-standard']):
        output = subprocess.check_output(['git', 'ls-files', '-z', *options], cwd=ROOT)
        paths.update(Path(p.decode('utf-8')) for p in output.split(b'\0') if p)
    for name in ('results', 'archive', 'incoming-results'):
        paths.update(p.relative_to(ROOT) for p in (ROOT / name).rglob('*') if p.is_file())
    paths = sorted(p for p in paths if (ROOT / p).is_file()
                   and '__pycache__' not in p.parts
                   and not p.name.endswith('.lock'))
    entries = []
    target.mkdir(parents=True)
    for relative in paths:
        source, destination = ROOT / relative, target / relative
        before = digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(destination) != before or digest(source) != before:
            raise RuntimeError(f'Copy mismatch or source changed: {relative}')
        entries.append(dict(path=relative.as_posix(), size=source.stat().st_size, sha256=before))
    manifest = dict(destination=str(target),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    scope='Tracked/untracked nonignored files plus all results/archive/incoming-results artifacts. '
                          'Excludes environments, raw datasets, caches outside results, credentials, and locks.',
                    files=entries, total_bytes=sum(e['size'] for e in entries))
    evidence = ROOT / 'results/provenance/research-snapshot-2026-11-10.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(evidence, target / 'TRANSFER_MANIFEST.json')
    print(f'Verified {len(entries)} files, {manifest["total_bytes"]} bytes: {target}')


if __name__ == '__main__':
    main()
