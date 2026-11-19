"""User-launched saved-evidence report; agent-safe --check-only/--verify-only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_stage8.evidence import SPEC, digest, prepare, require, sha


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def lock(root):
    # OS-held lock; the presence of a released lock file does not indicate a writer.
    with (root / '.report.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verify(directory):
    root = local_path(directory, must_exist=True)
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    complete = json.loads((root / 'COMPLETE.json').read_text(encoding='utf-8'))
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == digest(manifest),
            'Synthesis completion identity differs')
    require(complete['stage8_scoped_complete'] is True, 'Synthesis scope is not complete')
    for name, expected in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and sha(path) == expected, 'Synthesis artifact differs: ' + name)
    for name, expected in manifest['source_files_sha256'].items():
        require(sha(local_path(name, must_exist=True)) == expected, 'Synthesis source differs: ' + name)
    print('Stage-8 saved synthesis verified:', len(complete['outputs']), 'outputs; no model execution.', flush=True)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    spec = json.loads(local_path(SPEC).read_text(encoding='utf-8'))
    if args.verify_only:
        verify(args.output_root or spec['output_root'])
        return 0
    prepared = prepare(args.output_root)
    root = prepared['root']
    if args.check_only:
        print('Stage-8 preflight passed: 54 saved cost/checkpoint/history identities, 12 comparison packages, '
              'source ledgers and all report inputs available. No tests, aggregation, figures or model execution.', flush=True)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with lock(root):
        if (root / 'COMPLETE.json').exists():
            verify(root)
            return 0
        manifest = prepared['manifest']
        write_json(root / 'MANIFEST.json', manifest)
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_stage8'],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (root / 'CONTRACT_TESTS.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
        require(tests.returncode == 0, 'Report contract tests failed; inspect CONTRACT_TESTS.txt')
        from scripts.adaptation_stage8.reporting import render
        audit = render(prepared)
        # Detect changes during report generation, including source/control records.
        for name, expected in manifest['source_files_sha256'].items():
            require(sha(local_path(name, must_exist=True)) == expected, 'Evidence changed during report generation: ' + name)
        outputs = {}
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.name not in ('.report.lock', 'COMPLETE.json') and not path.name.endswith('.tmp'):
                outputs[portable_path(path)] = sha(path)
        write_json(root / 'COMPLETE.json', dict(status='complete', manifest_sha256=digest(manifest),
            completed_utc=datetime.now(timezone.utc).isoformat(), outputs=outputs,
            stage8_scoped_complete=audit['stage8_scoped_complete'], new_inference=False, training=False))
        verify(root)
        print('Report:', portable_path(root / 'SUMMARY.md'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
