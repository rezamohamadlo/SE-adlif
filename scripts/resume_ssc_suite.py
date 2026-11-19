"""Resume the frozen SSC suite with scoped Windows JSON-write retry compatibility."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
import uuid

from project_paths import ROOT, local_path, portable_path
from scripts.ssc_resume_io import AtomicJSONWriter, filesystem_preflight


def check_imports():
    import statistics
    expected = (Path(sysconfig.get_path('stdlib')) / 'statistics.py').resolve()
    if Path(statistics.__file__).resolve() != expected or not callable(statistics.mean):
        raise ValueError('Launch from the repository root with -m scripts.resume_ssc_suite')


def source_hashes():
    files = ['scripts/resume_ssc_suite.py', 'scripts/ssc_resume_io.py', 'tests/test_ssc_resume_io.py']
    return {name: hashlib.sha256(local_path(name).read_bytes()).hexdigest() for name in files}


def check_frozen_sources(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['source_sha256'].items():
        actual = hashlib.sha256(local_path(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != checksum:
            raise ValueError('Frozen scientific source changed: ' + name)
    return len(manifest['source_sha256'])


def verify_compatibility(root):
    from scripts.adaptation_analysis import pilot
    path = root / 'IO_COMPATIBILITY_COMPLETE.json'
    if not path.exists():
        raise ValueError('No completed I/O compatibility record; the resumed execution is not complete')
    record = pilot.read_json(path)
    if record['status'] != 'complete' or record['suite_completion_sha256'] != pilot.sha256(root / 'COMPLETE.json'):
        raise ValueError('Compatibility/suite completion identity differs')
    for name, checksum in record['outputs'].items():
        actual = local_path(name, must_exist=True)
        if not actual.is_relative_to(root) or pilot.sha256(actual) != checksum:
            raise ValueError('Compatibility provenance is missing, changed or outside the suite')
    print('I/O compatibility completion and provenance verified.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', default='results/experiments/adaptation_ssc_suite_v2')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    check_imports()
    from scripts.adaptation_analysis import pilot
    from scripts.adaptation_analysis.records import experiment_directory
    root = experiment_directory(args.output_root)
    if not (root / 'MANIFEST.json').is_file():
        raise ValueError('This compatibility launcher resumes an existing suite; recorded MANIFEST.json is required')
    core_args = ['--device', args.device, '--output-root', portable_path(root)]
    if args.verify_only:
        from scripts.adaptation_ssc_suite.run import main as core_main
        result = core_main(core_args + ['--verify-only'])
        if (root / 'IO_COMPATIBILITY_COMPLETE.json').exists():
            verify_compatibility(root)
        else:
            print('No I/O compatibility completion record is present; only core-suite integrity was verified.')
        return result
    frozen_count = check_frozen_sources(root)
    probe = filesystem_preflight(root, pilot.write_json)
    print(f'Atomic JSON preflight passed on the result volume; {frozen_count} frozen source hashes unchanged. '
          'Only disposable probe files were written.', flush=True)
    if args.check_only:
        from scripts.adaptation_ssc_suite.run import main as core_main
        return core_main(core_args + ['--check-only'])

    # The compatibility lock also protects callers before the frozen queue acquires
    # its own existing .analysis.lock. Neither lock is removed or bypassed.
    with pilot.study_lock(root / '.io-compatibility.lock'):
        if (root / 'COMPLETE.json').exists():
            from scripts.adaptation_ssc_suite.run import main as core_main
            result = core_main(core_args + ['--verify-only'])
            if (root / 'IO_COMPATIBILITY_COMPLETE.json').exists():
                verify_compatibility(root)
            return result
        original_writer = pilot.write_json
        writer = AtomicJSONWriter(root, original_writer)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        attempt = root / 'io_compatibility' / name
        attempt.mkdir(parents=True)
        record = dict(status='starting', execution_root=portable_path(root), started_utc=datetime.now(timezone.utc).isoformat(),
            compatibility_source_sha256=source_hashes(), unchanged_frozen_source_count=frozen_count,
            policy=dict(scope='JSON destinations inside this suite only', replacement_attempts=writer.attempts,
                        retry_winerrors=[5, 32, 33], unique_same_directory_temporary_files=True,
                        serialization='Original pilot.write_json JSON options, UTF-8 and native newline behavior',
                        fallback_delete_or_inplace_overwrite=False, training_or_numerical_change=False),
            filesystem_preflight=probe, command=['.venv/Scripts/python.exe', '-m', 'scripts.resume_ssc_suite', *core_args])
        writer(attempt / 'PROVENANCE.json', record)
        started = time.monotonic()
        try:
            command = [sys.executable, '-m', 'unittest', 'tests.test_ssc_resume_io']
            print('Running user-launched atomic I/O compatibility contracts before the frozen queue.', flush=True)
            tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (attempt / 'io_contract_tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
            writer(attempt / 'IO_CONTRACT_TESTS.json', dict(status='passed' if tests.returncode == 0 else 'failed',
                   returncode=tests.returncode, module='tests.test_ssc_resume_io'))
            if tests.returncode:
                print(tests.stdout + tests.stderr, file=sys.stderr)
                raise ValueError('I/O compatibility contracts failed; frozen queue was not started')
            # Install before importing the queue/report modules, so their imported
            # write_json aliases also use this scoped I/O-only implementation.
            pilot.write_json = writer
            from scripts.adaptation_ssc_suite.run import main as core_main
            from scripts.adaptation_ssc_suite import reporting as suite_reporting
            if suite_reporting.write_json is not writer:
                raise ValueError('Report modules were imported before I/O compatibility; launch in a fresh Python process')
            result = core_main(core_args)
            if result != 0:
                raise RuntimeError('Frozen queue returned nonzero status')
            record.update(status='complete', elapsed_seconds=time.monotonic()-started,
                          json_writes=writer.writes, retry_events=writer.retries)
            writer(attempt / 'PROVENANCE.json', record)
            outputs = {portable_path(p): pilot.sha256(p) for p in attempt.iterdir() if p.is_file() and p.suffix != '.tmp'}
            writer(root / 'IO_COMPATIBILITY_COMPLETE.json', dict(status='complete',
                suite_completion_sha256=pilot.sha256(root / 'COMPLETE.json'), outputs=outputs,
                compatibility_source_sha256=source_hashes()))
            verify_compatibility(root)
            return result
        except BaseException as error:
            record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__, winerror=getattr(error, 'winerror', None),
                          elapsed_seconds=time.monotonic()-started, json_writes=writer.writes, retry_events=writer.retries)
            try:
                writer(attempt / 'PROVENANCE.json', record)
            except OSError:
                print('Could not update I/O compatibility provenance; original error and temporary payloads are retained.', file=sys.stderr)
            raise
        finally:
            pilot.write_json = original_writer


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('Error: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
