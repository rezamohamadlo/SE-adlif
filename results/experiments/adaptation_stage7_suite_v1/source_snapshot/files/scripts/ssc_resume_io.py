"""Bounded Windows atomic-JSON compatibility, separate from frozen experiment code."""
import json
import os
from pathlib import Path
import tempfile
import time


class AtomicJSONWriter:
    """Preserve the original JSON representation and atomic replacement semantics."""
    def __init__(self, root, fallback, *, replace=None, sleep=None, attempts=12):
        if attempts < 1:
            raise ValueError('At least one atomic replacement attempt is required')
        self.root = Path(root).resolve()
        self.fallback = fallback
        self.replace = replace or os.replace
        self.sleep = sleep or time.sleep
        self.attempts = attempts
        self.writes = 0
        self.retries = []
        self.last_temporary = None

    def __call__(self, path, data):
        destination = Path(path).resolve()
        if destination.suffix.lower() != '.json' or not destination.is_relative_to(self.root):
            return self.fallback(path, data)
        # Same dumps options, UTF-8 encoding and native newline handling as pilot.write_json.
        payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
        descriptor, name = tempfile.mkstemp(prefix='.' + destination.name + '.ssc-',
                                            suffix='.tmp', dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        self.last_temporary = temporary
        temporary.write_text(payload, encoding='utf-8')
        for attempt in range(self.attempts):
            try:
                self.replace(temporary, destination)
                self.writes += 1
                return
            except PermissionError as error:
                retryable = getattr(error, 'winerror', None) in (5, 32, 33)
                if not retryable or attempt + 1 == self.attempts:
                    error.add_note('Atomic JSON publication stopped; destination was not deleted or rewritten in place. '
                                   'The uncommitted temporary payload is retained at ' + str(temporary))
                    raise
                delay = min(.05 * 2**attempt, 1.)
                self.retries.append(dict(destination=destination.relative_to(self.root).as_posix(),
                                         winerror=error.winerror, attempt=attempt + 1, delay_seconds=delay))
                if attempt == 0:
                    print('Windows denied atomic JSON replacement; retrying briefly: ' + destination.name, flush=True)
                self.sleep(delay)


def filesystem_preflight(root, fallback):
    """Probe only disposable metadata on the result volume; no scientific file changes."""
    root = Path(root).resolve()
    parent = root / 'io_failure_review'
    if not parent.resolve().is_relative_to(root):
        raise ValueError('Preflight review directory escapes its result package')
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.atomic-preflight-', dir=parent, ignore_cleanup_errors=True) as directory:
        probe = Path(directory).resolve()
        if not probe.is_relative_to(parent.resolve()):
            raise ValueError('Preflight probe escaped its own review directory')
        payload = dict(preflight='atomic_json_only', value=1, labels=['metadata', 'unicode \u06a9'])
        expected = probe / 'expected.json'
        expected.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        writer = AtomicJSONWriter(probe, fallback)
        actual = probe / 'probe.json'
        writer(actual, dict(preflight='old'))
        writer(actual, payload)
        if actual.read_bytes() != expected.read_bytes():
            raise ValueError('Atomic compatibility writer changed JSON serialization')
        if any(probe.glob('*.tmp')):
            raise ValueError('Successful atomic probe left temporary payloads')
        return dict(status='passed', probe_writes=writer.writes, retries=len(writer.retries),
                    scientific_files_touched=False, exact_serialization=True)
