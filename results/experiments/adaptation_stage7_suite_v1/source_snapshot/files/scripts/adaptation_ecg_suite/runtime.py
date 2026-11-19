"""User-launched four-pair queue with shared caches and hard clean-anchor gating."""
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import time

import torch

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.data import require
from scripts.adaptation_ecg.preflight import guard
from scripts.adaptation_ecg.runtime import check_anchor, infer, validate_rows
from scripts.adaptation_ecg_suite.preflight import failure_source
from scripts.adaptation_ecg_suite.reporting import write_pair
from scripts.ssc_resume_io import AtomicJSONWriter


def prediction_keys(spec):
    return {(model, seed, condition, rep) for pair in spec['pairs'] for model in (pair['parent'], pair['variant'])
            for seed in spec['seeds'] for condition, rep in spec['conditions']}


def require_all_anchors(anchors, expected=18):
    require(len(anchors) == expected and len({(r['model'], r['seed']) for r in anchors}) == expected and
            all(r['status'] == 'passed' for r in anchors),
            'ECG clean-anchor gate failed; deletion is blocked. Inspect CLEAN_ANCHOR_REVIEW.json. '
            'Do not relax tolerances or reuse failed v1 predictions.')


def snapshot_failure(root, spec, write):
    source = failure_source(root, spec)
    destination = root/'reviews/failed_alignment_pilot_v1'
    destination.mkdir(parents=True, exist_ok=True)
    paths, index = [], {}
    for suffix, checksum in spec['failed_pilot']['files'].items():
        target = (destination/suffix).resolve()
        require(target.is_relative_to(destination.resolve()), 'Failure-review path escapes its package')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copyfile(source/suffix, target)
        require(pilot.sha256(target) == checksum, 'Failure-review snapshot changed')
        paths.append(target)
        index[spec['failed_pilot']['root']+'/'+suffix] = portable_path(target)
    record = dict(path_index=index, source_sha256=spec['failed_pilot']['files'],
        scientific_reuse=False, meaning='Failed anchor evidence, preserved byte-for-byte; never used as current predictions')
    path = destination/'PATH_INDEX.json'
    if path.exists(): pilot.validate_manifest(pilot.read_json(path), record)
    else: write(path, record)
    return paths+[path]


def verify(directory):
    root = experiment_directory(directory)
    manifest, complete = [pilot.read_json(root/name) for name in ('MANIFEST.json', 'COMPLETE.json')]
    require(complete['status'] == 'complete' and complete['manifest_sha256'] == pilot.digest(manifest) and
            complete['completed_pair_evaluations'] == [p['id'] for p in manifest['specification']['pairs']] and
            not complete['full_mechanism_study_complete'], 'ECG completion identity/scope differs')
    require(complete['outputs'].get(portable_path(root/'MANIFEST.json')) == pilot.sha256(root/'MANIFEST.json'),
            'ECG completion does not register its own manifest')
    for name, checksum in complete['outputs'].items():
        path = local_path(name, must_exist=True)
        require(path.is_relative_to(root) and pilot.sha256(path) == checksum, 'ECG output changed: '+name)
    require_all_anchors(pilot.read_json(root/'CLEAN_ANCHOR_REVIEW.json'))
    tests = pilot.read_json(root/'CONTRACT_TESTS.json')
    require(tests['status'] == 'passed' and tests['returncode'] == 0, 'No passing ECG contract-test evidence')
    summary = pilot.read_json(root/'SUMMARY.json')
    require(summary['prediction_chunks'] == 72 and summary['paired_rows'] == 1440 and
            len(summary['pairs']) == 4, 'Incomplete ECG evaluation coverage')
    return complete


def execute(prepared):
    root, spec, manifest, data = (prepared[k] for k in ('root', 'spec', 'manifest', 'data'))
    started = time.monotonic()
    with pilot.study_lock(root/'.analysis.lock'):
        guard(root, manifest)
        if (root/'COMPLETE.json').exists():
            print('Already complete; verified ECG outputs:', len(verify(root)['outputs']))
            return
        write = AtomicJSONWriter(root, pilot.write_json)
        if not (root/'MANIFEST.json').exists(): write(root/'MANIFEST.json', manifest)
        outputs = [root/'MANIFEST.json']
        identity, cache, index = pilot.digest(manifest), {}, []

        def progress(phase, **details):
            write(root/'PROGRESS.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details))

        def predictions(run, condition, replicate):
            require(torch.get_float32_matmul_precision() == 'high', 'Precision contract drifted before ECG inference')
            key = (run['model'], run['seed'], condition, replicate)
            if key in cache: return cache[key]
            path = root/'predictions'/run['model']/f"seed_{run['seed']}"/condition/f'rep_{replicate}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            contract = dict(manifest_sha256=identity, checkpoint_sha256=run['checkpoint_sha256'],
                condition=condition, replicate=replicate, float32_matmul_precision='high')
            if path.exists():
                record = pilot.read_json(path)
                pilot.validate_manifest(record['identity'], contract)
                require(pilot.digest(record['value']) == record['value_sha256'], 'ECG cached predictions damaged')
                rows = record['value']
            else:
                rows = infer(run, data, spec, condition, replicate, torch.device(manifest['environment']['device']))
                write(path, dict(identity=contract, value=rows, value_sha256=pilot.digest(rows)))
            validate_rows(rows, data, spec, run, condition, replicate)
            cache[key] = rows
            outputs.append(path)
            index.append(dict(model=run['model'], seed=run['seed'], condition=condition, replicate=replicate,
                              path=portable_path(path), sha256=pilot.sha256(path)))
            return rows

        try:
            progress('contract_tests')
            tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation_ecg',
                                    'tests.test_adaptation_ecg_suite'], cwd=ROOT, capture_output=True,
                                    text=True, encoding='utf-8', errors='replace')
            (root/'CONTRACT_TESTS.txt').write_text(tests.stdout+tests.stderr, encoding='utf-8')
            write(root/'CONTRACT_TESTS.json', dict(status='passed' if tests.returncode == 0 else 'failed',
                returncode=tests.returncode, modules=['tests.test_adaptation_ecg', 'tests.test_adaptation_ecg_suite'],
                log_sha256=pilot.sha256(root/'CONTRACT_TESTS.txt')))
            outputs += [root/'CONTRACT_TESTS.txt', root/'CONTRACT_TESTS.json']
            require(tests.returncode == 0, 'ECG contract tests failed; inspect CONTRACT_TESTS.txt')
            outputs += snapshot_failure(root, spec, write)
            anchors = []
            for run in manifest['runs']:
                progress('clean_anchors', completed=len(anchors), total=18, model=run['model'], seed=run['seed'])
                print(f"ECG clean anchor ({len(anchors)+1}/18): {run['model']} seed {run['seed']}", flush=True)
                rows = predictions(run, 'clean', 0)
                try:
                    anchors.append(check_anchor(rows, run, spec))
                except ValueError as error:
                    from scripts.adaptation_ecg.runtime import aggregate
                    measured = aggregate(rows)
                    anchors.append(dict(status='failed', model=run['model'], seed=run['seed'],
                        selected_epoch=run['selected_epoch'], measured=measured, saved_validation=run['validation'],
                        accuracy_delta=measured['step_weighted_accuracy']-run['validation']['accuracy'],
                        loss_delta=measured['step_weighted_loss']-run['validation']['loss'], error=str(error)))
                write(root/'CLEAN_ANCHOR_REVIEW.json', anchors)
            outputs.append(root/'CLEAN_ANCHOR_REVIEW.json')
            require_all_anchors(anchors)
            print('All 18 clean anchors passed. Starting the shared four-pair deletion queue.', flush=True)
            for number, pair in enumerate(spec['pairs'], 1):
                progress('pair_evaluation', pair=pair['id'], pair_number=number, pairs=4)
                for name in (pair['parent'], pair['variant']):
                    for seed in spec['seeds']:
                        run = next(r for r in manifest['runs'] if r['model'] == name and r['seed'] == seed)
                        for condition, rep in spec['conditions']:
                            print(f"ECG pair {number}/4 {pair['id']}: {name} seed {seed} {condition}/{rep}", flush=True)
                            predictions(run, condition, rep)
                paths, _ = write_pair(root, pair, spec, cache, write)
                outputs += paths
            require(set(cache) == prediction_keys(spec) and len(cache) == 72, 'ECG prediction coverage differs')
            write(root/'PREDICTION_INDEX.json', index)
            outputs.append(root/'PREDICTION_INDEX.json')
            pair_summaries = [pilot.read_json(root/'pairs'/p['id']/'SUMMARY.json') for p in spec['pairs']]
            summary = dict(status='four_pair_evaluations_complete', pairs=pair_summaries, prediction_chunks=len(cache),
                paired_rows=sum(p['paired_rows'] for p in pair_summaries), validation_records=30, clean_anchors=18,
                full_mechanism_study_complete=False, pending=manifest['pending'], training=False, official_test_opened=False,
                precision='high; all clean anchors must reproduce saved validation before deletion',
                interpretation=manifest['interpretation'])
            write(root/'SUMMARY.json', summary)
            outputs.append(root/'SUMMARY.json')
            lines = ['# Four-pair ECG validation evaluation queue', '',
                'All four pair evaluations finished: 18 clean anchors, 72 unique prediction chunks, 1440 paired rows.',
                'All 30 validation records; seeds 42/123/456; clean plus 10% signed-event deletion (three realizations).',
                'Training precision high, supervised CE, no training or official-test access. Parent predictions are shared.', '',
                '| Pair | Evaluation | Paired rows | Evidence |', '|---|---|---:|---|']
            for p in spec['pairs']:
                lines.append(f"| {p['variant']} / {p['parent']} | complete | 360 | [summary](pairs/{p['id']}/SUMMARY.md) |")
            lines += ['', 'These are descriptive validation evaluations. Full ECG mechanism studies remain incomplete.',
                      'Pending: '+', '.join(manifest['pending'])+'.', '']
            (root/'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
            outputs.append(root/'SUMMARY.md')
            write(root/'COMPLETE.json', dict(status='complete', manifest_sha256=identity,
                completed_utc=datetime.now(timezone.utc).isoformat(), attempt_elapsed_seconds=time.monotonic()-started,
                completed_pair_evaluations=[p['id'] for p in spec['pairs']], full_mechanism_study_complete=False,
                outputs={portable_path(p): pilot.sha256(p) for p in outputs}))
            progress('complete', registered_outputs=len(verify(root)['outputs']))
            print('Four ECG pair evaluations complete:', portable_path(root/'SUMMARY.md'), flush=True)
        except Exception as error:
            progress('failed', error_type=type(error).__name__, error=str(error))
            raise
