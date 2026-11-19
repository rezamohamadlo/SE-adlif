"""One-PC SSC queue. Agents may run --check-only/--verify-only; users launch all jobs."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_ssc_suite.common import (
    require, load_spec, guard, snapshot_source, validate_source, verify_completion, seal,
)


def verify(root, spec):
    complete, manifest = verify_completion(root, root)
    require(complete['scope'] == 'ssc_validation_mechanism_suite' and
            complete['completed_pairs'] == [p['id'] for p in spec['pairs']], 'Suite is not complete for all four pairs')
    require(complete['completed_jobs'] == manifest['jobs'], 'Suite completion job list differs')
    contracts = pilot.read_json(root / 'CONTRACT_TESTS.json')
    require(contracts['status'] == 'passed' and contracts['returncode'] == 0, 'No passing contract-test evidence')
    validate_source(spec, root / 'shared/amt_mt_full_validation')
    for pair in spec['pairs']:
        for step in ('evaluation', 'statistics_v1'):
            verify_completion(root / 'pairs' / pair['id'] / step, root)
    print(f"Verified complete SSC suite: {len(complete['outputs'])} registered outputs; four evaluations and four statistics jobs.", flush=True)
    return complete, manifest


def preflight(args):
    from scripts.adaptation_ssc_suite.preflight import prepare
    from scripts.adaptation_ssc_suite.evaluations import pair_manifest, pair_directory
    from scripts.adaptation_ssc_suite.statistics import load_inputs
    from scripts.adaptation_ssc_suite.common import snapshot_directory
    # Path-only preflight includes Windows TEMP, which may use an 8.3 spelling.
    # Nothing is created or copied under this nonexisting probe directory.
    probe = snapshot_directory(Path(tempfile.gettempdir()) / 'ssc_snapshot_path_preflight')
    require(portable_path(probe) == portable_path(probe.resolve()), 'Snapshot destination is not canonical')
    print('Snapshot path preflight passed, including the resolved Windows temporary-directory spelling; no copies.', flush=True)
    prepared = prepare(args)
    for pair in prepared['spec']['pairs']:
        guard(pair_directory(prepared, pair), pair_manifest(prepared, pair))
    pair = prepared['spec']['pairs'][0]
    inputs = load_inputs(prepared, pair, pair_directory(prepared, pair), pair_manifest(prepared, pair),
                         prepared['source_directory'], prepared['source_manifest'], verify=False)
    print(f"Statistics design preflight: {len(inputs['ids'])} SSC records, {len(inputs['keep'])} complete cases, "
          f"{len(inputs['speaker_ids'])} speaker groups; {len(inputs['excluded'])} explicit exclusions. No regression or resampling.", flush=True)
    print('All 18 checkpoint/data/config prerequisites passed. Remaining pair outcomes are produced and checked at their queue steps.', flush=True)
    return prepared


def consolidate(prepared):
    import numpy as np
    root, spec = prepared['root'], prepared['spec']
    outputs, rows = [root / 'CONTRACT_TESTS.json', root / 'contract_tests.log'], []
    shared = root / 'shared/amt_mt_full_validation'
    _, _, mapping = validate_source(spec, shared)
    outputs += [local_path(value) for value in mapping.values()]
    outputs += [shared / 'COMPLETE.json', shared / 'PATH_INDEX.json']
    reference_weights, reference_exclusions = None, None
    for pair in spec['pairs']:
        entry = dict(comparison=pair)
        for step in ('evaluation', 'statistics_v1'):
            directory = root / 'pairs' / pair['id'] / step
            complete, _ = verify_completion(directory, root)
            entry[step] = dict(directory=portable_path(directory), completion_sha256=pilot.sha256(directory / 'COMPLETE.json'),
                               elapsed_seconds=complete['elapsed_seconds'])
            outputs += [directory / 'COMPLETE.json']
            outputs += [local_path(value) for value in complete['outputs']]
        eval_dir = root / 'pairs' / pair['id'] / 'evaluation'
        if pair['id'] == 'amt_mt':
            clean_path = shared / 'analyses/clean/SUMMARY.json'
        else:
            clean_path = eval_dir / 'analyses/clean/SUMMARY.json'
        clean = pilot.read_json(clean_path)
        group = next(g for g in clean['across_seed_aggregates'] if g['condition'] == 'clean')
        entry['clean_saved_aggregate'] = group
        entry['statistics_summary'] = pilot.read_json(root / 'pairs' / pair['id'] / 'statistics_v1/SUMMARY.json')
        statistics_root = root / 'pairs' / pair['id'] / 'statistics_v1'
        with np.load(statistics_root / 'diagnostic_rows/bootstrap_estimates.npz', allow_pickle=False) as arrays:
            weights = arrays['speaker_weights']
        exclusions = pilot.read_json(statistics_root / 'EXCLUSIONS.json')
        if reference_weights is None:
            reference_weights, reference_exclusions = weights, exclusions
        else:
            require(np.array_equal(weights, reference_weights) and exclusions == reference_exclusions,
                    'Shared speaker resamples or feature-exclusion cohorts differ across model pairs')
        rows.append(entry)
    summary = dict(version=spec['version'], dataset='SSC official validation', samples=9981, training_seeds=spec['seeds'],
        pairs=rows, interpretation='Four comparisons share inputs and some parent predictions; do not pool them as independent replications.',
        independent_confirmation=False, official_test_opened=False, training=False,
        remaining_scope=['ECG', 'gradient/diagnostic-input stage 7', 'direct SSC MT-TC window-stability evidence', 'final synthesis'])
    pilot.write_json(root / 'SUMMARY.json', summary)
    lines = ['# SSC validation mechanism suite', '',
        'All four evaluation/statistics jobs finished and passed their recorded integrity checks. '
        'These are exploratory validation results conditional on three trained seeds, not independent test confirmation. '
        'Shared MT predictions are the same observations across comparisons.', '',
        '| Pair | Parent clean accuracy (%) | Variant clean accuracy (%) | Variant - parent (pp) | Statistics |',
        '|---|---:|---:|---:|---|']
    for entry in rows:
        pair = entry['comparison']
        metrics = entry['clean_saved_aggregate']['metrics']
        def cell(name, scale=1):
            value = metrics[name]
            return f"{scale*value['mean']:.4f} +/- {scale*value['sample_sd']:.4f}"
        lines.append(f"| {pair['variant']} / {pair['parent']} | {cell('parent_accuracy', 100)} | "
                     f"{cell('variant_accuracy', 100)} | {cell('accuracy_difference_pp')} | "
                     f"[effects and limitations](pairs/{pair['id']}/statistics_v1/SUMMARY.md) |")
    lines += ['', 'Values above are copied from each saved clean evaluation summary: mean and sample SD across three matching '
              'training seeds. Accuracy SD is in percentage points. No numeric paper baseline is compared here.', '',
              'Statistical jobs retain exclusions, speaker mappings, shared resampling weights and per-seed effects. '
              'Approximate intervals account for eight feature/pair and twelve intervention/pair effects in separate families. '
              'Read each direction, seed consistency and limitations; do not convert conditional intervals to causal or independent-confirmation claims.', '',
              'The portable package includes the exact AMT/MT source snapshot and its PATH_INDEX.json. Historical manifests '
              'are not rewritten. Transfer the entire suite directory under the same repository-relative layout.', '',
              'ECG, stage 7 gradients, direct SSC output-window stability and final scientific synthesis are not completed by this queue.', '']
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    return outputs + [root / 'SUMMARY.json', root / 'SUMMARY.md']


def run_contract_tests(root):
    """Only the user-launched execution path invokes these selected tests."""
    modules = ['tests.test_adaptation_ssc', 'tests.test_adaptation_ssc_full', 'tests.test_adaptation_ssc_suite']
    command = [sys.executable, '-m', 'unittest', *modules]
    started = time.monotonic()
    print('Running selected SSC input, integrity and queue contract tests; no training or real neural inference.', flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = result.stdout + result.stderr
    (root / 'contract_tests.log').write_text(log, encoding='utf-8')
    matches = re.search(r'Ran (\d+) tests?', log)
    pilot.write_json(root / 'CONTRACT_TESTS.json', dict(status='passed' if result.returncode == 0 else 'failed',
        returncode=result.returncode, command=[portable_path(command[0]), *command[1:]],
        test_count=int(matches.group(1)) if matches else None, elapsed_seconds=time.monotonic()-started,
        log=portable_path(root / 'contract_tests.log')))
    if result.returncode:
        print(log, file=sys.stderr, flush=True)
        raise ValueError('SSC contract tests failed; inspect contract_tests.log. No experiment job was started.')
    print('Selected SSC contract tests passed; evidence saved in CONTRACT_TESTS.json.', flush=True)


def execute(prepared):
    from scripts.adaptation_ssc_suite.evaluations import evaluate
    from scripts.adaptation_ssc_suite.statistics import execute as statistics
    root, spec, manifest = prepared['root'], prepared['spec'], prepared['manifest']
    with pilot.study_lock(root / '.analysis.lock'):
        guard(root, manifest)
        if (root / 'COMPLETE.json').exists():
            verify(root, spec)
            return
        pilot.write_json(root / 'MANIFEST.json', manifest)
        started = time.monotonic()
        progress = dict(status='running', completed_jobs=[], current_job='snapshot_source',
                        training=False, official_test_opened=False)

        def update(job):
            progress.update(current_job=job, elapsed_seconds=time.monotonic()-started)
            pilot.write_json(root / 'PROGRESS.json', progress)
            print('\nSSC queue: ' + job, flush=True)

        try:
            update('contract_tests')
            run_contract_tests(root)
            progress['completed_jobs'].append('contract_tests')
            update('snapshot_source')
            source = snapshot_source(spec, root)
            source_manifest, _, _ = validate_source(spec, source)
            progress['completed_jobs'].append('snapshot_source')
            for pair in spec['pairs']:
                if pair['id'] != 'amt_mt':
                    update(pair['id'] + '_evaluation')
                directory, evaluation_manifest = evaluate(prepared, pair, source)
                if pair['id'] != 'amt_mt':
                    progress['completed_jobs'].append(pair['id'] + '_evaluation')
                update(pair['id'] + '_statistics')
                statistics(prepared, pair, directory, evaluation_manifest, source, source_manifest)
                progress['completed_jobs'].append(pair['id'] + '_statistics')
            update('summary')
            outputs = consolidate(prepared)
            progress['completed_jobs'].append('summary')
            require(progress['completed_jobs'] == manifest['jobs'], 'Queue did not complete the frozen job list')
            progress.update(status='complete', current_job=None, elapsed_seconds=time.monotonic()-started)
            pilot.write_json(root / 'PROGRESS.json', progress)
            seal(root, manifest, outputs, scope='ssc_validation_mechanism_suite',
                 elapsed_seconds=time.monotonic()-started, completed_pairs=[p['id'] for p in spec['pairs']],
                 completed_jobs=progress['completed_jobs'])
            verify(root, spec)
            print('SSC queue finished: ' + portable_path(root), flush=True)
        except BaseException as error:
            progress.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started)
            pilot.write_json(root / 'PROGRESS.json', progress)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.verify_only:
        from scripts.adaptation_analysis.records import experiment_directory
        spec, _ = load_spec()
        verify(experiment_directory(args.output_root or spec['output_root']), spec)
        return 0
    prepared = preflight(args)
    if args.check_only:
        print('SSC suite preflight passed. No model forward, feature extraction, fit, bootstrap, tests, snapshot copies or result writes.')
        return 0
    execute(prepared)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as error:
        print('Error: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
