"""ECG validation alignment pilot. Only --check-only and --verify-only are agent-safe."""
import argparse

from project_paths import local_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_ecg.preflight import prepare, SPEC


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only', action='store_true')
    flags.add_argument('--verify-only', action='store_true')
    parser.add_argument('--readiness-report', help='Optional preflight metadata only, outside the pilot output root')
    args = parser.parse_args(argv)
    if args.readiness_report and not args.check_only:
        parser.error('--readiness-report requires --check-only')
    if args.verify_only:
        import yaml
        from scripts.adaptation_ecg.runtime import verify
        spec = yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
        result = verify(experiment_directory(args.output_root or spec['output_root']))
        print('Verified ECG pilot outputs:', len(result['outputs']))
        return 0
    spec, root, manifest, _data = prepare(args)
    if args.check_only:
        if args.readiness_report:
            from scripts.adaptation_ecg.data import require
            path = local_path(args.readiness_report)
            require(not path.is_relative_to(root) and path.suffix == '.json' and not path.exists() and
                    path.is_relative_to(local_path('results/experiments')), 'Readiness report requires a new JSON path outside the pilot package')
            path.parent.mkdir(parents=True, exist_ok=True)
            pilot.write_json(path, dict(status='preflight_passed', scientific_execution=False,
                tests_run=False, model_forward=False, metadata=manifest))
        print('ECG preflight passed. No inference, features, tests, fitting or resampling.')
        return 0
    from scripts.adaptation_ecg.runtime import execute
    execute(spec, root, manifest, _data)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
