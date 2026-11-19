"""User entrypoint; only --check-only and --verify-only are agent-safe."""
import argparse

from project_paths import local_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.methods import require
from scripts.adaptation_gradient_strict.preflight import load_spec, prepare


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group()
    flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    parser.add_argument('--readiness-report');args=parser.parse_args(argv)
    if args.readiness_report and not args.check_only:parser.error('--readiness-report requires --check-only')
    if args.verify_only:
        from scripts.adaptation_gradient_strict.runtime import verify
        print('Verified strict pilot outputs:',len(verify(args.output_root or load_spec()['output_root'])['outputs']));return 0
    prepared=prepare(args)
    if args.check_only:
        if args.readiness_report:
            path=local_path(args.readiness_report)
            require(path.suffix=='.json' and not path.exists() and path.is_relative_to(local_path('results/experiments')) and
                    all(not path.is_relative_to(prepared[k]) for k in ('root','reference_root','legacy_root')),
                    'Choose a new readiness JSON outside all experiment roots')
            path.parent.mkdir(parents=True,exist_ok=True)
            pilot.write_json(path,dict(status='preflight_passed',scientific_execution=False,tests_run=False,
                model_forward=False,model_backward=False,stability_verified=False,metadata=prepared['manifest']))
        return 0
    from scripts.adaptation_gradient_strict.runtime import execute
    execute(prepared);return 0


if __name__=='__main__':raise SystemExit(main())
