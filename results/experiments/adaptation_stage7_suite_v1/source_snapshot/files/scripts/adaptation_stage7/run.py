"""Registered complete stage-7 scope; no training or official-test evaluation."""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import sys
import time
import uuid

import torch

from project_paths import ROOT,local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.methods import require
from scripts.adaptation_stage7.preflight import prepare
from scripts.adaptation_stage7.runtime import Cache,job_root,run_job,run_transfer,verify


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--output-root')
    parser.add_argument('--datasets',nargs='+',choices=['SHD','SSC','ECG'])
    parser.add_argument('--seeds',nargs='+',type=int,choices=[42,123,456])
    parser.add_argument('--models',nargs='+')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true')
    flags.add_argument('--verify-only',action='store_true');flags.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args(argv)
    if args.verify_only:
        root=args.output_root or 'results/experiments/adaptation_stage7_suite_v1';c=verify(root)
        print('Stage-7 scope complete:',c.get('stage7_scoped_complete',False),'verified outputs:',len(c['outputs']));return 0
    prepared=prepare(args);root=prepared['root'];m=prepared['manifest'];spec=prepared['spec']
    if args.models:require(set(args.models)<=set(spec['models']),'Unknown model scheduling assignment')
    if args.check_only:
        print('All stage-7 checkpoints, data, sources and scope passed preflight. No experiment execution.');return 0
    with pilot.study_lock(root/'.suite.lock'):
        if (root/'COMPLETE.json').exists():
            c=verify(root);print('Previously finalized stage-7 scope complete:',c.get('stage7_scoped_complete'));return 0
        cache=Cache(root,m)
        attempt_id=uuid.uuid4().hex;attempt_start=time.monotonic()
        provenance=root/'run_provenance';provenance.mkdir(exist_ok=True)
        attempt=dict(attempt_id=attempt_id,started_utc=datetime.now(timezone.utc).isoformat(),
            environment=pilot.environment(prepared['device']),datasets=args.datasets,seeds=args.seeds,models=args.models,
            authorization='User explicitly authorized agent experiments through stage-7 completion, 2026-11-16',training=False)
        cache.write(provenance/(attempt_id+'.json'),dict(attempt,status='running'))
        tests=subprocess.run([sys.executable,'-m','unittest','tests.test_adaptation_stage7'],cwd=ROOT,
            capture_output=True,text=True,encoding='utf-8',errors='replace')
        (root/'CONTRACT_TESTS.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
        (provenance/(attempt_id+'_tests.txt')).write_text(tests.stdout+tests.stderr,encoding='utf-8')
        cache.write(root/'CONTRACT_TESTS.json',dict(status='passed' if tests.returncode==0 else 'failed',returncode=tests.returncode,
            log_sha256=pilot.sha256(root/'CONTRACT_TESTS.txt')))
        require(tests.returncode==0,'Common stage-7 contract tests failed; inspect test log')
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
        selected=[r for r in m['runs'] if (not args.datasets or r['dataset'] in args.datasets) and
            (not args.seeds or r['seed'] in args.seeds) and (not args.models or r['model'] in args.models)]
        if not args.finalize_only:
            for number,run in enumerate(selected,1):
                cache.progress('checkpoint_jobs',dataset=run['dataset'],model=run['model'],seed=run['seed'],
                    assignment_job=number,assignment_total=len(selected),global_jobs=sum((job_root(root,r)/'COMPLETE.json').exists() for r in m['runs']))
                print(f"Stage-7 job {number}/{len(selected)}: {run['dataset']} {run['model']} seed {run['seed']}",flush=True)
                run_job(prepared,run)
        for ds in spec['datasets']:
            for seed in spec['seeds']:
                if (not args.datasets or ds in args.datasets) and (not args.seeds or seed in args.seeds):
                    cache.progress('diagnostic_transfers',dataset=ds,seed=seed);run_transfer(prepared,ds,seed)
        cache.progress('finalizing')
        cache.write(provenance/(attempt_id+'.json'),dict(attempt,status='execution_phase_finished',
            elapsed_seconds_before_reporting=time.monotonic()-attempt_start,completed_utc=datetime.now(timezone.utc).isoformat()))
        from scripts.adaptation_stage7.reporting import finalize
        complete=finalize(prepared)
        cache.progress('complete' if complete else 'partial_or_numerical_review_required',stage7_scoped_complete=complete,
            global_jobs=sum((job_root(root,r)/'COMPLETE.json').exists() for r in m['runs']))
        print('Stage-7 registered scope complete:',complete,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
