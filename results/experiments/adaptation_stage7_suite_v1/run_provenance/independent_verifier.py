"""Independent saved-artifact audit; never constructs a neural model or differentiates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_stage7.data import Adapter


def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode('ascii')+a.tobytes()).hexdigest()


def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=1e-13)


def ledger(root):
    manifest=pilot.read_json(root/'MANIFEST.json');complete=pilot.read_json(root/'COMPLETE.json')
    assert complete['status']=='complete' and complete['manifest_sha256']==pilot.digest(manifest)
    assert complete['outputs'][portable_path(root/'MANIFEST.json')]==pilot.sha256(root/'MANIFEST.json')
    for name,expected in complete['outputs'].items():
        target=local_path(name,must_exist=True);assert target.is_relative_to(root)
        assert pilot.sha256(target)==expected,name
    return manifest,complete


def array_cache(path):
    header=pilot.read_json(path.with_suffix('.json'))
    assert header['artifact_sha256']==pilot.sha256(path)
    with np.load(path,allow_pickle=False) as z:
        identity=json.loads(z['identity_json'].item());value=json.loads(z['summary_json'].item())
        arrays={k:z[k].copy() for k in z.files if k not in ('identity_json','summary_json')}
    assert identity==header['identity'] and value==header['value'] and pilot.digest(value)==header['value_sha256']
    return identity,value,arrays


def native_check(original,candidate,domain,budget):
    a,b=original.astype(np.float64),candidate.astype(np.float64)
    assert a.shape==b.shape and np.isfinite(b).all()
    if domain=='counts':
        assert np.all(b>=0) and np.equal(b,np.rint(b)).all()
        assert np.array_equal(a.sum(0),b.sum(0)) and np.all(b<=a.max(0,keepdims=True))
        assert np.all(b[a!=0]<=a[a!=0])
    else:
        assert np.isin(b,[-1,0,1]).all()
        assert np.all((b[a!=0]==0)|(b[a!=0]==a[a!=0]))
        for sign in [-1,1]:assert np.array_equal((a==sign).sum(0),(b==sign).sum(0))
    moved=np.abs(b-a).sum()/2
    assert moved==int(moved) and moved<=budget
    return int(moved)


def audit_job(root,a,spec):
    m,c=ledger(root);run=m['run'];summary=pilot.read_json(root/'SUMMARY.json')
    gradient_rows=pilot.read_json(root/'GRADIENT_SUMMARIES.json');seen=set();gradients={};grad_records={}
    for row in gradient_rows:
        sid,rep=row['sample_id'],row['repeat'];seen.add((sid,rep))
        identity,value,payload=array_cache(root/'gradients'/f'sample_{sid}_rep_{rep}.npz')
        x=a.input(sid);g=payload['gradient'];physical=np.abs(g[:len(x)].astype(np.float64));mass=float(physical.sum())
        assert identity['manifest_sha256']==pilot.digest(m) and identity['input_sha256']==digest_array(x)
        assert value==row and digest_array(g)==row['gradient_sha256'] and g.dtype==np.float32 and np.isfinite(g).all()
        assert close(mass,row['physical_abs_gradient_sum']) and close(np.linalg.norm(g[:len(x)].astype(np.float64)),row['physical_gradient_l2'])
        assert close(physical[x!=0].sum(),row['event_cell_abs_gradient_sum'])
        assert close(physical[x==0].sum(),row['empty_cell_abs_gradient_sum'])
        assert close(physical[:a.burn].sum(),row['burn_in_input_abs_gradient_sum'])
        assert close(np.abs(g[len(x):].astype(np.float64)).sum(),row['synthetic_padding_abs_gradient_sum'])
        if mass:
            positions=np.minimum(19,np.arange(len(x))*20//len(x))
            bins=np.bincount(positions,weights=physical.sum(1),minlength=20)/mass
            np.testing.assert_allclose(bins,row['relative_time_abs_mass'],rtol=1e-11,atol=1e-13)
        else:assert row['relative_time_abs_mass'] is None and row['channel_abs_mass'] is None
        forward=row['autograd_forward_check']
        assert forward['prediction_matches']==(row['clean']['predictions']==row['autograd_forward']['predictions'])
        assert close(forward['loss_difference'],row['autograd_forward']['loss']-row['clean']['loss'])
        assert forward['passed']==(forward['prediction_matches'] and abs(forward['loss_difference'])<=forward['loss_bound'])
        gradients[sid,rep]=g;grad_records[sid,rep]=row
    expected={(sid,0) for sid in a.selected}|{(sid,1) for sid in a.repeat_ids};assert seen==expected
    repeat_rows=pilot.read_json(root/'REPEAT_CHECKS.json')
    for row in repeat_rows:
        sid=row['sample_id'];first,second=[gradients[sid,r][:len(a.input(sid))].astype(np.float64).reshape(-1) for r in [0,1]]
        norms=[np.linalg.norm(first),np.linalg.norm(second)];distance=np.linalg.norm(first-second)
        relative=distance/max(norms) if max(norms) else 0
        assert close(relative,row['symmetric_relative_l2'])
        if min(norms):assert close(np.dot(first,second)/(norms[0]*norms[1]),row['cosine'])
        else:assert row['cosine'] is None
    candidates=0;accepted=0;optim_gradients=0
    for diagnostic in pilot.read_json(root/'DIAGNOSTICS.json'):
        sid=diagnostic['sample_id'];original=a.input(sid);current=original.copy();current_score=grad_records[sid,0]['clean']
        for step in diagnostic['trace']:
            identity,value,payload=array_cache(local_path(step['candidate_path'],must_exist=True))
            assert identity['current_sha256']==digest_array(current) and identity['current_score_sha256']==pilot.digest(current_score)
            native_check(original,payload['input'],a.domain,diagnostic['budget'])
            assert np.isfinite(payload['gradient']).all();optim_gradients+=1
            assert step['candidate_sha256']==digest_array(payload['input']) and value['status']==step['status']
            if value['accepted']:
                assert value['score']['loss']>current_score['loss']+value['acceptance_bound']
                current=payload['input'];current_score=value['score']
            candidates+=1
        _,value,payload=array_cache(root/'optimized'/f'sample_{sid}.npz')
        assert value==diagnostic and np.array_equal(payload['input'],current)
        moved=native_check(original,current,a.domain,diagnostic['budget'])
        assert moved==diagnostic['moves'] and current_score==diagnostic['optimized']
        accepted+=int(moved>0)
        allocation={(r['channel'],r['sign']):r['moves'] for r in diagnostic['allocation']}
        for rep in range(3):
            _,control,arrays=array_cache(root/'controls'/f'sample_{sid}_rep_{rep}.npz')
            assert native_check(original,arrays['input'],a.domain,diagnostic['budget'])==moved
            for (channel,sign),n in allocation.items():
                removed=(original[:,channel]-arrays['input'][:,channel])*sign
                assert int(removed[original[:,channel]!=0].sum())==n if a.domain=='counts' else int(np.sum((original[:,channel]==sign)&(arrays['input'][:,channel]==0)))==n
    frozen=pilot.read_json(root/'FROZEN_STATE.json')
    assert frozen['status']=='unchanged' and frozen['model_state_sha256']==run['model_state_sha256'] and frozen['all_parameter_gradients_none']
    assert summary['accepted_diagnostic_inputs']==accepted and summary['gradient_records']==len(gradient_rows)
    return dict(dataset=run['dataset'],model=run['model'],seed=run['seed'],gradients=len(gradient_rows),repeat_checks=len(repeat_rows),
        optimization_attempts=4,accepted_inputs=accepted,candidate_arrays=candidates,optimization_gradient_arrays=optim_gradients,
        status='verified',registered_outputs=len(c['outputs']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',default='results/experiments/adaptation_stage7_suite_v1')
    parser.add_argument('--completed-only',action='store_true');parser.add_argument('--emit-report')
    args=parser.parse_args(argv);root=local_path(args.output_root,must_exist=True);m=pilot.read_json(root/'MANIFEST.json');spec=m['specification']
    baseline=yaml.safe_load(local_path(spec['inherits']).read_text())['paper_baseline'];adapters={}
    for ds in spec['datasets']:
        entry=spec['sources'][ds][0];source=pilot.read_json(local_path(entry['root'])/'MANIFEST.json')
        run=next(r for r in m['runs'] if r['dataset']==ds)
        adapters[ds]=Adapter(ds,source,run['config'],baseline,spec)
    audited=[]
    for run in m['runs']:
        directory=root/'jobs'/run['dataset']/run['model']/f"seed_{run['seed']}"
        if not (directory/'COMPLETE.json').exists():
            assert args.completed_only,'Missing checkpoint job: '+str(directory);continue
        audited.append(audit_job(directory,adapters[run['dataset']],spec))
    if not args.completed_only:
        _,complete=ledger(root);assert complete['stage7_scoped_complete']
        assert len(audited)==54
        audit=pilot.read_json(root/'STAGE7_CLOSEOUT_AUDIT.json');assert audit['stage7_scoped_complete']
        ledger(root/'source_snapshot')
    report=dict(status='completed_jobs_verified' if args.completed_only else 'stage7_registered_scope_independently_verified',
        verifier_source_sha256=pilot.sha256(Path(__file__)),jobs=audited,count=len(audited),
        no_model_forward_or_backward=True,stage7_scoped_complete=not args.completed_only)
    if args.emit_report:pilot.write_json(local_path(args.emit_report),report)
    print('Independently verified checkpoint jobs:',len(audited))
    return 0


if __name__=='__main__':raise SystemExit(main())
