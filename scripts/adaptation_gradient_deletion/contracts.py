"""Authenticated primary arrays, exact matched controls and atomic report output."""
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.methods import (
    array_hash, event_budget, exact_random_delta, guided_delta, publish, require, validate_delta,
)

GUIDED = 'mt_abs_gradient_deletion'
RANDOM = 'exact_random_deletion'


def select_primary(index, models, ids):
    selected=[r for r in index if r['layout']=='singleton_natural' and r['repeat']==0]
    keys=[(r['model'],r['sample_id']) for r in selected]
    require(len(keys)==len(set(keys))==80 and set(keys)=={(m,i) for m in models for i in ids} and
            all(r['seed']==42 for r in selected),'Primary gradients must be exactly the fixed 80 natural repeat-0 arrays')
    return {(r['model'],r['sample_id']):r for r in selected}


def load_primary(path, item, source_manifest_sha256, backend):
    path=Path(path)
    require(pilot.sha256(path)==item['sha256'],'Primary array file hash differs')
    with np.load(path,allow_pickle=False) as z:
        identity=json.loads(z['identity_json'].item());value=json.loads(z['summary_json'].item());g=z['gradient'].copy()
    require(identity['manifest_sha256']==source_manifest_sha256 and identity['backend']==backend and
            identity['model']==item['model'] and identity['sample_id']==item['sample_id'] and identity['seed']==42 and
            identity['layout']=='singleton_natural' and identity['repeat']==0 and
            value['model']==item['model'] and value['sample_id']==item['sample_id'] and value['repeat']==0 and
            value['seed']==42 and value['layout']=='singleton_natural' and
            g.dtype==np.float32 and g.shape==(value['stored_steps'],140) and np.isfinite(g).all() and
            array_hash(g)==value['gradient_sha256'] and value['autograd_forward_check']['passed'] and
            value['strict_singleton_reference_check']['passed'], 'Primary gradient identity/shape/stability differs')
    return value,g


def plan_controls(counts, parent_gradient, spec):
    budget=event_budget(counts,spec['deletion_fraction'])
    require(parent_gradient.shape==counts.shape and np.isfinite(parent_gradient).all(),'Parent physical gradient shape differs')
    eligible=budget>0 and bool(np.any(parent_gradient[counts>0]!=0))
    conditions=([(GUIDED,0)] if eligible else [])+[(RANDOM,i) for i in range(spec['random_replicates'])]
    return budget,eligible,conditions


def control_seed(spec, sid, condition, replicate):
    return None if condition==GUIDED else int(pilot.digest([spec['version'],spec['random_seed'],sid,condition,replicate])[:16],16)


def build_delta(counts, gradient, budget, condition, seed):
    require(condition in (GUIDED,RANDOM),'Unknown deletion condition')
    delta=guided_delta(counts,gradient,budget) if condition==GUIDED else exact_random_delta(counts,budget,seed)
    require(delta is not None,'Ineligible guidance cannot be fabricated')
    validate_delta(counts,delta,budget)
    return delta


def linear_prediction(gradient,delta):
    require(gradient.ndim==delta.ndim==2 and gradient.shape[1]==delta.shape[1] and len(gradient)>=len(delta) and
            np.isfinite(gradient).all(),'Linear prediction needs the model own aligned gradient')
    return float(np.sum(gradient[:len(delta)].astype(np.float64)*delta))


def require_clean_checks(checks):
    require(len(checks)==80 and len({(r['model'],r['sample_id']) for r in checks})==80 and all(r['passed'] for r in checks),
            'Fresh clean reference failed; deletion blocked. Preserve evidence and inspect CLEAN_REFERENCE_CHECKS.json')


def atomic_csv(path, rows):
    path=Path(path);keys=list(dict.fromkeys(k for r in rows for k in r))
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.tmp',dir=path.parent)
    with os.fdopen(descriptor,'w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    publish(path,name)
