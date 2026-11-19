"""LT-DTH/DTH adjusted statistics. Agent may run --check-only, not the analysis."""
import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import yaml
from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory,verify_completion
from scripts.adaptation_statistics import analyze as original
from scripts.adaptation_statistics.methods import cluster_analysis

SPEC='config/studies/adaptation_lt_dth_statistics_v1.yaml'


def load_spec():
    definition=yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    if definition['inherits']!=original.SPEC:raise ValueError('Unsupported inherited statistical method')
    base=yaml.safe_load(local_path(definition['inherits']).read_text(encoding='utf-8'))
    spec=dict(base,**definition)
    if (spec['parent'],spec['variant'])!=('DTH_SE_adLIF','LT_DTH_SE_adLIF'):
        raise ValueError('Unexpected comparison identity')
    if spec['trace_analysis']!='descriptive_model_layer_level_only_no_per_sample_parameter_regression':
        raise ValueError('Unsupported parameter-level analysis')
    return spec


def load_traces(spec,manifest):
    if (manifest['resolved_study']['parent'],manifest['resolved_study']['variant'])!=(spec['parent'],spec['variant']):
        raise ValueError('Source model pair differs')
    records=[]
    for run in manifest['runs']:
        path=local_path(spec['source'])/'parameters'/f"{run['model']}_seed_{run['seed']}.json"
        record=pilot.read_json(path)
        if any(record[k]!=run[k] for k in ('model','seed','checkpoint','checkpoint_sha256')):
            raise ValueError('Trace record checkpoint identity differs')
        if len(record['layers'])!=2 or {x['layer'] for x in record['layers']}!={'l1','l2'}:
            raise ValueError('Expected two layer-level trace descriptors')
        for layer in record['layers']:
            if layer['learned']!=(run['model']==spec['variant']) or layer['bin_ms']!=manifest['data']['bin_ms']:
                raise ValueError('Learned/fixed trace status or physical bin differs')
            if not 0<layer['fast']['rho']<layer['slow']['rho']<1:raise ValueError('Trace ordering differs')
            for kind in ('fast','slow'):
                r=layer[kind]
                if not math.isclose(r['e_folding_frames'],-1/math.log(r['rho']),rel_tol=1e-10):
                    raise ValueError('Invalid frame timescale descriptor')
                if not math.isclose(r['e_folding_ms'],layer['bin_ms']*r['e_folding_frames'],rel_tol=1e-10):
                    raise ValueError('Invalid physical timescale descriptor')
        records.append(record)
    return records


def trace_summary(records,spec):
    rows=[]
    for model in (spec['parent'],spec['variant']):
        for layer in ('l1','l2'):
            for kind in ('fast','slow'):
                entries=[dict(seed=r['seed'],**next(x for x in r['layers'] if x['layer']==layer)[kind])
                         for r in records if r['model']==model]
                if sorted(x['seed'] for x in entries)!=spec['training_seeds']:
                    raise ValueError('Trace training-seed cohort differs')
                rows.append(dict(model=model,layer=layer,trace=kind,n_models=len(entries),per_seed=entries,
                    mean_ms=statistics.mean(x['e_folding_ms'] for x in entries),
                    sample_sd_ms=statistics.stdev(x['e_folding_ms'] for x in entries)))
    return dict(unit='model/layer, not sample',groups=rows,
        limitation='Three seeds and two layers do not identify a population-level parameter/performance relationship. No parameter-data correlation, p-value or per-sample parameter regression is fitted.')


def make_manifest(spec,inputs,records):
    manifest=original.make_manifest(spec,inputs)
    manifest.update(comparison=dict(parent=spec['parent'],variant=spec['variant']),trace_records=records)
    for path in list((ROOT/'scripts/adaptation_lt_statistics').glob('*.py'))+[local_path(SPEC)]:
        manifest['source_sha256'][portable_path(path)]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return manifest


def write_outputs(destination,summary,result,spec,inputs,records):
    summary.update(comparison=dict(parent=spec['parent'],variant=spec['variant']),
                   hypotheses=dict(fast_slow=spec['fast_slow_hypothesis'],autocorrelation=spec['autocorrelation_hypothesis'],mechanism=spec['mechanism_hypothesis']))
    paths=original.write_outputs(destination,summary,result,spec,inputs)
    path=destination/'SUMMARY.md'
    text=path.read_text(encoding='utf-8').replace('loss(MT) - loss(AMT), positive favors AMT','loss(DTH) - loss(LT-DTH), positive favors LT-DTH')
    old='The fast/slow hypothesis predicts a positive adjusted coefficient. Inspect both seed consistency and leave-one-speaker-out ranges; a conditional interval direction is not evidence of population-wide training-seed robustness. Autocorrelation is exploratory, with no directional AMT mechanism specified. Negative or inconclusive outcomes are retained.'
    if old not in text:raise ValueError('Inherited report contract changed; inspect variant-specific hypotheses')
    text=text.replace(old,'The LT-DTH hypothesis concerns dependence on input temporal scale, without assuming a monotonic positive direction. Autocorrelation crossing lag is the input-timescale proxy; fast/slow contrast is an exploratory joint covariate. Both coefficient intervals form the two-feature family. A conditional association does not prove causality, learned-parameter alignment or robustness over training seeds. Negative/inconclusive outcomes are retained.')
    traces=trace_summary(records,spec)
    pilot.write_json(destination/'TRACE_SUMMARY.json',traces)
    text+='\n\n## Layer-level trace descriptors\n\nThese are model/layer properties, not independent sample-level parameters. Milliseconds use the saved input bin width, not neuron dt. No parameter/performance correlation is fitted.\n\n| Model | Layer | Trace | Models (seeds) | E-folding ms, mean +/- sample SD |\n|---|---|---|---:|---:|\n'
    for row in traces['groups']:
        text+=f"| {row['model']} | {row['layer']} | {row['trace']} | {row['n_models']} | {row['mean_ms']:.3f} +/- {row['sample_sd_ms']:.3f} |\n"
    path.write_text(text,encoding='utf-8')
    return paths+[destination/'TRACE_SUMMARY.json']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv);spec=load_spec()
    destination=experiment_directory(args.output_root or spec['output_root'])
    if destination==experiment_directory(spec['source']):raise ValueError('Statistics cannot overwrite source evaluation')
    if args.verify_only:
        print('Verified statistics outputs:',len(verify_completion(destination,filename='COMPLETE.json')));return 0
    inputs=original.load_inputs(spec)
    records=load_traces(spec,inputs['source_manifest'])
    manifest=make_manifest(spec,inputs,records);path=destination/'MANIFEST.json'
    if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest)
    elif destination.exists() and any(p.name!='.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty output directory without matching manifest')
    if args.check_only:
        print(f"LT-DTH statistics preflight passed: {len(inputs['ids'])} samples, {len(inputs['keep'])} complete cases, {len(np.unique(inputs['all_speakers']))} speakers, {len(records)} model trace records. No adjusted fits, bootstrap, output writes, inference or training.")
        return 0
    with pilot.study_lock(destination/'.analysis.lock'):
        if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest)
        else:pilot.write_json(path,manifest)
        if (destination/'COMPLETE.json').exists():
            verify_completion(destination,filename='COMPLETE.json');print('Already complete and verified.');return 0
        started=time.monotonic()
        result=cluster_analysis(inputs['design'],inputs['centered_outcome'],inputs['speakers'],inputs['effects'],inputs['all_speakers'],spec['bootstrap'])
        summary=original.summarize(result,spec,inputs)
        outputs=[path]+write_outputs(destination,summary,result,spec,inputs,records)
        pilot.write_json(destination/'COMPLETE.json',dict(status='complete',manifest_sha256=pilot.digest(manifest),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            outputs={portable_path(p):pilot.sha256(p) for p in outputs}))
        print('LT-DTH statistics completed:',portable_path(destination))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)
