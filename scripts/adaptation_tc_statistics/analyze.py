"""MT-TC/MT statistics; agents may run only --check-only, not analysis or tests."""
import argparse
import hashlib
from pathlib import Path
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

SPEC='config/studies/adaptation_mt_tc_statistics_v1.yaml'


def load_spec():
    definition=yaml.safe_load(local_path(SPEC).read_text(encoding='utf-8'))
    if definition['inherits']!=original.SPEC:raise ValueError('Unsupported inherited method')
    base=yaml.safe_load(local_path(definition['inherits']).read_text(encoding='utf-8'))
    spec=dict(base,**definition)
    if (spec['parent'],spec['variant'])!=('MT_SE_adLIF','MT_TC_SE_adLIF'):
        raise ValueError('Unexpected MT-TC comparison identity')
    if spec['parameter_analysis']!='none_no_direct_window_evidence_or_mediation_claim':
        raise ValueError('Parameter analysis is not supported by this specification')
    return spec


def validate_pair(spec,inputs):
    manifest=inputs['source_manifest']
    if (manifest['resolved_study']['parent'],manifest['resolved_study']['variant'])!=(spec['parent'],spec['variant']):
        raise ValueError('Source model pair differs from specification')
    if manifest['comparison_loss'] != spec['comparison_loss']:
        raise ValueError('Comparison loss must exclude auxiliary JS')
    # The source completion has already authenticated the provenance file.
    provenance=pilot.read_json(local_path(spec['source'])/'REUSE_PROVENANCE.json')
    if provenance!=manifest['reuse']:raise ValueError('Reuse provenance differs')
    if len(provenance['chunks'])!=24:raise ValueError('Expected 18 MT and 6 shared feature chunks')
    for filename,record in provenance['chunks'].items():
        if not (filename.startswith(spec['parent']+'_seed_') or filename.startswith('input_features_')):
            raise ValueError('Unexpected reused record')
        cache=pilot.read_json(local_path(spec['source'])/'diagnostic_rows'/filename)
        if cache['value_sha256']!=record['value_sha256'] or pilot.digest(cache['value'])!=record['value_sha256']:
            raise ValueError('Reused MT/input values changed')
        expected=dict(record['original_identity'],manifest_sha256=pilot.digest(manifest))
        if cache['identity']!=expected:raise ValueError('Derived reuse identity differs')


def make_manifest(spec,inputs):
    manifest=original.make_manifest(spec,inputs)
    manifest.update(comparison=dict(parent=spec['parent'],variant=spec['variant']),
                    reuse_interpretation=spec['reuse_interpretation'])
    for path in list((ROOT/'scripts/adaptation_tc_statistics').glob('*.py'))+[local_path(SPEC)]:
        manifest['source_sha256'][portable_path(path)]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest['comparison_loss']=spec['comparison_loss']
    manifest['analysis_scope']='input_feature_associations_only_window_diagnostic_separate'
    return manifest


def write_outputs(destination,summary,result,spec,inputs):
    summary['comparison_loss']=spec['comparison_loss']
    summary.update(comparison=dict(parent=spec['parent'],variant=spec['variant']),
        hypotheses=dict(fast_slow=spec['fast_slow_hypothesis'],autocorrelation=spec['autocorrelation_hypothesis'],mechanism=spec['mechanism_hypothesis']),
        reuse_interpretation=spec['reuse_interpretation'])
    summary['limitations']+=['Input contrast and ACF do not directly measure temporal stability of model class evidence; direct window diagnostics are separate.',
                            'Reused MT predictions are the original parent observations, not new independent replications.']
    paths=original.write_outputs(destination,summary,result,spec,inputs)
    path=destination/'SUMMARY.md'
    text=path.read_text(encoding='utf-8').replace('loss(MT) - loss(AMT), positive favors AMT','loss(MT) - loss(MT-TC), positive favors MT-TC')
    old='The fast/slow hypothesis predicts a positive adjusted coefficient. Inspect both seed consistency and leave-one-speaker-out ranges; a conditional interval direction is not evidence of population-wide training-seed robustness. Autocorrelation is exploratory, with no directional AMT mechanism specified. Negative or inconclusive outcomes are retained.'
    if old not in text:raise ValueError('Inherited report contract changed; inspect MT-TC hypotheses')
    text=text.replace(old,'Both input-feature associations are exploratory and two-sided for MT-TC. Neither input contrast nor ACF directly measures stability of adjacent output-window class evidence. The window diagnostic is a separate frozen-model evaluation. Inspect seed and speaker sensitivity; conditional directions do not establish causality or robustness across training seeds. The auxiliary JS penalty is excluded from every loss-benefit calculation. Reused MT records are not independent replications.')
    path.write_text(text,encoding='utf-8')
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--check-only',action='store_true');flags.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv);spec=load_spec()
    destination=experiment_directory(args.output_root or spec['output_root'])
    if destination==experiment_directory(spec['source']):raise ValueError('Statistics cannot overwrite evaluation outputs')
    if args.verify_only:
        print('Verified MT-TC statistics outputs:',len(verify_completion(destination,filename='COMPLETE.json')));return 0
    inputs=original.load_inputs(spec)
    validate_pair(spec,inputs)
    manifest=make_manifest(spec,inputs);path=destination/'MANIFEST.json'
    if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest)
    elif destination.exists() and any(p.name!='.analysis.lock' for p in destination.iterdir()):
        raise ValueError('Nonempty destination without matching manifest')
    if args.check_only:
        print(f"MT-TC statistics preflight passed: {len(inputs['ids'])} samples, {len(inputs['keep'])} complete cases, {len(np.unique(inputs['all_speakers']))} speakers; 24 reused chunk identities checked. No fits, bootstrap, tests, inference or output writes.")
        return 0
    with pilot.study_lock(destination/'.analysis.lock'):
        if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest)
        else:pilot.write_json(path,manifest)
        if (destination/'COMPLETE.json').exists():
            verify_completion(destination,filename='COMPLETE.json');print('Already complete and verified.');return 0
        started=time.monotonic()
        result=cluster_analysis(inputs['design'],inputs['centered_outcome'],inputs['speakers'],inputs['effects'],inputs['all_speakers'],spec['bootstrap'])
        summary=original.summarize(result,spec,inputs)
        outputs=[path]+write_outputs(destination,summary,result,spec,inputs)
        pilot.write_json(destination/'COMPLETE.json',dict(status='complete',manifest_sha256=pilot.digest(manifest),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            outputs={portable_path(p):pilot.sha256(p) for p in outputs}))
        print('MT-TC statistics completed:',portable_path(destination))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr,flush=True);raise SystemExit(1)
