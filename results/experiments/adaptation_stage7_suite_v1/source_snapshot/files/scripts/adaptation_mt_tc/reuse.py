"""Strict reuse of saved parent predictions. Never computes model predictions."""
from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import verify_package
from scripts.adaptation_analysis.evaluate import check_anchor


def validate_reuse(study,requested):
    root=local_path(study['reuse_source'],must_exist=True)
    verify_package(root)
    saved=pilot.read_json(root/'MANIFEST.json')
    if pilot.digest(saved)!=study['reuse_manifest_sha256'] or pilot.sha256(root/'COMPLETE.json')!=study['reuse_completion_sha256']:
        raise ValueError('Frozen reuse source identity differs')
    for key in ('sample_ids','samples','seeds','data','environment','ordinary_inference','training','official_test_opened','condition_definitions'):
        if saved[key]!=requested[key]:raise ValueError('Reuse numerical/input identity differs: '+key)
    if saved['resolved_study']['features']!=study['features']:
        raise ValueError('Reuse feature definitions differ')
    for path,expected in saved['source_sha256'].items():
        if requested['source_sha256'].get(path)!=expected:
            raise ValueError('Source implementation differs from reuse evaluator: '+path)
    parent_runs=[r for r in requested['runs'] if r['model']==study['parent']]
    if len(parent_runs)!=len(study['seeds']):raise ValueError('Incomplete parent cohort')
    for run in parent_runs:
        source=next(r for r in saved['runs'] if r['model']==run['model'] and r['seed']==run['seed'])
        for key in ('checkpoint','checkpoint_sha256','selected_epoch','validation'):
            if source[key]!=run[key]:raise ValueError('Reuse parent identity differs: '+key)
        if pilot.effective_config(source['config'])!=pilot.effective_config(run['config']):
            raise ValueError('Reuse parent effective configuration differs')
    chunks={}
    for name,rep,_,_ in requested['condition_definitions']:
        names=[f'input_features_{name}_rep_{rep}.json']
        names += [f"{study['parent']}_seed_{seed}_{name}_rep_{rep}.json" for seed in study['seeds']]
        for filename in names:
            path=root/'diagnostic_rows'/filename;cache=pilot.read_json(path)
            identity=dict(manifest_sha256=pilot.digest(saved),condition=name,replicate=rep)
            if filename.startswith('input_features_'):identity['kind']='model_independent_input_features'
            else:
                run=next(r for r in parent_runs if filename.startswith(f"{r['model']}_seed_{r['seed']}_"))
                identity['checkpoint_sha256']=run['checkpoint_sha256']
            if cache['identity']!=identity or cache['value_sha256']!=pilot.digest(cache['value']):
                raise ValueError('Reused chunk identity/hash differs')
            if len(cache['value'])!=requested['samples'] or {r['sample_id'] for r in cache['value']}!=set(requested['sample_ids']):
                raise ValueError('Reused chunk full-validation coverage differs')
            chunks[filename]=dict(source_path=portable_path(path),source_sha256=pilot.sha256(path),
                                  original_identity=identity,value_sha256=cache['value_sha256'])
    return dict(source=study['reuse_source'],source_manifest_sha256=study['reuse_manifest_sha256'],
                source_completion_sha256=study['reuse_completion_sha256'],chunks=chunks,
                interpretation='Derived cache identities are new; all reused values retain their original value hashes. Reuse is not an independent training/evaluation replicate.')


def import_reuse(destination,manifest):
    raw=destination/'diagnostic_rows';raw.mkdir(exist_ok=True)
    for filename,record in manifest['reuse']['chunks'].items():
        source=local_path(record['source_path'],must_exist=True)
        if pilot.sha256(source)!=record['source_sha256']:raise ValueError('Reuse source changed after preflight')
        cache=pilot.read_json(source)
        if cache['value_sha256']!=record['value_sha256'] or pilot.digest(cache['value'])!=record['value_sha256']:
            raise ValueError('Reuse payload changed')
        identity=dict(record['original_identity'],manifest_sha256=pilot.digest(manifest))
        derived=dict(identity=identity,value=cache['value'],value_sha256=cache['value_sha256'])
        target=raw/filename
        if target.exists():pilot.validate_manifest(pilot.read_json(target),derived)
        else:pilot.write_json(target,derived)
    path=destination/'REUSE_PROVENANCE.json'
    if path.exists():pilot.validate_manifest(pilot.read_json(path),manifest['reuse'])
    else:pilot.write_json(path,manifest['reuse'])
    return path


def reuse_parent(run,study,manifest,raw,anchors,outputs):
    for name,rep,_,_ in manifest['condition_definitions']:
        path=raw/f"{run['model']}_seed_{run['seed']}_{name}_rep_{rep}.json"
        cache=pilot.read_json(path)
        identity=dict(manifest_sha256=pilot.digest(manifest),checkpoint_sha256=run['checkpoint_sha256'],condition=name,replicate=rep)
        if cache['identity']!=identity or cache['value_sha256']!=pilot.digest(cache['value']):
            raise ValueError('Imported parent cache differs')
        if len(cache['value'])!=manifest['samples'] or {r['sample_id'] for r in cache['value']}!=set(manifest['sample_ids']):
            raise ValueError('Imported parent coverage differs')
        if name=='clean':anchors.append(check_anchor(cache['value'],run,study))
        outputs.append(path)
    print(f"Reused frozen {run['model']} seed {run['seed']} for all conditions; no parent inference.",flush=True)
