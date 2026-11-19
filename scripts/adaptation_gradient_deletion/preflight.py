"""Read-only source, checkpoint and clean-input checks; never creates controls."""
from argparse import Namespace
import hashlib

import yaml

from datasets.utils.pad_tensors import PadTensors
from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_analysis.records import experiment_directory
from scripts.adaptation_batch_precision.contracts import batch_identity
from scripts.adaptation_gradient_deletion.contracts import load_primary, select_primary
from scripts.adaptation_gradient_pilot.methods import count_hash, require
from scripts.adaptation_gradient_pilot.preflight import guard
from scripts.adaptation_gradient_strict.preflight import prepare as source_prepare
from scripts.adaptation_gradient_strict.runtime import verify as verify_source

SPEC='config/studies/adaptation_gradient_deletion_v1.yaml'
TEST='tests/test_adaptation_gradient_deletion.py'


def load_spec():
    spec=yaml.safe_load(local_path(SPEC,must_exist=True).read_text(encoding='utf-8'))
    expected=dict(inherits='config/paper_baseline.yaml',dataset_section='SHD',report_label='SHD*',precision_mode='highest_matmul',
        models=['MT_SE_adLIF','AMT_SE_adLIF'],seeds=[42],samples=40,primary_layout='singleton_natural',primary_repeat=0,
        deletion_fraction=.01,random_replicates=3,random_seed=20261116,score_atol=2e-5,score_rtol=1e-5,
        training=False,backward=False,official_test_access=False,input_optimization=False,stage7_complete=False)
    require(all(spec[k]==v for k,v in expected.items()),'Versioned strict matched-deletion scope differs')
    return spec


def prepare(args):
    spec=load_spec();source=experiment_directory(spec['source_root'])
    require(pilot.sha256(source/'MANIFEST.json')==spec['source_manifest_sha256'] and
            pilot.sha256(source/'COMPLETE.json')==spec['source_complete_sha256'],'Verified strict gradient source changed')
    verify_source(source)
    summary=pilot.read_json(source/'SUMMARY.json')
    require(summary['stability_criteria_passed'] and summary['required_stability_checks']==86,
            'Matched deletion requires the registered strict stability criteria to pass')
    ready=source_prepare(Namespace(device=args.device,output_root=spec['source_root']))
    old,data=ready['manifest'],ready['data'];root=experiment_directory(args.output_root or spec['output_root'])
    for other in (source,ready['reference_root'],ready['legacy_root']):
        require(root!=other and not root.is_relative_to(other) and not other.is_relative_to(root),
                'Deletion output must be separate from all frozen studies')
    primary=select_primary(pilot.read_json(source/'GRADIENT_INDEX.json'),spec['models'],old['pilot_ids'])
    for (model,sid),item in primary.items():
        path=local_path(item['path'],must_exist=True);require(path.is_relative_to(source),'Primary array escapes source package')
        value,_g=load_primary(path,item,pilot.digest(old),old['backend'])
        item_data=data.load(sid)
        require(count_hash(item_data['counts'])==value['input_sha256'] and value['model_state_sha256']==
                next(r['model_state_sha256'] for r in old['runs'] if r['model']==model) and
                batch_identity(PadTensors()([data.model_item(sid,item_data['counts'])]))==old['input_identities'][str(sid)],
                'Primary input/state/target/mask differs')
    hashes=dict(old['source_sha256'])
    for path in list((ROOT/'scripts/adaptation_gradient_deletion').glob('*.py'))+[local_path(SPEC),local_path(TEST)]:
        hashes[portable_path(path)]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    for name,h in hashes.items():
        require(hashlib.sha256(local_path(name,must_exist=True).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,
                'Frozen dependency differs: '+name)
    manifest=dict(version=spec['version'],specification=spec,output_root=portable_path(root),
        source_manifest_sha256=pilot.digest(old),source_complete_sha256=spec['source_complete_sha256'],
        source_sha256=hashes,source_hash_encoding='CRLF_normalized_to_LF',environment=old['environment'],backend=old['backend'],
        source_backend=old['source_backend'],runs=old['runs'],pilot_ids=old['pilot_ids'],data=old['data'],
        input_identities=old['input_identities'],primary_gradients=[primary[m,i] for m in spec['models'] for i in old['pilot_ids']],
        paper_baseline_version=old['paper_baseline_version'],paper_baseline_dataset='SHD',report_label='SHD*',
        expected_clean_evaluations=80,maximum_shared_deletion_inputs=160,maximum_model_evaluations=400,
        maximum_paired_condition_records=200,training=False,backward=False,official_test_opened=False,
        input_optimization=False,stage7_complete=False,
        limitations=['one pair, one seed and forty validation records; source full-validation anchors do not expand this pilot',
                    'MT guidance uses model/loss sensitivity and is not symmetric model selection or causal attribution',
                    'surrogate linear predictions need not match finite hard-spike CE changes; no direction-agreement gate',
                    'three random replicates average within record, not independent seeds or population inference',
                    'record 3896 retained; original source_high evidence remains unchanged'])
    guard(root,manifest)
    print('Strict deletion preflight passed: 80 authenticated natural repeat-0 arrays, 2 checkpoints, 40 fixed records.',flush=True)
    print('Planned: 80 clean gates, up to 160 shared deletion inputs and 400 model evaluations. No new backward.',flush=True)
    print('No tests, model forward/backward, control generation or analysis executed.',flush=True)
    return dict(root=root,spec=spec,manifest=manifest,data=data,source_root=source,
                protected_roots=[source,ready['reference_root'],ready['legacy_root']])
