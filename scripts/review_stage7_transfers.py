"""Independent arithmetic/identity audit of sealed shared-input comparisons."""
import argparse
import csv
import math

import numpy as np

from project_paths import local_path
from scripts.adaptation_analysis import pilot
from scripts.review_stage7_suite import array_cache,digest_array,ledger,native_check


def equal(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12)


def cached(path):
    row=pilot.read_json(path);assert pilot.digest(row['value'])==row['value_sha256'];return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',default='results/experiments/adaptation_stage7_suite_v1')
    args=parser.parse_args(argv);root=local_path(args.output_root,must_exist=True);m,complete=ledger(root)
    assert complete['stage7_scoped_complete'];spec=m['specification']
    originals={};truth={};input_cache={};diagnostic_cache={};all_rows={};score_count=0
    cohort,_=ledger(root/'cohort_snapshot')
    for record in cohort['samples']:
        with np.load(local_path(record['path']),allow_pickle=False) as z:
            originals[record['dataset'],record['sample_id']]=z['input'].copy()
            y=z['targets'];truth[record['dataset'],record['sample_id']]=y[y!=-1].tolist()
    for ds in spec['datasets']:
        for seed in spec['seeds']:
            transfer=root/'transfers'/ds/f'seed_{seed}';tm,_=ledger(transfer)
            rows=pilot.read_json(transfer/'TRANSFER_ROWS.json')
            assert len(rows)==576
            for row in rows:
                target,origin,sid,condition,rep=[row[k] for k in ['target','origin','sample_id','condition','replicate']]
                key=(ds,seed,target,origin,sid,condition,rep);assert key not in all_rows;all_rows[key]=row
                clean=cached(transfer/'clean'/f'{target}_{sid}.json')['value']
                source=root/'jobs'/ds/origin/f'seed_{seed}'
                if (ds,seed,origin) not in diagnostic_cache:
                    diagnostic_cache[ds,seed,origin]={r['sample_id']:r for r in pilot.read_json(source/'DIAGNOSTICS.json')}
                diagnostic=diagnostic_cache[ds,seed,origin][sid]
                input_key=ds,seed,origin,sid,condition,rep
                if input_key not in input_cache:
                    path=source/('optimized' if condition=='optimized' else 'controls')/(f'sample_{sid}.npz' if condition=='optimized' else f'sample_{sid}_rep_{rep}.npz')
                    _,_,arrays=array_cache(path);x=arrays['input']
                    moves=native_check(originals[ds,sid],x,'signed' if ds=='ECG' else 'counts',diagnostic['budget'])
                    assert moves==row['moves']==diagnostic['moves']
                    input_cache[input_key]=digest_array(x),pilot.sha256(path)
                score=cached(transfer/'scores'/target/f'{origin}_{sid}_{condition}_{rep}.json')
                identity,value=score['identity'],score['value']
                assert identity['manifest_sha256']==pilot.digest(tm)
                assert identity['input_sha256']==row['input_sha256']==input_cache[input_key][0]
                assert identity['source_artifact_sha256']==input_cache[input_key][1]
                assert math.isfinite(value['loss']) and value['loss']>=0 and value['valid_steps']==len(truth[ds,sid])
                assert value['correct_steps']==sum(a==b for a,b in zip(value['predictions'],truth[ds,sid]))
                assert equal(value['accuracy'],value['correct_steps']/value['valid_steps'])
                assert equal(row['loss'],value['loss']) and equal(row['clean_loss'],clean['loss'])
                assert equal(row['loss_change'],value['loss']-clean['loss']);score_count+=1
    paired=list(csv.DictReader((root/'PAIRED_DIAGNOSTIC_EFFECTS.csv').open(newline='',encoding='utf-8')))
    assert len(paired)==288
    for row in paired:
        ds,seed,sid,origin=row['dataset'],int(row['seed']),int(row['sample_id']),row['origin_model']
        pair=next(r for r in spec['pairs'] if r['id']==row['pair'])
        parent=all_rows[ds,seed,pair['parent'],origin,sid,'optimized',0]
        variant=all_rows[ds,seed,pair['variant'],origin,sid,'optimized',0]
        assert parent['input_sha256']==variant['input_sha256']
        optimized=parent['loss_change']-variant['loss_change'];random=[]
        for rep in range(3):
            p=all_rows[ds,seed,pair['parent'],origin,sid,'random',rep]
            v=all_rows[ds,seed,pair['variant'],origin,sid,'random',rep]
            assert p['input_sha256']==v['input_sha256'];random.append(p['loss_change']-v['loss_change'])
        average=math.fsum(random)/3
        assert equal(row['optimized_benefit_change'],optimized) and equal(row['mean_random_benefit_change'],average)
        assert equal(row['optimized_minus_random'],optimized-average)
    aggregates=pilot.read_json(root/'SEED_AGGREGATES.json');assert len(aggregates)==24
    for group in aggregates:
        selected=[r for r in paired if r['dataset']==group['dataset'] and r['pair']==group['pair'] and r['origin_role']==group['origin_role']]
        for field in ['optimized_benefit_change','mean_random_benefit_change','optimized_minus_random']:
            values=[math.fsum(float(r[field]) for r in selected if int(r['seed'])==seed)/4 for seed in spec['seeds']]
            mean=math.fsum(values)/3;sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/2)
            assert equal(mean,group[field]['mean']) and equal(sd,group[field]['sample_sd']) and group[field]['n_training_seeds']==3
    report=dict(status='independently_verified',verifier_source_sha256=pilot.sha256(__file__),
        transfer_bundles=9,target_evaluations=score_count,unique_shared_diagnostic_inputs=len(input_cache),
        paired_rows=288,seed_aggregate_groups=24,model_execution=False,training=False)
    pilot.write_json(root/'INDEPENDENT_TRANSFER_VERIFICATION.json',report)
    print(report);return 0


if __name__=='__main__':raise SystemExit(main())
