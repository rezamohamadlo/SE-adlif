"""Synthetic JSON/statistics only; no datasets, models, trainer or GPU."""
import copy
import hashlib
import json
import unittest

from module.final_factorial_summary import summarize_final_factorial, render_final_factorial, ARMS, DATASETS, SEEDS
from module.paired_seed_statistics import paired_statistics


def fixture():
    plan=dict(status='frozen',purpose='final',test_selects_ideas=False,
        architecture_and_settings_frozen=True,arms=list(ARMS),datasets=list(DATASETS),
        training_seeds=list(SEEDS),evaluator_source_identity={'synthetic_fixture':True},checkpoints={})
    plan_sha='a'*64
    evaluator=hashlib.sha256(json.dumps(plan['evaluator_source_identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    scores={'Baseline':[.80,.82,.84],'A':[.81,.84,.87],'B':[.82,.83,.88],'A+B':[.84,.86,.90]}
    results={}
    for dataset in DATASETS:
        for arm in ARMS:
            for index,seed in enumerate(SEEDS):
                path=f'results/experiments/synthetic_final/{arm}/{dataset}/{seed}/best.ckpt'
                sha=hashlib.sha256(path.encode()).hexdigest()
                plan['checkpoints'][path]=dict(arm=arm,sha256=sha)
                results[path]=dict(checkpoint=path,ordinary_inference=True,test_time_augmentation=False,
                    metrics={'test_acc':scores[arm][index]},study_evidence=dict(purpose='final',
                        dataset=dataset,seed=seed,test_selects_ideas=False,verified_configurations=36,
                        preliminary=False,completed_training_epochs={'SHD':300,'SSC':40,'ECG':400}[dataset],
                        full_budget={'SHD':300,'SSC':40,'ECG':400}[dataset],evaluation_plan_sha256=plan_sha,
                        evaluator_identity_sha256=evaluator,checkpoint_sha256=sha,
                        test_data_sha256=hashlib.sha256(dataset.encode()).hexdigest()))
    return plan,plan_sha,results


class FinalFactorialSummaryTests(unittest.TestCase):
    def test_all36_means_sample_sd_and_every_paired_contrast(self):
        plan,sha,records=fixture();before=copy.deepcopy((plan,records))
        summary=summarize_final_factorial(plan,sha,records)
        self.assertEqual(before,(plan,records))
        self.assertEqual(summary['metric'],'ordinary_final_test_accuracy')
        self.assertNotIn('best_model',summary)
        for dataset in DATASETS:
            item=summary['datasets'][dataset]
            self.assertAlmostEqual(item['arms']['Baseline']['mean_test_accuracy_percent'],82)
            self.assertAlmostEqual(item['arms']['Baseline']['sample_sd_percent'],2)
            effect=item['contrasts']['A_minus_Baseline']
            self.assertEqual([p['seed'] for p in effect['per_seed']],list(SEEDS))
            for p,v in zip(effect['per_seed'],[1,2,3]):self.assertAlmostEqual(p['difference_pp'],v)
            self.assertAlmostEqual(effect['mean_pp'],2)
            self.assertAlmostEqual(effect['sample_sd_pp'],1)
            self.assertAlmostEqual(effect['ci95_pp'][1],4.4841377,places=6)
            self.assertAlmostEqual(item['contrasts']['B_minus_Baseline']['mean_pp'],7/3)
            self.assertAlmostEqual(item['contrasts']['AB_minus_A']['mean_pp'],8/3)
            self.assertAlmostEqual(item['contrasts']['AB_minus_B']['mean_pp'],7/3)
            self.assertAlmostEqual(item['contrasts']['interaction']['mean_pp'],1/3)
        self.assertEqual(summary['datasets']['SHD']['dataset_label'],'SHD*')

    def test_paper_delta_uses_unrounded_mean_never_sd_difference(self):
        plan,sha,records=fixture()
        for path,payload in records.items():
            if payload['study_evidence']['dataset']=='SHD' and plan['checkpoints'][path]['arm']=='Baseline':
                payload['metrics']['test_acc']+=.00004999
        paper={'SHD':dict(compatible=True,dataset_label='SHD*',mean_accuracy_percent=93.79,source='synthetic paper fixture')}
        summary=summarize_final_factorial(plan,sha,records,paper)
        stats=summary['datasets']['SHD']['arms']['Baseline']
        self.assertAlmostEqual(stats['delta_paper_pp'],82.004999-93.79)
        self.assertAlmostEqual(stats['sample_sd_percent'],2)
        text=render_final_factorial(summary)
        self.assertIn('Mean test accuracy ± sample SD (%) (Δ paper)',text)
        self.assertIn('82.00 ± 2.00 (-11.79 pp)',text)
        self.assertIn('Seed 42: test accuracy (%) (Δ paper)',text)
        self.assertIn('individual training seed with the published mean',text)
        self.assertNotIn('(Δ paper)',text.split('## SSC')[1])

    def test_missing_or_incompatible_paper_has_no_numeric_delta(self):
        plan,sha,records=fixture()
        result=summarize_final_factorial(plan,sha,records,{'SHD':{'compatible':False,'mean_accuracy_percent':95.81}})
        self.assertIsNone(result['datasets']['SHD']['arms']['A']['delta_paper_pp'])
        self.assertNotIn('(Δ paper)',render_final_factorial(result))
        with self.assertRaises(ValueError):
            summarize_final_factorial(plan,sha,records,{'SHD':dict(compatible=True,dataset_label='SHD',source='fixture',mean_accuracy_percent=95.81)})

    def test_reject_missing_extra_or_duplicate_seed_results(self):
        plan,sha,records=fixture();partial=copy.deepcopy(records);partial.pop(next(iter(partial)))
        with self.assertRaises(ValueError):summarize_final_factorial(plan,sha,partial)
        extra=copy.deepcopy(records);extra['unplanned']={}
        with self.assertRaises(ValueError):summarize_final_factorial(plan,sha,extra)
        duplicate=copy.deepcopy(records);paths=list(duplicate);duplicate[paths[1]]['study_evidence']['seed']=42
        with self.assertRaisesRegex(ValueError,'Repeated'):summarize_final_factorial(plan,sha,duplicate)

    def test_reject_validation_exploratory_tta_partial_or_wrong_plan(self):
        plan,sha,records=fixture();key=next(iter(records))
        for field,value in [('purpose','exploratory'),('preliminary',True),('verified_configurations',12),
                ('completed_training_epochs',30),('evaluation_plan_sha256','b'*64),('evaluator_identity_sha256','c'*64)]:
            bad=copy.deepcopy(records);bad[key]['study_evidence'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize_final_factorial(plan,sha,bad)
        for field,value in [('ordinary_inference',False),('test_time_augmentation',True)]:
            bad=copy.deepcopy(records);bad[key][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize_final_factorial(plan,sha,bad)
        bad=copy.deepcopy(records);bad[key]['metrics']={'val_acc':.99}
        with self.assertRaises(ValueError):summarize_final_factorial(plan,sha,bad)
        bad_plan=copy.deepcopy(plan);bad_plan['purpose']='development'
        with self.assertRaises(ValueError):summarize_final_factorial(bad_plan,sha,records)

    def test_reject_invalid_metrics_and_mixed_test_sources(self):
        plan,sha,records=fixture();keys=list(records)
        for value in [True,float('nan'),float('inf'),-.1,1.1,'0.9']:
            bad=copy.deepcopy(records);bad[keys[0]]['metrics']['test_acc']=value
            with self.subTest(value=value),self.assertRaises(ValueError):summarize_final_factorial(plan,sha,bad)
        bad=copy.deepcopy(records);bad[keys[0]]['study_evidence']['test_data_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'test source'):summarize_final_factorial(plan,sha,bad)
        bad=copy.deepcopy(records);bad[keys[0]]['study_evidence']['checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):summarize_final_factorial(plan,sha,bad)

    def test_retains_regression_and_all_datasets_without_winner_selection(self):
        plan,sha,records=fixture()
        for path,item in records.items():
            if item['study_evidence']['dataset']=='ECG' and plan['checkpoints'][path]['arm']=='A':
                item['metrics']['test_acc']-=.03
        summary=summarize_final_factorial(plan,sha,records)
        self.assertAlmostEqual(summary['datasets']['ECG']['contrasts']['A_minus_Baseline']['mean_pp'],-1)
        text=render_final_factorial(summary)
        for label in ['## SHD*','## SSC','## ECG']:self.assertIn(label,text)
        self.assertIn('No cross-dataset pooling or test-selected winner',text)
        self.assertIn('does not prove synergy',text)

    def test_shared_statistics_preserve_existing_small_n_behavior(self):
        empty=paired_statistics([]);self.assertIsNone(empty['mean_pp']);self.assertIsNone(empty['ci95_pp'])
        single=paired_statistics([2]);self.assertEqual(single['mean_pp'],2);self.assertIsNone(single['sample_sd_pp'])
        self.assertEqual(paired_statistics([1,2,3])['sample_sd_pp'],1)


if __name__=='__main__':unittest.main()
