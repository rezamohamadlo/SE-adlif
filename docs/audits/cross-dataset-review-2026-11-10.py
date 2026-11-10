"""Aggregate existing ordinary-inference evidence; never train or run a model."""
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import local_path, portable_path

OUTPUT = Path(__file__).with_suffix('')
SEEDS = [42, 123, 456]
DATASETS = ['SHD', 'SSC', 'ECG']
SOURCES = {
    'Reference': 'results/valid/reference',
    'DTH': 'results/valid/DTH_SE_adLIF',
    'MT': 'results/valid/MT_SE_adLIF',
    'AMT': 'results/not_improved/AMT_SE_adLIF',
    'MT-TC': 'results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF',
    'LT-DTH': 'results/not_improved/LT_DTH_SE_adLIF',
    'SA-MT': 'results/experiments/sa_mt_v1/SA_MT_SE_adLIF',
    'LG-DTH': 'results/not_improved/LG_DTH_SE_adLIF',
    'DA': 'results/not_improved/DA_SE_adLIF',
    'FP': 'results/not_improved/FP_SE_adLIF',
    'MR': 'results/not_improved/MR_SE_adLIF/K_2',
    'MT-DTH': 'results/not_improved/MT_DTH_SE_adLIF',
    'VS-MT': 'results/experiments/vs_mt_v1/VS_MT_SE_adLIF',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def moments(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, sample_sd=None)
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None)


def main():
    runs, groups = [], []
    for model, source in SOURCES.items():
        datasets = DATASETS if model in list(SOURCES)[:7] else (['SSC'] if model=='VS-MT' else ['SHD'])
        for dataset in datasets:
            selected = []
            for seed in ([42] if model=='VS-MT' else SEEDS):
                directory = local_path(source)/dataset/f'seed_{seed}'
                cost = read(directory/'COST_RESULTS.json')
                score = cost['measurements']['reproduced_test_acc']
                if model=='Reference':
                    accepted = read(directory/'REFERENCE_RESULT.json')['test_acc']
                    assert abs(accepted-score)<1e-6
                    score=accepted
                files = list(directory.glob('logs/mlp_snn/version_*/metrics.csv'))
                if model=='VS-MT' and not files:
                    item=dict(model=model,dataset=dataset,seed=seed,test_percent=score*100,
                              best_validation_percent=None,train_selected_percent=None,train_final_percent=None,
                              selected_epoch=cost['best_epoch'],completed_validation_epochs=None,
                              cost_record=portable_path(directory/'COST_RESULTS.json'),metrics=None,
                              checkpoint_sha256=cost['checkpoint_sha256'],reference_budget_matched=True,
                              evidence_status='Saved cost reproduction only; local raw training CSV/checkpoint absent.')
                    runs.append(item);selected.append(item)
                    continue
                assert len(files)==1,(directory,files)
                with files[0].open(newline='',encoding='utf-8-sig') as stream:
                    rows=list(csv.DictReader(stream))
                validation=[r for r in rows if r.get('val_acc')]
                epochs=[int(float(r['epoch'])) for r in validation]
                assert len(set(epochs))==len(epochs),(model,dataset,seed)
                best=max(validation,key=lambda r:float(r['val_acc']))
                epoch=int(float(best['epoch']))
                assert epoch==cost['best_epoch'],(directory,epoch,cost['best_epoch'])
                tests={float(r['test_acc']) for r in rows if r.get('test_acc')}
                if tests:
                    assert all(abs(v-score)<1e-6 for v in tests),(directory,tests,score)
                training={int(float(r['epoch'])):float(r['train_acc_epoch']) for r in rows if r.get('train_acc_epoch')}
                item=dict(model=model,dataset=dataset,seed=seed,test_percent=score*100,
                          best_validation_percent=float(best['val_acc'])*100,
                          train_selected_percent=training[epoch]*100,
                          train_final_percent=training[max(training)]*100,
                          selected_epoch=epoch,completed_validation_epochs=len(validation),
                          cost_record=portable_path(directory/'COST_RESULTS.json'),
                          metrics=portable_path(files[0]),checkpoint_sha256=cost['checkpoint_sha256'],
                          reference_budget_matched=dataset!='ECG')
                runs.append(item);selected.append(item)
            groups.append(dict(model=model,dataset=dataset,
                               test=moments([r['test_percent'] for r in selected]),
                               validation=moments([r['best_validation_percent'] for r in selected]),
                               train_selected=moments([r['train_selected_percent'] for r in selected]),
                               train_final=moments([r['train_final_percent'] for r in selected])))
    references={g['dataset']:g for g in groups if g['model']=='Reference'}
    for group in groups:
        if group['model']=='Reference':
            continue
        paired=[]
        for r in runs:
            if r['model']==group['model'] and r['dataset']==group['dataset']:
                reference=next(v for v in runs if v['model']=='Reference' and v['dataset']==r['dataset'] and v['seed']==r['seed'])
                paired.append(r['test_percent']-reference['test_percent'])
        group['paired_difference_from_reference_pp']=moments(paired)
        group['paired_differences_pp']=paired
    extension = []
    extension_roots={('DTH','SHD'):'test_remaining_v1/dth_shd',
                     ('MT','SHD'):'test_mt_shd_v1',
                     ('MT','ECG'):'pc1_ecg/MT_SE_adLIF/ECG',
                     ('SA-MT','ECG'):'test_remaining_v1/sa_mt_ecg'}
    for (model,dataset),suffix in extension_roots.items():
        original=[r['test_percent'] for r in runs if r['model']==model and r['dataset']==dataset]
        added=[]
        for seed in [1009,2027,3037,4049,5051,6067,7079]:
            path=ROOT/'results/experiments/extended_seeds_v1'/suffix/f'seed_{seed}'/'TEST_RESULTS.json'
            record=read(path)
            added.append(dict(seed=seed,test_percent=record['metrics']['test_acc']*100,path=portable_path(path)))
        extension.append(dict(model=model,dataset=dataset,original_three=moments(original),
                              added_seven=moments([r['test_percent'] for r in added]),
                              combined_ten=moments(original+[r['test_percent'] for r in added]),records=added,
                              note='Reference has only three seeds; no ten-seed paired reference claim.'))
    selection_path=ROOT/'results/experiments/ssc_validation_search_v1/selection/select-lr.json'
    selection=read(selection_path)
    validation_records=list((ROOT/'results/experiments/ssc_validation_search_v1').rglob('VALIDATION_RESULTS.json'))
    study=dict(frozen_lr_selection=portable_path(selection_path),ranking=selection['ranking'],
               validation_completion_records=len(validation_records),
               completed_final_confirmation=False,
               note='Imported summaries without local metrics/checkpoints are not locally resumable.')
    OUTPUT.mkdir(exist_ok=True)
    summary=dict(metric='ordinary test accuracy percent; differences in percentage points',
                 scope='13 reference/model definitions; original-seed evidence only where accepted',
                 runs=runs,groups=groups,extensions=extension,ssc_study=study,
                 caveats=['ECG historical reference stopping/budget is unmatched.',
                          'Historical summed-loss SSC is separate from current mean-loss.',
                          'Test-informed development remains exploratory.',
                          'No training or model inference performed.'])
    (OUTPUT/'EVIDENCE.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    names=['DTH','MT','AMT','MT-TC','LT-DTH','SA-MT']
    matrix=np.array([[next(g for g in groups if g['model']==model and g['dataset']==dataset)
                       ['paired_difference_from_reference_pp']['mean'] for dataset in DATASETS] for model in names])
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.05,1.2]})
    image=axes[0].imshow(matrix,cmap='RdBu',vmin=-1.5,vmax=1.5,aspect='auto')
    axes[0].set_xticks(range(3),['SHD*','SSC','ECG (unmatched)']);axes[0].set_yticks(range(len(names)),names)
    axes[0].set_title('Difference from local reference (pp)\nOriginal three matching training seeds')
    for i in range(matrix.shape[0]):
        for j in range(3):axes[0].text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>1 else 'black')
    fig.colorbar(image,ax=axes[0],shrink=.7,label='Percentage points')
    for index,entry in enumerate(extension):
        color=['#2563a6','#167d5a','#b56916','#9a3885'][index]
        for key,x in [('original_three',index-.12),('combined_ten',index+.12)]:
            stats=entry[key]
            axes[1].errorbar(x,stats['mean'],yerr=stats['sample_sd'],fmt='o' if key=='original_three' else 's',
                             color=color,capsize=4,label=('3 seeds' if key=='original_three' else '10 seeds') if index==0 else None)
    axes[1].set_xticks(range(4),[f"{r['model']}\n{'SHD*' if r['dataset']=='SHD' else 'ECG'}" for r in extension])
    axes[1].set_title('Initial results versus extended cohorts\nMeans +/- sample SD; overlapping cohorts')
    axes[1].set_ylabel('Official test accuracy (%)');axes[1].grid(axis='y',alpha=.2);axes[1].legend()
    fig.suptitle('Cross-dataset evidence: gains concentrate on SHD*',fontsize=14)
    fig.tight_layout(rect=(0,.03,1,.94))
    fig.savefig(OUTPUT/'cross-dataset-evidence.png',dpi=180)
    svg=OUTPUT/'cross-dataset-evidence.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    print(f'Checked {sum(bool(r["metrics"]) for r in runs)} original-seed cost/metric pairs and one saved VS-MT cost record; saved evidence and figure.')


if __name__=='__main__':
    main()
