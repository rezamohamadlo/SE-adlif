"""Read saved speaker-study evidence and plot diagnostics; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import local_path

COLLECTION = ROOT / 'results/experiments/shd_speaker_combined_v1'
OUTPUT = Path(__file__).with_suffix('')
PROFILES = ['augmented_v1', 'dropout030_v1', 'dropout035_v1', 'spike_delete010_v1']
MODELS = ['DTH_SE_adLIF', 'LG_DTH_SE_adLIF']
LABELS = ['A: .25 / 5%', 'B: .30 / 5%', 'C: .35 / 5%', 'D: .25 / 10%']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), values=values)


def main():
    ledger = json.loads((COLLECTION / 'RESULTS.json').read_text())
    split = json.loads((COLLECTION / 'SPLIT_MANIFEST.json').read_text())
    curves, runs = {}, []
    for record in ledger['runs']:
        directory = COLLECTION / record['profile'] / record['model'] / 'SHD' / f"seed_{record['seed']}"
        with (directory / 'evidence/metrics.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        arrays = {}
        for field in ('train_acc_epoch', 'train_loss_epoch', 'val_acc', 'val_loss'):
            selected = [r for r in rows if r.get(field)]
            epochs = [int(float(r['epoch'])) for r in selected]
            assert epochs == list(range(300)), (directory, field)
            arrays[field] = np.array([float(r[field]) for r in selected])
        epoch = record['selected_epoch']
        assert arrays['val_acc'][epoch] == arrays['val_acc'].max() == record['best_validation_accuracy']
        assert arrays['train_acc_epoch'][epoch] == record['train_accuracy_selected']
        assert arrays['train_acc_epoch'][-1] == record['train_accuracy_final']
        lrs = sorted({float(r['lr-Adam']) for r in rows if r.get('lr-Adam')})
        assert lrs == [0.01]
        assert digest(local_path(record['selected_checkpoint'])) == record['selected_checkpoint_sha256']
        test_record = local_path(record['source_test_record'])
        if test_record.suffix == '.csv':
            values = {float(r['test_acc']) for r in rows if r.get('test_acc')}
            assert values == {record['test_accuracy']}
        else:
            test = json.loads(test_record.read_text())
            assert test['metrics']['test_acc'] == record['test_accuracy']
        config = json.loads((directory / 'evidence/FULL_RUN_CONFIG.json').read_text())
        assert config['dropout'] == record['dropout']
        assert config['dataset']['train_augmentation']['spike_delete_probability'] == record['spike_delete_probability']
        curves[(record['profile'], record['model'], record['seed'])] = arrays
        adjacent = np.abs(np.diff(arrays['val_acc'][50:]))
        neighborhood = arrays['val_acc'][max(0, epoch-2):min(300, epoch+3)]
        runs.append(dict(profile=record['profile'], model=record['model'], seed=record['seed'],
                         selected_epoch=epoch, train_selected=record['train_accuracy_selected'] * 100,
                         train_final=record['train_accuracy_final'] * 100,
                         validation=record['best_validation_accuracy'] * 100,
                         test=record['test_accuracy'] * 100,
                         validation_final=arrays['val_acc'][-1] * 100,
                         validation_adjacent_change_pp=float(adjacent.mean()*100),
                         selected_minus_neighborhood_pp=float((arrays['val_acc'][epoch]-neighborhood.mean())*100),
                         first_epoch_train_above_99=int(np.flatnonzero(arrays['train_acc_epoch']>.99)[0]),
                         selected_checkpoint=record['selected_checkpoint'],
                         selected_checkpoint_sha256=record['selected_checkpoint_sha256']))
    groups, deltas = [], []
    for model in MODELS:
        for profile in PROFILES:
            records = sorted([r for r in runs if r['model']==model and r['profile']==profile], key=lambda r:r['seed'])
            groups.append(dict(model=model, profile=profile, n=3,
                               **{key:stats([r[key] for r in records]) for key in
                                  ('train_selected','train_final','validation','test')}))
            if profile != 'augmented_v1':
                baseline = sorted([r for r in runs if r['model']==model and r['profile']=='augmented_v1'], key=lambda r:r['seed'])
                deltas.append(dict(model=model, profile=profile,
                                   validation_pp=stats([r['validation']-b['validation'] for r,b in zip(records,baseline)]),
                                   test_pp=stats([r['test']-b['test'] for r,b in zip(records,baseline)])))
    data_counts = {}
    for subset in ('train','test'):
        path = ROOT / 'data/SHDWrapper' / f'shd_{subset}.h5'
        assert digest(path) == split['dataset_identity'][f'{subset}_sha256']
        with h5py.File(path,'r') as dataset:
            ids, counts = np.unique(dataset['extra/speaker'][:], return_counts=True)
            data_counts[subset] = {str(int(speaker)):int(count) for speaker,count in zip(ids,counts)}
    observed = dict(epoch9={}, epoch299={})
    for label,index in [('epoch9',9),('epoch299',299)]:
        observed[label] = {key:float(np.mean([v[key][index] for v in curves.values()])) for key in
                           ('train_acc_epoch','train_loss_epoch','val_acc','val_loss')}
    summary = dict(units='accuracy percent; differences percentage points; loss cross entropy',
                   evidence='saved CSV/JSON/configs and checkpoint hashes; no model inference or training',
                   runs=runs,groups=groups,paired_deltas_vs_A=deltas,
                   speaker_recording_counts=data_counts, fixed_learning_rate=0.01,
                   pooled_curve_observations=observed,
                   mean_adjacent_validation_change_pp=statistics.mean(r['validation_adjacent_change_pp'] for r in runs),
                   mean_selected_minus_neighborhood_pp=statistics.mean(r['selected_minus_neighborhood_pp'] for r in runs))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(12,8.5))
    colors = ['#2563a6','#c35f17']
    for metric,ax,title in [('validation',axes[0,0],'Validation-selected accuracy'),
                            ('test',axes[0,1],'Official test accuracy (exploratory)')]:
        for model,color,offset in zip(MODELS,colors,[-.12,.12]):
            selected = [next(g for g in groups if g['model']==model and g['profile']==p) for p in PROFILES]
            ax.errorbar(np.arange(4)+offset,[g[metric]['mean'] for g in selected],
                        yerr=[g[metric]['sample_sd'] for g in selected],fmt='o',capsize=4,
                        label=model.replace('_SE_adLIF','').replace('_','-'),color=color)
        ax.set_xticks(range(4),LABELS,rotation=10)
        ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend()
    epochs=np.arange(5,295)
    for model,color in zip(MODELS,colors):
        subset=[curves[('augmented_v1',model,seed)] for seed in (42,123,456)]
        label=model.replace('_SE_adLIF','').replace('_','-')
        for key,style,kind in [('train_acc_epoch','-','train'),('val_acc','--','validation')]:
            mean=np.mean([r[key] for r in subset],axis=0)*100
            axes[1,0].plot(epochs,np.convolve(mean,np.ones(11)/11,mode='valid'),style,color=color,label=f'{label} {kind}')
        for key,style,kind in [('train_loss_epoch','-','train'),('val_loss','--','validation')]:
            mean=np.mean([r[key] for r in subset],axis=0)
            axes[1,1].plot(epochs,np.convolve(mean,np.ones(11)/11,mode='valid'),style,color=color,label=f'{label} {kind}')
    axes[1,0].set_ylabel('Accuracy (%)');axes[1,0].set_ylim(75,101)
    axes[1,1].set_ylabel('Cross entropy');axes[1,1].set_ylim(0,1.8)
    for ax,title in zip(axes[1],['Baseline A: learning curves','Baseline A: training and validation loss']):
        ax.set_title(title);ax.set_xlabel('Epoch (zero-based)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Speaker-holdout SHD: four settings, DTH/LG-DTH, three seeds',fontsize=14)
    fig.text(.5,.015,'Top: means +/- sample SD across seeds 42/123/456. Settings show dropout / deletion.\n'
             'Bottom: baseline A means across three seeds, centered 11-epoch smoothing. Online train and eval validation differ.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.955))
    fig.savefig(OUTPUT/'evaluation.png',dpi=180)
    fig.savefig(OUTPUT/'evaluation.svg')
    svg_path = OUTPUT/'evaluation.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',
                        encoding='utf-8')
    plt.close(fig)
    print(f'Verified {len(runs)} runs, selected checkpoint hashes and original HDF5 identities; saved {OUTPUT.relative_to(ROOT)}')


if __name__=='__main__':
    main()
