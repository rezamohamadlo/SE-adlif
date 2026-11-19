"""Presentation-only export from a sealed stage-7 package; no neural execution.

Preserves all frozen scientific outputs. Correct protocol labels, explicit color
scales and shared scales across diagnostic seeds live in a derived package.
"""
import argparse

import numpy as np

from project_paths import local_path,portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.methods import require
from scripts.adaptation_stage7.runtime import seal,verify

LABEL={'SHD':'SHD*','SSC':'SSC','ECG':'ECG'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',default='results/experiments/adaptation_stage7_suite_v1')
    args=parser.parse_args(argv);root=local_path(args.output_root,must_exist=True);parent=verify(root)
    require(parent['stage7_scoped_complete'],'Presentation requires verified scoped completion')
    m=pilot.read_json(root/'MANIFEST.json');spec=m['specification'];destination=root/'presentation_v1'
    if (destination/'COMPLETE.json').exists():verify(destination);print('Presentation already verified.');return 0
    destination.mkdir(exist_ok=True)
    manifest=dict(parent_complete_sha256=pilot.sha256(root/'COMPLETE.json'),parent_manifest_sha256=pilot.digest(m),
        role='derived presentation only; no weights/inference/backward or evidence changes',
        source_sha256=pilot.sha256(__file__),protocol_labels=LABEL,diagnostic_color_scaling='shared across all six models and three seeds per physical record')
    if (destination/'MANIFEST.json').exists():pilot.validate_manifest(pilot.read_json(destination/'MANIFEST.json'),manifest)
    else:pilot.write_json(destination/'MANIFEST.json',manifest)
    outputs=[destination/'MANIFEST.json'];figures=destination/'figures';figures.mkdir(exist_ok=True)
    cohort={(r['dataset'],r['sample_id']):r for r in pilot.read_json(root/'cohort_snapshot/MANIFEST.json')['samples']}
    inputs={};targets={}
    for key,item in cohort.items():
        path=local_path(item['path'],must_exist=True);require(pilot.sha256(path)==item['sha256'],'Native cohort snapshot changed')
        with np.load(path,allow_pickle=False) as z:inputs[key]=z['input'].copy();targets[key]=z['targets'].copy()
    records={}
    for ds in spec['datasets']:
        for model in spec['models']:
            for seed in spec['seeds']:
                job=root/'jobs'/ds/model/f'seed_{seed}'
                records[ds,model,seed]={r['sample_id']:r for r in pilot.read_json(job/'GRADIENT_SUMMARIES.json') if r['repeat']==0}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        path=figures/name;fig.savefig(path,dpi=160);plt.close(fig);outputs.append(path)
    for ds in spec['datasets']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout='constrained')
        for ax,pair in zip(axes.flat,spec['pairs']):
            for model in [pair['parent'],pair['variant']]:
                means=[]
                for seed in spec['seeds']:
                    maps=[r['relative_time_abs_mass'] for r in records[ds,model,seed].values() if r['relative_time_abs_mass'] is not None]
                    if maps:means.append(np.mean(maps,axis=0))
                if means:
                    means=np.stack(means);center=means.mean(0);sd=means.std(0,ddof=1) if len(means)>1 else np.zeros(20)
                    time=(np.arange(20)+.5)*5;ax.plot(time,center,label=f'{model} (n={len(means)} seeds)')
                    ax.fill_between(time,np.maximum(0,center-sd),center+sd,alpha=.15)
            ax.set(title=pair['id'],xlabel='Relative physical-input position (%)',ylabel='Within-record normalized |surrogate gradient|');ax.legend(fontsize=7)
        fig.suptitle(f'{LABEL[ds]}: equal-record means and sample SD across training seeds; no causal attribution')
        save(fig,f'{ds}_paired_gradient_profiles.png')
        if ds!='ECG':
            classes=20 if ds=='SHD' else 35;matrices=[]
            for model in spec['models']:
                rows=[]
                for cls in range(classes):
                    seed_means=[]
                    for seed in spec['seeds']:
                        maps=[r['relative_time_abs_mass'] for sid,r in records[ds,model,seed].items()
                            if int(targets[ds,sid][1])==cls and r['relative_time_abs_mass'] is not None]
                        if maps:seed_means.append(np.mean(maps,0))
                    rows.append(np.mean(seed_means,0) if seed_means else np.full(20,np.nan))
                matrices.append(np.asarray(rows))
            cap=max((np.nanmax(x) for x in matrices if np.isfinite(x).any()),default=1.)
            fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,sharey=True,layout='constrained')
            for ax,model,matrix in zip(axes.flat,spec['models'],matrices):
                image=ax.imshow(matrix,vmin=0,vmax=cap,aspect='auto',extent=[0,100,classes,0]);ax.set(title=model,xlabel='Relative physical-input position (%)',ylabel='True class')
            fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.65,label='Normalized physical gradient mass')
            fig.suptitle(f'{LABEL[ds]}: class profiles, common color scale; two records/class and three seeds')
            save(fig,f'{ds}_class_gradient_profiles.png')
        for sid in m['cohorts'][ds]['diagnostic_ids']:
            original=inputs[ds,sid];item=cohort[ds,sid];maps={};changes={};diagnostics={}
            for seed in spec['seeds']:
                for model in spec['models']:
                    job=root/'jobs'/ds/model/f'seed_{seed}'
                    with np.load(job/'gradients'/f'sample_{sid}_rep_0.npz',allow_pickle=False) as z:g=np.abs(z['gradient'][:len(original)].astype(np.float64))
                    mass=g.sum();maps[model,seed]=g/mass if mass else np.full_like(g,np.nan)
                    with np.load(job/'optimized'/f'sample_{sid}.npz',allow_pickle=False) as z:changes[model,seed]=z['input']-original
                    diagnostics[model,seed]=next(r for r in pilot.read_json(job/'DIAGNOSTICS.json') if r['sample_id']==sid)
            grad_cap=max((np.nanmax(x) for x in maps.values() if np.isfinite(x).any()),default=1.)
            delta_cap=max(1.,max(float(np.abs(x).max()) for x in changes.values()))
            for seed in spec['seeds']:
                fig,axes=plt.subplots(6,3,figsize=(14,14),sharex=True,sharey=True,layout='constrained')
                extent=[0,len(original)*(item['bin_ms']/1000 if item['bin_ms'] else 1),0,original.shape[1]]
                for row,model in enumerate(spec['models']):
                    d=diagnostics[model,seed]
                    im0=axes[row,0].imshow(original.T,aspect='auto',origin='lower',extent=extent,vmin=-1 if item['domain']=='signed' else 0,vmax=1 if item['domain']=='signed' else max(1,float(original.max())))
                    im1=axes[row,1].imshow(maps[model,seed].T,aspect='auto',origin='lower',extent=extent,vmin=0,vmax=grad_cap)
                    im2=axes[row,2].imshow(changes[model,seed].T,aspect='auto',origin='lower',extent=extent,vmin=-delta_cap,vmax=delta_cap,cmap='coolwarm')
                    axes[row,0].set_ylabel(model+'\ninput channel',fontsize=7)
                    axes[row,2].text(.01,.98,f"{d['status']}; moved {d['moves']}/{d['budget']}",transform=axes[row,2].transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.75))
                for ax,title in zip(axes[0],['Original native input','Normalized physical |surrogate gradient|','Accepted diagnostic input − original']):ax.set_title(title,fontsize=9)
                for ax in axes[-1]:ax.set_xlabel('Time (s)' if item['bin_ms'] else 'Model step; physical ms unverified')
                for column,image,label in [(0,im0,'Native events'),(1,im1,'Within-record gradient mass'),(2,im2,'Native event change')]:
                    fig.colorbar(image,ax=axes[:,column].tolist(),shrink=.5,label=label)
                target_label=f'true class {int(targets[ds,sid][1])}' if ds!='ECG' else 'mean CE over original valid temporal labels'
                fig.suptitle(f'{LABEL[ds]}, record {sid}, seed {seed}, {target_label}\nFixed weights; 1% within-channel relocation; all 18 model/seed panels share scales')
                save(fig,f'{ds}_sample_{sid}_seed_{seed}_diagnostics.png')
    aggregates=pilot.read_json(root/'SEED_AGGREGATES.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,ds in zip(axes,spec['datasets']):
        group=[r for r in aggregates if r['dataset']==ds and r['origin_role']=='parent']
        ax.errorbar(range(4),[r['optimized_minus_random']['mean'] for r in group],yerr=[r['optimized_minus_random']['sample_sd'] for r in group],fmt='o',capsize=4)
        ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(range(4),[r['pair'] for r in group],rotation=30)
        ax.set(title=LABEL[ds],ylabel='Optimized − random benefit change (CE)')
    fig.suptitle('Parent-optimized diagnostic inputs; mean ± sample SD of three training-seed means')
    save(fig,'paired_diagnostic_intervention_effects.png')
    source=(root/'SUMMARY.md').read_text(encoding='utf-8')
    source=source.replace('| SHD |','| SHD* |')
    source+='\nPresentation uses explicit SHD* labels and shared diagnostic color scales across six models and three seeds. Frozen scientific outputs remain unchanged.\n'
    (destination/'SUMMARY.md').write_text(source,encoding='utf-8');outputs.append(destination/'SUMMARY.md')
    seal(destination,manifest,outputs,presentation_only=True)
    print('Derived presentation verified outputs:',len(verify(destination)['outputs']));return 0


if __name__=='__main__':raise SystemExit(main())
