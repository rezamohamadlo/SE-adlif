"""Comparable stage-7 figures and descriptive seed-matched intervention contrasts."""
import csv
import json
import statistics

import numpy as np

from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_deletion.contracts import atomic_csv
from scripts.adaptation_gradient_pilot.methods import array_hash,require
from scripts.adaptation_stage7.methods import validate_native
from scripts.adaptation_stage7.runtime import job_root,load_input,seal,verify


def mean_sd(values):
    values=list(values)
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,n_training_seeds=len(values))


def finalize(prepared):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,spec,m=[prepared[k] for k in ('root','spec','manifest')]
    jobs={};transfers={};outputs=[root/'MANIFEST.json'];summaries=[]
    outputs += [root/'CONTRACT_TESTS.json',root/'CONTRACT_TESTS.txt']
    outputs += list((root/'run_provenance').glob('*'))
    for run in m['runs']:
        directory=job_root(root,run)
        if not (directory/'COMPLETE.json').exists():return False
        c=verify(directory);outputs += [directory/'COMPLETE.json']+[__import__('project_paths').local_path(n) for n in c['outputs']]
        jobs[run['dataset'],run['model'],run['seed']]=directory
        summaries.append(pilot.read_json(directory/'SUMMARY.json'))
    for ds in spec['datasets']:
        for seed in spec['seeds']:
            directory=root/'transfers'/ds/f'seed_{seed}'
            if not (directory/'COMPLETE.json').exists():return False
            c=verify(directory);outputs += [directory/'COMPLETE.json']+[__import__('project_paths').local_path(n) for n in c['outputs']]
            transfers[ds,seed]=pilot.read_json(directory/'TRANSFER_ROWS.json')
    figures=root/'figures';figures.mkdir(exist_ok=True)
    def save(figure,name):
        path=figures/name;figure.tight_layout();figure.savefig(path,dpi=150);plt.close(figure);outputs.append(path)
    profile_data={};record_data={}
    raw_magnitudes=[]
    for key,directory in jobs.items():
        records=[r for r in pilot.read_json(directory/'GRADIENT_SUMMARIES.json') if r['repeat']==0]
        record_data[key]={r['sample_id']:r for r in records}
        for r in records:
            raw_magnitudes.append(dict(dataset=key[0],model=key[1],seed=key[2],sample_id=r['sample_id'],
                physical_abs_gradient_sum=r['physical_abs_gradient_sum'],physical_gradient_l2=r['physical_gradient_l2'],
                event_cell_abs_gradient_sum=r['event_cell_abs_gradient_sum'],
                synthetic_padding_abs_gradient_sum=r['synthetic_padding_abs_gradient_sum'],
                normalization_status=r['normalization_status']))
        maps=[r['relative_time_abs_mass'] for r in records if r['relative_time_abs_mass'] is not None]
        profile_data[key]=np.mean(maps,axis=0) if maps else None
    atomic_csv(root/'RAW_GRADIENT_MAGNITUDES.csv',raw_magnitudes);outputs.append(root/'RAW_GRADIENT_MAGNITUDES.csv')
    for ds in spec['datasets']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,pair in zip(axes.flat,spec['pairs']):
            for model in [pair['parent'],pair['variant']]:
                means=[profile_data[ds,model,seed] for seed in spec['seeds'] if profile_data[ds,model,seed] is not None]
                if means:
                    values=np.stack(means);center=values.mean(0);sd=values.std(0,ddof=1) if len(values)>1 else np.zeros(20)
                    positions=(np.arange(20)+.5)*5;ax.plot(positions,center,label=model)
                    ax.fill_between(positions,np.maximum(0,center-sd),center+sd,alpha=.15)
            ax.set(title=pair['id'],xlabel='Relative physical-input position (%)',ylabel='Within-record normalized |surrogate gradient|')
            ax.legend(fontsize=7)
        fig.suptitle(f'{"SHD*" if ds=="SHD" else ds}: equal-record profiles; shaded sample SD across 3 training seeds')
        save(fig,f'{ds}_paired_gradient_profiles.png')
        if ds!='ECG':
            classes=20 if ds=='SHD' else 35;matrices=[]
            a=prepared['adapters'][ds]
            for model in spec['models']:
                matrix=[]
                for cls in range(classes):
                    seed_maps=[]
                    for seed in spec['seeds']:
                        maps=[r['relative_time_abs_mass'] for sid,r in record_data[ds,model,seed].items()
                              if int(a.native.labels[sid])==cls and r['relative_time_abs_mass'] is not None]
                        if maps:seed_maps.append(np.mean(maps,0))
                    matrix.append(np.mean(seed_maps,0) if seed_maps else np.full(20,np.nan))
                matrices.append(np.asarray(matrix))
            vmax=max((np.nanmax(a) for a in matrices if np.isfinite(a).any()),default=1.)
            fig,axes=plt.subplots(3,2,figsize=(11,12),sharex=True,sharey=True)
            for ax,model,matrix in zip(axes.flat,spec['models'],matrices):
                image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=vmax,extent=[0,100,classes,0]);ax.set(title=model,xlabel='Relative input position (%)',ylabel='True class')
            fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.6,label='Mean normalized physical gradient mass')
            fig.suptitle(f'{ds}: class profiles, same color scale for all six unique models; 2 records/class, 3 seeds')
            save(fig,f'{ds}_class_gradient_profiles.png')
        # All selected diagnostic examples/seeds are shown; no best-pattern selection.
        a=prepared['adapters'][ds]
        for sid in a.diagnostic_ids:
            original=a.input(sid)
            for seed in spec['seeds']:
                gradients=[];deltas=[];diagnostics=[]
                for model in spec['models']:
                    directory=jobs[ds,model,seed]
                    with np.load(directory/'gradients'/f'sample_{sid}_rep_0.npz',allow_pickle=False) as z:g=z['gradient'][:len(original)].astype(np.float64)
                    absolute=np.abs(g);mass=absolute.sum();gradients.append(absolute/mass if mass else np.full_like(absolute,np.nan))
                    optimized=load_input(directory/'optimized'/f'sample_{sid}.npz');deltas.append(optimized-original)
                    diagnostics.append(next(r for r in pilot.read_json(directory/'DIAGNOSTICS.json') if r['sample_id']==sid))
                vmax=max((np.nanmax(g) for g in gradients if np.isfinite(g).any()),default=1.)
                change_max=max(1.,max(float(np.abs(d).max()) for d in deltas))
                fig,axes=plt.subplots(6,3,figsize=(13,14),sharex=True,sharey=True)
                extent=[0,len(original)*(a.bin_ms/1000 if a.bin_ms else 1),0,original.shape[1]]
                for row,(model,g,delta,diagnostic) in enumerate(zip(spec['models'],gradients,deltas,diagnostics)):
                    axes[row,0].imshow(original.T,aspect='auto',origin='lower',extent=extent,vmin=-1 if a.domain=='signed' else 0,vmax=1 if a.domain=='signed' else max(1,float(original.max())))
                    axes[row,1].imshow(g.T,aspect='auto',origin='lower',extent=extent,vmin=0,vmax=vmax)
                    axes[row,2].imshow(delta.T,aspect='auto',origin='lower',extent=extent,cmap='coolwarm',vmin=-change_max,vmax=change_max)
                    axes[row,0].set_ylabel(model+'\ninput channel',fontsize=7)
                    axes[row,2].text(.01,.98,f"{diagnostic['status']}; moved {diagnostic['moves']}/{diagnostic['budget']}",
                        transform=axes[row,2].transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.7))
                for ax,title in zip(axes[0],['Original native input','Normalized physical |input gradient|','Optimized − original (native events)']):ax.set_title(title,fontsize=9)
                for ax in axes[-1]:ax.set_xlabel('Time (s)' if a.bin_ms else 'Model step (physical ms unverified)')
                fig.suptitle(f'{ds} record {sid}, seed {seed}: fixed model weights; 1% relocation budget; labels held as diagnostic targets')
                save(fig,f'{ds}_sample_{sid}_seed_{seed}_diagnostics.png')
    paired=[]
    for ds in spec['datasets']:
        for seed in spec['seeds']:
            rows=transfers[ds,seed]
            index={(r['target'],r['origin'],r['sample_id'],r['condition'],r['replicate']):r for r in rows}
            for pair in spec['pairs']:
                for origin_role in ['parent','variant']:
                    origin=pair[origin_role]
                    for sid in prepared['adapters'][ds].diagnostic_ids:
                        def get(target,condition,rep=0):return index[target,origin,sid,condition,rep]
                        parent,variant=get(pair['parent'],'optimized'),get(pair['variant'],'optimized')
                        require(parent['input_sha256']==variant['input_sha256'],'Pair did not receive the same optimized input')
                        random_changes=[]
                        for rep in range(3):
                            p,v=get(pair['parent'],'random',rep),get(pair['variant'],'random',rep)
                            require(p['input_sha256']==v['input_sha256'] and p['moves']==parent['moves'],'Random controls not shared/budget matched')
                            random_changes.append(p['loss_change']-v['loss_change'])
                        optimized_change=parent['loss_change']-variant['loss_change'];random_mean=statistics.mean(random_changes)
                        paired.append(dict(dataset=ds,pair=pair['id'],seed=seed,sample_id=sid,origin_role=origin_role,origin_model=origin,
                            moved_events=parent['moves'],optimized_benefit_change=optimized_change,mean_random_benefit_change=random_mean,
                            optimized_minus_random=optimized_change-random_mean))
    aggregates=[]
    for ds in spec['datasets']:
        for pair in spec['pairs']:
            for role in ['parent','variant']:
                subset=[r for r in paired if r['dataset']==ds and r['pair']==pair['id'] and r['origin_role']==role]
                per_seed=[dict(seed=seed,**{field:statistics.mean(r[field] for r in subset if r['seed']==seed)
                    for field in ['optimized_benefit_change','mean_random_benefit_change','optimized_minus_random']}) for seed in spec['seeds']]
                aggregates.append(dict(dataset=ds,pair=pair['id'],origin_role=role,per_seed=per_seed,
                    **{field:mean_sd(r[field] for r in per_seed) for field in ['optimized_benefit_change','mean_random_benefit_change','optimized_minus_random']}))
    atomic_csv(root/'PAIRED_DIAGNOSTIC_EFFECTS.csv',paired);outputs.append(root/'PAIRED_DIAGNOSTIC_EFFECTS.csv')
    pilot.write_json(root/'SEED_AGGREGATES.json',aggregates);outputs.append(root/'SEED_AGGREGATES.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,ds in zip(axes,spec['datasets']):
        group=[r for r in aggregates if r['dataset']==ds and r['origin_role']=='parent']
        means=[r['optimized_minus_random']['mean'] for r in group];sd=[r['optimized_minus_random']['sample_sd'] for r in group]
        ax.errorbar(range(4),means,yerr=sd,fmt='o',capsize=4);ax.axhline(0,color='black',linewidth=.6)
        ax.set_xticks(range(4),[r['pair'] for r in group],rotation=30);ax.set_title('SHD*' if ds=='SHD' else ds)
        ax.set_ylabel('Benefit change: optimized − matched random (CE)')
    fig.suptitle('Parent-optimized diagnostic inputs; mean ± sample SD across 3 training seeds, 4 fixed records/seed')
    save(fig,'paired_diagnostic_intervention_effects.png')
    required=dict(checkpoint_jobs=len(summaries),expected_checkpoint_jobs=54,
        primary_gradient_records=sum(r['physical_records'] for r in summaries),expected_primary_gradient_records=2520,
        repeat_checks=sum(r['repeat_checks'] for r in summaries),expected_repeat_checks=432,
        repeat_failures=sum(r['repeat_failures'] for r in summaries),autograd_failures=sum(r['autograd_failures'] for r in summaries),
        optimization_autograd_failures=sum(r['optimization_autograd_failures'] for r in summaries),
        optimized_attempts=sum(r['optimized_attempts'] for r in summaries),expected_optimized_attempts=216,
        skipped_unstable_inputs=sum(r['skipped_unstable_inputs'] for r in summaries),
        accepted_diagnostic_inputs=sum(r['accepted_diagnostic_inputs'] for r in summaries),transfer_bundles=len(transfers),
        transfer_rows=sum(len(r) for r in transfers.values()),expected_transfer_rows=5184,
        paired_diagnostic_rows=len(paired),expected_paired_diagnostic_rows=288)
    stable=required['repeat_failures']==required['autograd_failures']==required['skipped_unstable_inputs']==required['optimization_autograd_failures']==0
    scope_complete=(required['checkpoint_jobs']==54 and required['primary_gradient_records']==2520 and required['repeat_checks']==432 and
        required['optimized_attempts']==216 and required['transfer_rows']==5184 and len(paired)==288 and stable)
    audit=dict(status='verified_scoped_complete' if scope_complete else 'requires_numerical_followup',
        stage7_scoped_complete=scope_complete,checks=required,
        interpretation='Completion of registered exploratory three-dataset scope, not exhaustive validation gradients or independent confirmation',
        objectives=['common frozen-weight supervised input gradients for all four parent/variant pairs',
                    'native-domain, fixed-length, per-channel event-budgeted diagnostic input optimization',
                    'identical input transfer and matched random temporal-relocation controls',
                    'comparable figures, seed-matched CE effects and explicit negative outcomes'],
        limitations=['diagnostic cohorts only: SHD* 40, SSC 70, ECG 30; only 8 repeat records and 4 optimized records per dataset',
            'SLAYER sensitivities are surrogate derivatives; optimized labels held fixed do not guarantee unchanged semantic class',
            'parent/variant-optimized selection is model/loss coupled; no causal or architectural superiority claim',
            'ECG per-step targets, original all-zero vectors mapping to class 0 and model-step time units preserved',
            'no independent official test, no exact Nature figure reproduction, no new model-weight training',
            'source_high baseline remains separate; shared parents/records/random replicates are not independent seeds'])
    pilot.write_json(root/'STAGE7_CLOSEOUT_AUDIT.json',audit);outputs.append(root/'STAGE7_CLOSEOUT_AUDIT.json')
    pilot.write_json(root/'JOB_SUMMARIES.json',summaries);outputs.append(root/'JOB_SUMMARIES.json')
    lines=['# Stage-7 registered diagnostic scope','',f'Closure: {audit["status"]}.',
        'All four model pairs, three datasets and three saved training seeds. Frozen weights; no official test.',
        'Full strict validation references are separate from source_high baseline scores.',
        f'Primary gradient records: {required["primary_gradient_records"]}; repeat checks: {required["repeat_checks"]}; input-optimization attempts: {required["optimized_attempts"]}.',
        f'Accepted nonzero diagnostic inputs: {required["accepted_diagnostic_inputs"]}; ineffective/negative attempts remain in full traces.',
        f'Shared transfer scores: {required["transfer_rows"]}; paired diagnostic rows: {len(paired)}.', '',
        '| Dataset | Pair | Parent-optimized minus random benefit change (mean ± seed SD, CE) |', '|---|---|---:|']
    for r in aggregates:
        if r['origin_role']=='parent':
            metric=r['optimized_minus_random'];lines.append(f"| {r['dataset']} | {r['pair']} | {metric['mean']:+.7g} ± {metric['sample_sd']:.7g} |")
    lines += ['', 'Positive benefit change means the variant loses less supervised performance than its parent under the same diagnostic input.',
        'These are model-selected constrained diagnostic perturbations, not natural accuracy improvements or causal estimates.',
        'Mean/SD summarize three matching training seeds after averaging four fixed records within seed. No p-values or confidence intervals.',
        'See STAGE7_CLOSEOUT_AUDIT.json for scope, checklist and limitations. Stage 8 synthesis and cost reporting remain separate.','']
    (root/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');outputs.append(root/'SUMMARY.md')
    seal(root,m,outputs,stage7_scoped_complete=scope_complete)
    return scope_complete
