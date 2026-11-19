"""Descriptive matched-control reports; no fitting or population inference."""
import os
import tempfile

from scripts.adaptation_gradient_deletion.contracts import GUIDED, RANDOM
from scripts.adaptation_gradient_pilot.methods import publish, require


def mean(values):
    values=list(values)
    return float(sum(values)/len(values)) if values else None


def summarize(cases, models):
    paired=[];effects=[];contrasts=[];eligible=[]
    for case in cases:
        sid=case['sample_id'];target=case['target'];clean=case['conditions'][0]
        require(clean['condition']=='clean','Clean must precede matched conditions')
        clean_scores={r['model']:r for r in clean['evaluations']}
        clean_benefit=clean_scores[models[0]]['loss']-clean_scores[models[1]]['loss']
        for condition in case['conditions']:
            require([r['model'] for r in condition['evaluations']]==models,'Model order differs for a shared input')
            parent,variant=condition['evaluations']
            benefit=parent['loss']-variant['loss']
            paired.append(dict(sample_id=sid,seed=42,parent_model=models[0],variant_model=models[1],target=target,
                speaker=case['speaker'],condition=condition['condition'],replicate=condition['replicate'],
                input_sha256=condition['input_sha256'],deletion_budget=condition['budget'],
                parent_loss=parent['loss'],variant_loss=variant['loss'],loss_benefit=benefit,
                clean_loss_benefit=clean_benefit,benefit_change_from_clean=benefit-clean_benefit,
                parent_actual_loss_change=parent['actual_loss_change'],variant_actual_loss_change=variant['actual_loss_change'],
                parent_surrogate_linear_prediction=parent['surrogate_linear_prediction'],
                variant_surrogate_linear_prediction=variant['surrogate_linear_prediction']))
            for row in condition['evaluations']:
                effects.append(dict(sample_id=sid,seed=42,target=target,speaker=case['speaker'],
                    condition=condition['condition'],replicate=condition['replicate'],
                    input_sha256=condition['input_sha256'],deletion_budget=condition['budget'],**row))
        guided=next((r for r in case['conditions'] if r['condition']==GUIDED),None)
        random=[r for r in case['conditions'] if r['condition']==RANDOM]
        require(len(random)==3 and {r['replicate'] for r in random}=={0,1,2},'Exactly three matched random controls are required')
        if guided is not None:
            eligible.append(sid)
            for model in models:
                g=next(r for r in guided['evaluations'] if r['model']==model)
                random_changes=[next(r for r in c['evaluations'] if r['model']==model)['actual_loss_change'] for c in random]
                contrasts.append(dict(model=model,seed=42,sample_id=sid,guided_actual_loss_change=g['actual_loss_change'],
                    mean_random_actual_loss_change=mean(random_changes),guided_abs_loss_change=abs(g['actual_loss_change']),
                    mean_random_abs_loss_change=mean(abs(v) for v in random_changes),
                    guided_minus_random_abs_loss_change=abs(g['actual_loss_change'])-mean(abs(v) for v in random_changes)))
    groups=[]
    for model in models:
        contrast=[r for r in contrasts if r['model']==model]
        clean=[r for r in effects if r['model']==model and r['condition']=='clean']
        groups.append(dict(model=model,clean_records=len(clean),clean_accuracy=mean(r['correct'] for r in clean),
            clean_mean_loss=mean(r['loss'] for r in clean),common_guidance_eligible_records=len(contrast),
            mean_abs_guided_loss_change=mean(r['guided_abs_loss_change'] for r in contrast),
            mean_abs_random_loss_change=mean(r['mean_random_abs_loss_change'] for r in contrast),
            mean_guided_minus_random_abs_loss_change=mean(r['guided_minus_random_abs_loss_change'] for r in contrast)))
    summary=dict(status='matched_deletion_executed_pending_review',samples=len(cases),seed=42,n_training_seeds=1,
        primary_repeat=0,groups=groups,guided_eligible_records=len(eligible),common_guidance_eligible_ids=eligible,
        guidance_unavailable_ids=[r['sample_id'] for r in cases if r['sample_id'] not in eligible],
        paired_condition_records=len(paired),model_evaluation_records=len(effects),
        shared_deletion_inputs=sum(len(r['conditions'])-1 for r in cases),contrast_records=len(contrasts),
        training=False,backward=False,input_optimization=False,stage7_complete=False,population_inference=False,
        interpretation='random scalar effects average three realizations within record; guided/random contrasts use the same eligible records',
        limitations=['one seed and forty validation recordings; exploratory diagnostic, not full-validation perturbation coverage',
                    'absolute MT-guided selection is coupled to its gradient/loss; effects need not increase CE or prove model superiority',
                    'SLAYER surrogate linear predictions need not match finite hard-spike CE differences',
                    'record 3896 is retained; repeated gradients/random controls are not independent training seeds',
                    'no significance test, confidence interval, regression, official test or optimized input'])
    return summary,paired,effects,contrasts


def write_figure(root, contrasts, models, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    figure,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for ax,model in zip(axes,models):
        group={r['sample_id']:r for r in contrasts if r['model']==model}
        positions=[i for i,sid in enumerate(ids) if sid in group]
        ax.scatter(positions,[group[ids[i]]['guided_actual_loss_change'] for i in positions],label='MT-guided',s=20)
        ax.scatter(positions,[group[ids[i]]['mean_random_actual_loss_change'] for i in positions],label='Random: mean of 3 within record',s=20,marker='x')
        ax.axhline(0,color='black',linewidth=.6);ax.set(ylabel='CE change from fresh clean',title=model)
        if 3896 in group:
            i=ids.index(3896);ax.annotate('3896',(i,group[3896]['guided_actual_loss_change']),xytext=(4,5),textcoords='offset points')
        if not positions:ax.text(.5,.5,'No eligible guided controls',transform=ax.transAxes,ha='center')
        ax.legend()
    axes[-1].set_xlabel('Fixed pilot record order (not independent seeds)')
    figure.suptitle('SHD* strict-precision 1% exact-budget deletion; exploratory one-seed pilot')
    figure.tight_layout();path=directory/'matched_deletion_per_record.png'
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.tmp',dir=directory)
    try:
        with os.fdopen(descriptor,'wb') as stream:figure.savefig(stream,format='png',dpi=150)
        publish(path,name)
    finally:plt.close(figure)
    return path
