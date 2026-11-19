"""Shared supervised objective and constrained temporal event relocation."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.adaptation_gradient_pilot.methods import require, require_frozen, gradient_summary


def objective(model,batch,cfg,device,with_grad=False):
    from models.pl_module import aggregate_temporal_outputs
    x,targets,blocks=[v.to(device) for v in batch]
    require(targets.dtype==torch.long,'Classification targets must be int64')
    leaf=x.detach().clone().requires_grad_(True) if with_grad else x
    outputs=model(leaf)
    logits=aggregate_temporal_outputs(outputs,blocks,targets.shape[1],cfg['loss_agg'])
    valid=targets!=-1;require(valid.any(1).all().item() and not valid[:,0].any().item(),'Invalid supervised target mask')
    ce=F.cross_entropy(logits.transpose(1,2).float(),targets,ignore_index=-1,reduction='none')
    losses=(ce*valid).sum(1)/valid.sum(1)
    require(torch.isfinite(outputs).all().item() and torch.isfinite(losses).all().item(),'Nonfinite forward/CE')
    prediction=logits.argmax(-1);rows=[]
    for loss,p,t,mask in zip(losses.detach().cpu().tolist(),prediction.detach().cpu(),targets.cpu(),valid.cpu()):
        preds=p[mask].tolist();truth=t[mask].tolist()
        rows.append(dict(loss=loss,predictions=preds,valid_steps=len(preds),
            correct_steps=sum(a==b for a,b in zip(preds,truth)),accuracy=sum(a==b for a,b in zip(preds,truth))/len(preds)))
    return losses,leaf,rows


def forward(model,batch,cfg,device):
    with torch.no_grad():return objective(model,batch,cfg,device)[2]


def backward(model,batch,cfg,device,physical,burn,spec,expected_state):
    require(not torch.is_inference_mode_enabled(),'Gradient computation cannot use inference_mode')
    require_frozen(model,expected_state)
    clean=forward(model,batch,cfg,device)[0]
    with torch.enable_grad():
        losses,leaf,rows=objective(model,batch,cfg,device,True)
        require(losses.numel()==1,'Per-record backward requires a singleton')
        gradient=torch.autograd.grad(losses.sum(),leaf,create_graph=False,retain_graph=False)[0][0].detach().cpu().numpy().copy()
    require(np.isfinite(gradient).all(),'Nonfinite input gradient');require_frozen(model,expected_state)
    observed=rows[0];bound=spec['score_atol']+spec['score_rtol']*abs(clean['loss'])
    check=dict(prediction_matches=observed['predictions']==clean['predictions'],loss_difference=observed['loss']-clean['loss'],loss_bound=bound)
    check['passed']=check['prediction_matches'] and abs(check['loss_difference'])<=bound
    return dict(clean=clean,autograd_forward=observed,autograd_forward_check=check,
        **gradient_summary(gradient,np.abs(physical),burn,spec['profile_bins'])),gradient


def validate_native(original,candidate,domain,budget):
    a,b=np.asarray(original),np.asarray(candidate)
    require(a.shape==b.shape and np.isfinite(b).all(),'Input length/shape/finite constraint failed')
    if domain=='counts':
        require(np.all(b>=0) and np.equal(b,np.floor(b)).all() and np.array_equal(a.sum(0),b.sum(0)),
                'Nonnegative integer/per-channel count constraint failed')
        require(np.all(b<=a.max(0,keepdims=True)),'Candidate exceeds original per-channel maximum count')
    else:
        require(domain=='signed' and np.isin(b,[-1,0,1]).all() and
                all(np.array_equal((a==sign).sum(0),(b==sign).sum(0)) for sign in [-1,1]),'Signed event/polarity budget failed')
    require(np.all((b[a!=0]==0)|(b[a!=0]==a[a!=0])) if domain=='signed' else np.all(b[a!=0]<=a[a!=0]),
            'Edits must remove original events and add only into original zero cells')
    moved=float(np.abs(b.astype(np.float64)-a).sum()/2)
    require(moved<=budget and moved==int(moved),'Relocation exceeds registered event budget')
    return int(moved)


def budget_for(x,domain,fraction):
    n=int(x.sum()) if domain=='counts' else int(np.count_nonzero(x))
    return max(1,int(np.floor(n*fraction))) if n else 0


def propose(original,gradient,domain,budget):
    """Project the surrogate linear objective from original inputs each step.

    Move only original events to original zero cells within the same channel;
    count caps and ECG signs stay native. Budget is never accumulated across steps.
    """
    a=np.asarray(original);g=np.asarray(gradient)[:len(a)]
    require(a.shape==g.shape and np.isfinite(g).all(),'Proposal gradient alignment differs')
    result=a.copy();remaining=np.abs(a).astype(np.int64);groups=[];alloc={}
    for channel in range(a.shape[1]):
        zero=np.flatnonzero(a[:,channel]==0)
        for sign in ([1] if domain=='counts' else [-1,1]):
            sources=np.flatnonzero(a[:,channel]>0 if domain=='counts' else a[:,channel]==sign)
            if not len(sources) or not len(zero):continue
            src=sources[np.lexsort((sources,sign*g[sources,channel]))]
            dst=zero[np.lexsort((zero,-sign*g[zero,channel]))]
            cap=int(a[:,channel].max()) if domain=='counts' else 1
            groups.append(dict(channel=channel,sign=sign,src=src,dst=dst,sp=0,dp=0,cap=cap))
    for _ in range(budget):
        choices=[]
        for number,group in enumerate(groups):
            c=group['channel'];sign=group['sign']
            while group['sp']<len(group['src']) and remaining[group['src'][group['sp']],c]==0:group['sp']+=1
            while group['dp']<len(group['dst']):
                dest=group['dst'][group['dp']]
                if (domain=='counts' and result[dest,c]<group['cap']) or (domain=='signed' and result[dest,c]==0):break
                group['dp']+=1
            if group['sp']==len(group['src']) or group['dp']==len(group['dst']):continue
            source,dest=group['src'][group['sp']],group['dst'][group['dp']]
            gain=float(sign*(g[dest,c]-g[source,c]))
            if gain>0:choices.append((gain,-number,source,dest,number))
        if not choices:break
        _,_,source,dest,number=max(choices);group=groups[number];c=group['channel'];sign=group['sign']
        result[source,c]-=sign;result[dest,c]+=sign;remaining[source,c]-=1
        alloc[c,sign]=alloc.get((c,sign),0)+1
    moves=validate_native(a,result,domain,budget)
    allocation=[dict(channel=c,sign=sign,moves=n) for (c,sign),n in sorted(alloc.items())]
    require(sum(r['moves'] for r in allocation)==moves,'Move allocation differs from actual edit budget')
    return result,allocation


def random_relocation(original,allocation,domain,seed):
    a=np.asarray(original);result=a.copy();rng=np.random.default_rng(seed)
    for row in allocation:
        c,sign,n=row['channel'],row['sign'],row['moves'];zeros=np.flatnonzero((a[:,c]==0)&(result[:,c]==0))
        if domain=='counts':
            total=int(a[:,c].sum());selected=rng.choice(total,size=n,replace=False)
            sources=np.searchsorted(np.cumsum(a[:,c].astype(np.int64)),selected,side='right')
            cap=int(a[:,c].max());slots=rng.choice(len(zeros)*cap,size=n,replace=False);dest=zeros[slots//cap]
        else:
            sources=rng.choice(np.flatnonzero(a[:,c]==sign),size=n,replace=False)
            dest=rng.choice(zeros,size=n,replace=False)
        np.add.at(result[:,c],sources,-sign);np.add.at(result[:,c],dest,sign)
    validate_native(a,result,domain,sum(r['moves'] for r in allocation))
    return result
