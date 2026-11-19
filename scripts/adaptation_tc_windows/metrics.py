"""Explicit output-window descriptors; no model execution on import."""
import numpy as np


def describe(probabilities,target,support_frames,window=5,burn_in=10):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError('Expected finite frame class probabilities')
    if not 0<=target<p.shape[1] or not 0<=support_frames<=len(p) or window<1 or burn_in<0:
        raise ValueError('Invalid support, label or window definition')
    available=max(0,support_frames-burn_in)
    count=available//window
    result=dict(valid_windows=count,valid_support_frames=available,dropped_tail_frames=available%window,
                adjacent_js_nats=None,true_class_concentration=None,true_class_mean_probability=None,
                window_accuracy=None,top_class_switch_fraction=None,undefined_reason=None)
    if count<2:
        result['undefined_reason']='fewer_than_two_complete_windows';return result
    windows=p[burn_in:burn_in+count*window].reshape(count,window,-1).mean(1)
    left,right=windows[:-1],windows[1:];mid=(left+right)/2
    tiny=np.finfo(np.float64).tiny
    kl=lambda a,b:np.sum(a*(np.log(np.maximum(a,tiny))-np.log(np.maximum(b,tiny))),axis=1)
    result['adjacent_js_nats']=float(np.maximum(0,.5*(kl(left,mid)+kl(right,mid))).mean())
    mass=windows[:,target];total=mass.sum()
    if total>0:
        result['true_class_concentration']=float(np.clip((count*np.sum((mass/total)**2)-1)/(count-1),0,1))
    else:result['undefined_reason']='zero_true_class_mass'
    pred=windows.argmax(1)
    result.update(true_class_mean_probability=float(mass.mean()),window_accuracy=float(np.mean(pred==target)),
                  top_class_switch_fraction=float(np.mean(pred[:-1]!=pred[1:])),
                  true_class_window_probability=mass.tolist(),window_predicted_class=pred.tolist())
    return result
