"""SSC count inputs and deterministic pilot transforms with physical support."""
import hashlib
from scripts.adaptation_analysis.pilot import digest
from scripts.adaptation_analysis.features import measure_features
from scripts.adaptation_analysis.perturbations import thin_counts,warp_counts

def input_rows(data, ids, condition, study, features=True):
    name, replicate, transform, value = condition
    result = []
    for sample_id in ids:
        source = data.load(sample_id)
        x = source.pop('counts')
        perturb_seed = int(digest([study['selection_seed'], sample_id, name, replicate])[:16], 16)
        if transform == 'warp':
            x = warp_counts(x, value)
        elif transform == 'thin':
            x = thin_counts(x, value, perturb_seed)
        row = {k: v for k, v in source.items() if k not in ('framing', 'duration_ms', 'support_frames', 'stored_frames', 'frame_support_ms')}
        row.update(condition=name, replicate=replicate, perturbation_seed=perturb_seed,
                   input_sha256=hashlib.sha256(str(x.shape).encode() + x.astype('<i8', copy=False).tobytes()).hexdigest(),
                   original_duration_ms=source['duration_ms'], original_support_frames=source['support_frames'],
                   transformed_support_frames=len(x), stored_frames=max(len(x), study['minimum_frames']))
        if features:
            row.update(measure_features(x, **study['features']))
        result.append((sample_id, x, row))
    return result

class InputSequence:
    """Rebuild only one batch at a time, preserving saved validation order."""
    def __init__(self,data,ids,condition,study,feature_cache=None):
        self.data=data;self.ids=list(ids);self.condition=condition;self.study=study;self.feature_cache=feature_cache
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        if not isinstance(index,slice):raise TypeError('Inference requires batch slices')
        batch=input_rows(self.data,self.ids[index],self.condition,self.study,features=False)
        if self.feature_cache is None:return batch
        enriched=[]
        for sid,counts,row in batch:
            saved=self.feature_cache[sid]
            if any(saved[k]!=v for k,v in row.items()):raise ValueError('SSC reconstructed input differs from feature cache')
            enriched.append((sid,counts,saved))
        return enriched
