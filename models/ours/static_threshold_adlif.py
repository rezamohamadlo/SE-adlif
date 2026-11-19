"""Mechanism control: DTH's bounded static thresholds, zero frozen activity gains."""
import torch

from models.ours.dth_adlif import DTHSEAdLIF


class StaticThresholdSEAdLIF(DTHSEAdLIF):
    """Retain DTH initialization and scan; activity cannot change thresholds.

    Trace computation is deliberately retained for the controlled ablation.
    This class makes no inference-speed claim. Static threshold logits learn;
    both zero-initialized activity gains remain registered but nontrainable.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.homeostatic_gain.requires_grad_(False)
        self.transient_gain.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('homeostatic_gain', 'transient_gain'):
            value = state_dict.get(prefix+name)
            if value is not None and not torch.equal(value, torch.zeros_like(value)):
                error_msgs.append(f'{prefix}{name}: static threshold control requires zero activity gains')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
