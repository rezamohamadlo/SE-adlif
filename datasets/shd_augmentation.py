"""Training-only count-frame augmentation, applied after deterministic caching."""
import math

import torch


class SHDFrameAugmentation:
    VERSION = 'shd_frame_augmentation_v1'

    def __init__(self, version, time_scale, channel_shift, spike_delete_probability,
                 ignore_first_timesteps=10, pad_to_min_size=250):
        if (version != self.VERSION or len(time_scale) != 2
                or not all(math.isfinite(float(s)) for s in time_scale)
                or not 0 < time_scale[0] <= time_scale[1]
                or isinstance(channel_shift, bool) or not isinstance(channel_shift, int)
                or channel_shift < 0 or not 0 <= spike_delete_probability <= 1):
            raise ValueError('Invalid SHD augmentation version/settings')
        self.time_scale = tuple(time_scale)
        self.channel_shift = channel_shift
        self.spike_delete_probability = spike_delete_probability
        self.ignore_first_timesteps = ignore_first_timesteps
        self.pad_to_min_size = pad_to_min_size or 0

    def __call__(self, frames, target, block_idx):
        if (frames.ndim != 2 or not len(frames) or frames.device.type != 'cpu'
                or not torch.isfinite(frames).all() or (frames < 0).any()
                or not torch.equal(frames, frames.round()) or len(block_idx) != len(frames)):
            raise ValueError('SHD augmentation expects CPU count frames and aligned block indices')
        # Rebin whole time bins without duplicating counts; collisions add.
        # This is a frame-level approximation, not sub-bin event-time warping.
        scale = torch.empty(()).uniform_(*self.time_scale).item()
        destination = torch.floor(torch.arange(len(frames)) * scale).long()
        length = max(self.pad_to_min_size, int(destination[-1]) + 1)
        augmented = frames.new_zeros((length, frames.shape[1]))
        augmented.index_add_(0, destination, frames)
        shift = int(torch.randint(-self.channel_shift, self.channel_shift + 1, ()).item())
        if shift:
            shifted = torch.zeros_like(augmented)
            if abs(shift) < frames.shape[1]:
                if shift > 0:
                    shifted[:, shift:] = augmented[:, :-shift]
                else:
                    shifted[:, :shift] = augmented[:, -shift:]
            augmented = shifted  # Discard boundary events; never wrap channels.
        if self.spike_delete_probability:
            # Independent deletion per spike, including bins containing >1 spike.
            augmented = torch.binomial(augmented, torch.full_like(
                augmented, 1 - self.spike_delete_probability))
        blocks = block_idx.new_ones(length)
        blocks[:self.ignore_first_timesteps] = 0
        return augmented, target, blocks
