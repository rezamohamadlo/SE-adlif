"""Explicit train-only augmentation for a separately declared canonical SHD* study."""
from datasets.shd import SHDLDM
from datasets.shd_augmentation import SHDFrameAugmentation


class AugmentedSHDLDM(SHDLDM):
    def __init__(self, training_frame_augmentation, **kwargs):
        if (kwargs.get('validate_on') != .2 or kwargs.get('random_seed') != 42
                or not kwargs.get('validation_only_data') or kwargs.get('train_augmentation')
                or kwargs.get('additional_test_set_validation')):
            raise ValueError('Requires isolated canonical SHD* training, fixed split42, no test validation')
        super().__init__(**kwargs)
        # Original deterministic caches and split indices are unchanged.
        # full_transform in the inherited training loader runs AFTER cache reads.
        self.train_augmentation = SHDFrameAugmentation(
            **training_frame_augmentation,
            ignore_first_timesteps=self.ignore_first_timesteps,
            pad_to_min_size=self.pad_to_min_size)
