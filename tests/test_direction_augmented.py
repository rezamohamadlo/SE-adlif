"""No training: preserve canonical subsets, clean validation and cache transforms."""
import contextlib
import io
import unittest
import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from project_paths import ROOT, local_path
from datasets.shd_augmentation import SHDFrameAugmentation


class DirectionAugmentationTests(unittest.TestCase):
    def test_real_split_and_training_only_transform(self):
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg=compose(config_name='main',overrides=['experiment=SHD_DIRECTION_AUG_DTH_RDROP_SE_adLIF','+comparison=SHD',
                '++dataset.validation_only_data=true'])
        cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        cfg.dataset.data_path=str(local_path(cfg.dataset.data_path))
        with contextlib.redirect_stdout(io.StringIO()):data=hydra.utils.instantiate(cfg.dataset)
        self.assertEqual(len(data.data_train),6525);self.assertEqual(len(data.data_val),1631)
        self.assertFalse(set(data.data_train.indices)&set(data.data_val.indices))
        train=data.train_dataloader();valid=data.val_dataloader()
        self.assertIsInstance(train.dataset.full_transform,SHDFrameAugmentation)
        self.assertIsNone(valid.dataset.full_transform)
        sample=data.data_train[0];x,y,blocks=sample
        before=x.clone();label=int(y);torch.manual_seed(42)
        a,ay,ab=data.train_augmentation(x,y,blocks)
        self.assertTrue(torch.equal(x,before));self.assertEqual(int(ay),label)
        self.assertEqual(len(a),len(ab));self.assertTrue(torch.equal(a,a.round()))
        self.assertTrue(torch.isfinite(a).all() and (a>=0).all())
        self.assertEqual(data.train_augmentation.time_scale,(.9,1.1))
        self.assertEqual(data.train_augmentation.channel_shift,2)
        self.assertEqual(data.train_augmentation.spike_delete_probability,.05)
        clean1=data.data_val[0];clean2=data.data_val[0]
        for a,b in zip(clean1,clean2):self.assertTrue(torch.equal(torch.as_tensor(a),torch.as_tensor(b)))

if __name__=='__main__':unittest.main()
