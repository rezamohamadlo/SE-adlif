import unittest
from types import SimpleNamespace

import torch

from models.pl_module import MLPSNN


class ClassificationTargetDtypeTests(unittest.TestCase):
    def test_int32_labels_match_int64_loss_and_gradients(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                context = SimpleNamespace(auto_regression=False,
                                          output_func='summed_membrane_potentials',
                                          ignore_target_idx=-1,
                                          loss=torch.nn.CrossEntropyLoss(ignore_index=-1))
                outputs = torch.randn(3, 4, 5, device=device, requires_grad=True)
                blocks = torch.zeros(3, 4, dtype=torch.long, device=device)
                targets = torch.tensor([[2], [1], [-1]], dtype=torch.int32, device=device)
                compute = MLPSNN.process_predictions_and_compute_losses
                _, actual, _ = compute(context, outputs, targets, blocks)
                _, expected, _ = compute(context, outputs, targets.long(), blocks)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                self.assertTrue(torch.isfinite(outputs.grad).all())


if __name__ == '__main__':
    unittest.main()
