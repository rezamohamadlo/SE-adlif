import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from evaluate_tta import batch_predictions, gains_for
import runner


class TTATests(unittest.TestCase):
    def test_identity_matches_original_logits_and_padding_is_ignored(self):
        x = torch.tensor([[[100., -100.], [2., -1.], [-1., 2.], [900., -900.]]])
        y = torch.tensor([[-1, 0]])
        mask = torch.tensor([[0, 1, 1, 0]])
        before = x.clone()
        base, probs, labels = batch_predictions(lambda t: t, (x, y, mask), 'mean_membrane_potentials', [1.])
        torch.testing.assert_close(base, torch.tensor([[.5, .5]]))
        torch.testing.assert_close(probs, base.softmax(-1))
        torch.testing.assert_close(x, before)
        self.assertEqual(labels.tolist(), [0])

    def test_ecg_labels_remain_aligned_and_probabilities_are_averaged(self):
        x = torch.tensor([[[2., 0.], [0., 1.], [999., 0.]]])
        targets = torch.tensor([[-1, 0, 1]])
        blocks = torch.tensor([[1, 2, 0]])
        base, probs, labels = batch_predictions(lambda t: t, (x, targets, blocks),
                                               'summed_membrane_potentials', [1., .95, 1.05])
        expected = torch.stack([(x[:, :2]*g).reshape(2, 2).softmax(-1) for g in [1., .95, 1.05]]).mean(0)
        torch.testing.assert_close(probs, expected)
        self.assertEqual(labels.tolist(), [0, 1])

    def test_augmentation_does_not_update_model_parameters(self):
        model = torch.nn.Linear(2, 2).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        batch_predictions(model, (torch.ones(1, 2, 2), torch.tensor([[-1, 1]]), torch.ones(1, 2, dtype=torch.long)),
                          'mean_membrane_potentials', [1., .95, 1.05])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_policy_rejects_invalid_views(self):
        for values in [[.95, 1.05], [1., 1.], [1., float('nan')], [1., -1.]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                gains_for({'datasets': {'SSC': {'gains': values}}}, 'SSC')

    def test_runner_tta_never_calls_training(self):
        with patch('evaluate_tta.main') as evaluate, patch.object(runner, 'run') as train:
            runner.main(['--models', 'MT_SE_adLIF', '--datasets', 'SHD', 'SSC', 'ECG',
                         '--seeds', '42', '123', '456', '--tta', '--dry-run'])
            evaluate.assert_called_once()
            train.assert_not_called()
            self.assertIn('--dry-run', evaluate.call_args.args[0])

    def test_candidate_dispatch_never_trains(self):
        for model in ('AMT_SE_adLIF', 'MT_TC_SE_adLIF', 'LT_DTH_SE_adLIF', 'SA_MT_SE_adLIF'):
            with self.subTest(model=model), patch('evaluate_tta.main') as evaluate, patch.object(runner, 'run') as train:
                runner.main(['--models', model, '--datasets', 'SHD', 'SSC', 'ECG',
                             '--seeds', '42', '123', '456', '--tta', '--dry-run'])
                args = evaluate.call_args.args[0]
                self.assertIn(model, args)
                self.assertIn('--dry-run', args)
                self.assertIn('results/experiments/' + model.removesuffix('_SE_adLIF').lower() + '_tta_v1', args)
                train.assert_not_called()

    def test_runner_rejects_training_overrides_with_tta(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(['--models', 'MT_SE_adLIF', '--tta', '--epochs', '1'])

    def test_dth_dispatch_has_separate_outputs_and_no_training(self):
        with patch('evaluate_tta.main') as evaluate, patch.object(runner, 'run') as train:
            runner.main(['--models', 'DTH_SE_adLIF', '--datasets', 'SSC', '--seeds', '42', '--tta', '--dry-run'])
            args = evaluate.call_args.args[0]
            self.assertIn('DTH_SE_adLIF', args)
            self.assertIn('results/experiments/dth_tta_v1', args)
            train.assert_not_called()


    def test_mt_dth_dispatch_uses_separate_output_without_training(self):
        with patch('evaluate_tta.main') as evaluate, patch.object(runner, 'run') as train:
            runner.main(['--models', 'MT_DTH_SE_adLIF', '--datasets', 'SHD',
                         '--seeds', '42', '123', '456', '--tta', '--dry-run'])
            args = evaluate.call_args.args[0]
            self.assertIn('MT_DTH_SE_adLIF', args)
            self.assertIn('results/experiments/mt_dth_tta_v1', args)
            train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
