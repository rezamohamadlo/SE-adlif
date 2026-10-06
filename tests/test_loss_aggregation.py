import unittest

import torch

from models.pl_module import aggregate_temporal_outputs


class TemporalLossAggregationTests(unittest.TestCase):
    def setUp(self):
        # Timesteps 0..9 are burn-in, 10..12 are valid, and 13 is padding.
        self.outputs = torch.tensor(
            [
                [[0.03 * t, -0.02 * t, 0.01 * (t + 1)] for t in range(14)]
            ],
            dtype=torch.float64,
        )
        self.block_idx = torch.tensor(
            [[0] * 10 + [1, 1, 1] + [0]], dtype=torch.int64
        )

    def aggregate(self, mode, outputs=None):
        if outputs is None:
            outputs = self.outputs
        return aggregate_temporal_outputs(
            outputs=outputs,
            block_idx=self.block_idx,
            num_blocks=2,
            loss_agg=mode,
        )[:, 1]

    def test_shd_is_sum_of_per_timestep_softmax(self):
        actual = self.aggregate("sum_softmax_over_time")
        expected = torch.softmax(self.outputs[:, 10:13], dim=-1).sum(dim=1)
        torch.testing.assert_close(actual, expected)

    def test_ssc_is_sum_of_membrane_potentials(self):
        actual = self.aggregate("summed_membrane_potentials")
        expected = self.outputs[:, 10:13].sum(dim=1)
        torch.testing.assert_close(actual, expected)

    def test_burn_in_and_padding_do_not_contribute(self):
        changed = self.outputs.clone()
        changed[:, :10] += 10_000
        changed[:, 13] -= 10_000
        for mode in ("sum_softmax_over_time", "summed_membrane_potentials"):
            torch.testing.assert_close(
                self.aggregate(mode), self.aggregate(mode, outputs=changed)
            )

    def test_shd_and_ssc_aggregations_are_different(self):
        self.assertFalse(
            torch.allclose(
                self.aggregate("sum_softmax_over_time"),
                self.aggregate("summed_membrane_potentials"),
            )
        )

    def test_invalid_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported loss_agg"):
            self.aggregate("unknown_aggregation")


if __name__ == "__main__":
    unittest.main()
