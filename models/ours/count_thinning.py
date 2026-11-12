"""Training-only independent event deletion; ordinary SE-adLIF inference."""
from models.pl_module import MLPSNN
from functional.count_thinning import thin_input_counts


class CountThinningMLPSNN(MLPSNN):
    def forward(self, inputs):
        return super().forward(thin_input_counts(inputs,self.hparams.cfg.spike_delete_probability,self.training))
