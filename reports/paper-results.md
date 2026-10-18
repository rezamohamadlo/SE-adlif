# Paper results - SE-adLIF

Two-layer SE-adLIF results from Baronig et al., *Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation*, [Table 2, PDF page 11](../reference_paper.pdf#page=11). Parameter counts, means and SDs follow the published table.

| Dataset | Model | Recurrent | Parameters | Runs | Test accuracy (mean ± SD, %) | Hidden spikes/sample | Dense GFLOPs/sample |
|---|---|---|---:|---:|---:|---|---|
| SHD* | SE-adLIF (2 layers) | yes | 0.45M | 20 | 93.79 ± 0.76 | Not reported | Not reported |
| SSC | SE-adLIF (2 layers) | yes | 1.6M | 5 | 80.44 ± 0.26 | Not reported | Not reported |
| ECG | SE-adLIF (2 layers) | yes | 4.6k | 20 | 88.18 ± 0.36 | Not reported | Not reported |

The paper does not report hidden-spike counts or dense GFLOPs for these classification results. Its bottleneck spike rates in [Table 3, PDF page 15](../reference_paper.pdf#page=15) concern audio compression and are not classification hidden-layer counts. Local-reference measurements are reported separately.

SHD* reserves 20% of the official training data for validation and selects the checkpoint by validation accuracy before final test evaluation.

[Paper versus reference](paper-vs-reference.md) | [Reference versus proposed models](reference-vs-models.md)
