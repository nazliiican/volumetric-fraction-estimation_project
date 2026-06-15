# Soil Composition Estimation - Final Results Summary

## Dataset
161 physical mixtures, 5 photographs each (805 images), 6 materials
(CL, GP1, GP2, MH1, MH2, SP). Each mixture contains 1-3 materials.

## Evaluation protocol
All models share one leakage-free split **grouped by `Code`** (seed 42): the 5
images of a mixture always stay in the same split. Metrics use the same
functions in `src/metrics.py`. The headline numbers are **test set, mixture
level, mass space** (the 5 image predictions per Code averaged and renormalized).

## Main comparison

| Model | MAE | RMSE | R2 | Presence F1 | FP absent mass (%) |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.125 | 0.173 | 0.412 | 0.680 | 32.1 |
| Softmax CNN | 0.100 | 0.154 | 0.535 | 0.780 | 19.8 |
| Sparse Multitask CNN | 0.097 | 0.152 | 0.545 | 0.796 | 19.1 |

## Key findings
- The biggest improvement is Random Forest -> Softmax CNN: pretrained CNN
  features carry far more compositional signal than hand-crafted colour/texture.
- The Sparse Multitask CNN is best on every metric, but only modestly better
  than the softmax CNN; most of the gain is from the learned representation.
- The explicit presence head reduces but does not eliminate absent-material
  mass (~19% remains).
- Coarse gravels (GP1, GP2) are predicted well; remaining error concentrates in
  visually similar fine powders (CL, MH1, MH2, SP).

## Limitations
Small dataset (161 mixtures); single grouped split; non-independent images per
mixture; near-identical densities; visually similar fine powders; frozen
backbone; soft (non-zeroing) presence gating.

## Conclusion
The sparse multitask model achieved the best overall performance, but the
largest gain came from replacing hand-crafted features with pretrained CNN
features. A simple, frozen transfer-learning backbone with an explicit
presence-aware composition head is well suited to this small, grouped
image-composition dataset.
