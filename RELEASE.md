# v0.1.0 — Gold-medal model weights

Framework and classification checkpoints from the **21st-place solo gold-medal** RSNA Knee Abnormality Detection solution (**0.964 public / 0.955 private**).

| Files | Role | License |
|---|---|---|
| `llm_fold0.pt` … `llm_fold4.pt` | Five ConvNeXt-Small models after two-epoch LLM-label continuation | Apache-2.0 |
| `distilled_fold0.pt` … `distilled_fold4.pt` | Five ImageNet-initialized ConvNeXt-Small students trained on 75% LLM + 25% teacher probabilities | Apache-2.0 |
| `public_convnext_small.pt` | Unmodified nartaa public model, originally `E_prior_s42_fall.pt` | CC BY-NC 4.0; attribution to nartaa and Heliosli |

Download through `knee-mri download --output weights`. The command verifies the versioned SHA-256 manifest and retrieves the anatomy and orientation auxiliaries from their original Kaggle datasets.

Within each five-fold branch, average logits before sigmoid. Blend branch probabilities with weights **45% / 45% / 10%**. Use the documented 384/352 and 320/288 input paths; no report text is required for inference.

The ten five-fold weights contain the active model tensors from the submitted checkpoints; disabled report-state heads were removed. Raw DICOMs, reports, annotation tables, credentials and optimizer states are not release assets.

See the repository README for installation, inference, data preparation, grouped training, and same-fold self-distillation. Third-party model licenses are not overridden by the repository's Apache-2.0 license.
