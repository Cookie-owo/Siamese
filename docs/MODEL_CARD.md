# Siamese Attention ResNet34 Model Card

## Model identity

- Algorithm: Siamese
- Architecture: AttentionResNet34
- Backbone: ResNet-34
- Reference branch: single-channel binary reference map
- Attention locations: layer3 and layer4
- Output: one binary-classification logit
- Pretrained backbone: no
- Current released checkpoint: `model_artifacts/siamese_attention_resnet34_supplementary_checkpoint.pth`
- Checkpoint role: released checkpoint used for the current evaluation

## Intended use

The model is intended for research evaluation of paired RGB image patches and reference-map inputs in the SimoLFA workflow. It may also be used for exploratory folder-level inference when the input data and access conditions are approved.

The model is not released as a clinically validated diagnostic device. Patch-level metrics must not be interpreted as patient-level diagnostic performance.

## Inputs

The model receives two tensors with matching spatial dimensions:

- RGB tensor: `[3, 224, 224]`;
- reference-map tensor: `[1, 224, 224]`;
- RGB normalization: ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`;
- reference-map range: `[0, 1]` after nearest-neighbor resizing.

## Reference-map generation

The RGB image is converted to grayscale. A threshold equal to `mean - K × standard_deviation` is applied with `cv2.THRESH_BINARY_INV`, using `K = 3.0`. No additional pixel inversion is applied.

## Training record

The historical configuration records 100 epochs, batch size 32, AdamW optimization, learning rate `1e-4`, weight decay `1e-5`, positive-class weight 2.0, seed 42, deterministic execution enabled, and online stochastic augmentation.

## Reported checkpoint evaluation

On the current 4,054-patch SimoLFA test partition, the released checkpoint produces:

| Metric | Value |
|---|---:|
| Accuracy | 0.9842 |
| Precision | 0.9857 |
| Sensitivity / recall | 0.9827 |
| Specificity | 0.9857 |
| F1-score | 0.9842 |
| ROC-AUC | 0.9984 |
| Average precision | 0.9986 |

These are patch-level historical-partition results. The partition has no exact cross-split SHA-256 duplicate files after replacement. The meaning of filename-derived source prefixes requires confirmation before making source-level independence claims.

## Limitations

- The checkpoint is the model artifact associated with the current released evaluation.
- Patient-level separation is not established in the available records.
- The current test partition is a patch-level partition; its source-group semantics require confirmation.
- The model has not been clinically validated.
- The folder-level positive threshold is `0.088` (8.8%), as defined by the paper's negative-control calibration.
- Performance may depend on acquisition, device, image quality, and preprocessing distributions.
- Confidence intervals and external-cohort validation are not included in the current release.

## Reproducibility

Use `configs/release_supplementary_inference.yaml`, the checkpoint SHA-256 file, and the formal scripts under `scripts/`. Every evaluation should archive the effective configuration, environment, command, and checkpoint hash.
