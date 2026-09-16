# Reproducibility

Install the environment from `environment.yml` or `requirements.txt`, then run from the repository root with `src` on `PYTHONPATH`.

```bash
python scripts/evaluate.py --data_root data/SimoLFA_Dataset/test --weights_path model_artifacts/siamese_attention_resnet34_supplementary_checkpoint.pth --output_dir results/test_set_evaluation --batch_size 32 --threshold 0.5 --k 3.0
```

The fixed test partition contains 4,054 patches. The evaluation produces aggregate metrics, per-patch predictions, and figures. Renaming files does not alter image content or predictions.

Training uses online stochastic augmentation. The archived curves describe the paper-associated run; a new run is not expected to reproduce every epoch-level value exactly. Test metrics are patch-level and must not be interpreted as patient-level clinical performance.
