# Saved experiment evidence

These files make the experiments inspectable without downloading weights or retraining.

| Directory | Contents |
|---|---|
| `news_ablation/` | Initial-screen predictions, metrics, and comparison metadata, copied from the original `models/news-ablation-869989/` runs |
| `pretrained_ladder/` | Pretrained feature-ladder predictions and summaries for 2022–2025 |
| `walk_forward/` | Controlled evaluation metrics, per-window losses, and run metadata |
| `permutation_pretrained/` | Pretrained feature-swap experiment |
| `permutation_control21/`, `permutation_control21_t0/` | Control-feature swap experiments |
| `permutation_finetuned/` | Matched pretrained and fine-tuned feature-importance results |

`news_ablation/gdelt_fed-seed42/validation_predictions.csv` supplies the offline demo:
50 forecast origins × 5 horizons × 3 models = 750 rows. Its original
`comparison_metadata.json` includes model revision, covariates, training configuration,
and source hashes. The demo recomputes scores directly from predictions and actuals.

Historical metadata is preserved as evidence, including paths and package versions
from the original runs. Paths to original `models/` directories are historical records,
not an indication that trained adapters are bundled. Reproducing model inference or
training requires installing the modeling dependencies and obtaining model weights.
