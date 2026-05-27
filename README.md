# VenomRec

This repository contains the minimal VIP5/T5-small code path used for the VenomRec experiments. It keeps only the source code needed to generate poisoned data, fine-tune VIP5, evaluate recommendation metrics with ER, and evaluate text stealthiness with ROUGE.

The repository intentionally does not include datasets, extracted image features, checkpoints, logs, caches, notebooks, or generated poisoned artifacts.

## Code Layout

- `attack/ours/dcip_ieos/`: DCIP-IEOS poisoning implementation.
- `src/`: VIP5 model, data loading, training, and adapter code.
- `scripts/`: command-line entry points for fine-tuning, evaluation, candidate cache generation, and text-quality evaluation.
- `tools/export_competition_pool.py`: helper for rebuilding competition pools.
- `analysis/results/{clothing,sports}/`: target and high-popularity item lists used by the main experiments.

## Environment

Install a CUDA-compatible PyTorch build for your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The code expects to be run from the repository root:

```bash
export PYTHONPATH=.
```

## Required Local Artifacts

Place the dataset and feature files using this layout:

```text
data/
  clothing/
    sequential_data.txt
    negative_samples.txt
    datamaps.json
    review_splits.pkl
    exp_splits.pkl
    user_id2idx.pkl
    user_id2name.pkl
    meta.json.gz
  sports/
    ...
features/
  vitb32_features/
    clothing/
      <item>.npy
    sports/
      <item>.npy
snap/
  beauty/
    0805/
      NoAttack_0.0_beauty-vitb32-2-8-20/
        BEST_EVAL_LOSS.pth
```

If your pretrained VIP5 checkpoint is stored elsewhere, pass `--victim-ckpt` to `run_dcip_ieos.py` and `--load` to `scripts/run_finetune.sh`.

## Clothing Experiment

Generate poisoned data:

```bash
python -m attack.ours.dcip_ieos.run_dcip_ieos \
  clothing 0.001 0 vitb32 2 8 12 \
  --sequence-length 10 \
  --interaction-rounds 4 \
  --mask-vis-ratio 0.15 \
  --mask-txt-ratio 0.18 \
  --img-eps 0.05 \
  --txt-ratio-max 0.20 \
  --txt-embed-eps 0.01 \
  --min-txt-replacements 2 \
  --sim-threshold 0.92 \
  --seed 2022 \
  --victim-device cuda:0 \
  --run-tag mr0p001_full \
  --poison-subdir dcip_ieos_fc_mr0.001_ir4_img0.05_txt0.2
```

Optionally precompute direct-task candidates:

```bash
python scripts/precompute_candidates.py \
  --dataset clothing \
  --attack-mode NoAttack \
  --mr 0.0 \
  --candidate-num 99 \
  --poison-subdir dcip_ieos_fc_mr0.001_ir4_img0.05_txt0.2
```

Fine-tune with the poisoned data under the NoAttack interface:

```bash
bash scripts/run_finetune.sh clothing NoAttack 0 0,1,2,3 vitb32 2 8 20 \
  --poison_subdir dcip_ieos_fc_mr0.001_ir4_img0.05_txt0.2
```

Evaluate recommendation metrics and ER:

```bash
python scripts/evaluate_direct.py \
  --ckpt snap/clothing/<timestamp>/NoAttack_0.0_clothing-vitb32-2-8-20/BEST_EVAL_LOSS.pth \
  --task direct \
  --prompts B-5 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda:0 \
  --attack-mode NoAttack \
  --mr 0.0 \
  --split clothing \
  --poison-subdir dcip_ieos_fc_mr0.001_ir4_img0.05_txt0.2 \
  --eval-er
```

Evaluate text stealthiness:

```bash
python scripts/eval_text_quality.py \
  --dataset clothing \
  --poison-subdir dcip_ieos_fc_mr0.001_ir4_img0.05_txt0.2
```

## Sports Experiment

Rebuild the sports competition pool:

```bash
python tools/export_competition_pool.py \
  --dataset sports \
  --pop-path analysis/results/sports/high_pop_items_sports_highcount_100.txt \
  --targets-path analysis/results/sports/low_pop_items_sports_lowcount_1.txt \
  --output-dir attack/ours/dcip_ieos/caches \
  --c-size 8 \
  --keywords-top 50 \
  --min-keywords 50 \
  --feat-root features \
  --feat-backbone vitb32_features \
  --no-kmeans
```

Regenerate the popular-center prototype if needed:

```bash
rm -f attack/ours/dcip_ieos/caches/prototypes/sports/c_pop.pkl
python - <<'PY'
from attack.ours.dcip_ieos.prototypes import build_or_load_pop_center
vec = build_or_load_pop_center(
    split_dir='data/sports',
    cache_dir='attack/ours/dcip_ieos/caches',
    feat_root='features',
    feat_backbone='vitb32_features',
    pop_path='analysis/results/sports/high_pop_items_sports_highcount_100.txt',
    fallback_anchors=None,
)
print(f'c_pop dim={len(vec)}, first3={vec[:3]}')
PY
```

Generate poisoned data:

```bash
python -m attack.ours.dcip_ieos.run_dcip_ieos \
  sports 0.001 0 vitb32 2 8 12 \
  --sequence-length 10 \
  --interaction-rounds 4 \
  --mask-vis-ratio 0.15 \
  --img-eps 0.05 \
  --mask-txt-ratio 0.18 \
  --txt-ratio-max 0.20 \
  --txt-embed-eps 0.01 \
  --min-txt-replacements 2 \
  --sim-threshold 0.92 \
  --seed 2022 \
  --victim-device cuda:0 \
  --run-tag mr0p001_full \
  --poison-subdir dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92
```

Optionally precompute direct-task candidates:

```bash
python scripts/precompute_candidates.py \
  --dataset sports \
  --attack-mode NoAttack \
  --mr 0.0 \
  --candidate-num 99 \
  --poison-subdir dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92
```

Fine-tune:

```bash
bash scripts/run_finetune.sh sports NoAttack 0 0,1,2,3 vitb32 2 8 20 \
  --poison_subdir dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92
```

Evaluate recommendation metrics and ER:

```bash
python scripts/evaluate_direct.py \
  --ckpt snap/sports/<timestamp>/NoAttack_0.0_sports-vitb32-2-8-20/BEST_EVAL_LOSS.pth \
  --task direct \
  --prompts B-5 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda:0 \
  --attack-mode NoAttack \
  --mr 0.0 \
  --split sports \
  --poison-subdir dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92 \
  --eval-er
```

Evaluate text stealthiness:

```bash
python scripts/eval_text_quality.py \
  --dataset sports \
  --poison-subdir dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92
```

Main sports poison subdirectory:

```text
dcip_ieos_ir4_mvisr0.15_imgeps0.05_mtxtr0.18_txtrmax0.20_txtembeps0.01_mintxtreplace2_simthres0.92
```
