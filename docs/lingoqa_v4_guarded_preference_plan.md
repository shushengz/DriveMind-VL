# LingoQA v4 Guarded Preference Plan

## Why v4

SFT-v2 visual-scale training worked as a normal-answer learner: clean dev normal
F1 rose from `0.2763` to `0.3680`. However, text-only, wrong-image, and
blank-image controls also rose. The model became better at answering LingoQA
questions, but it still often answers from priors instead of depending on the
correct frames.

The next step should not be another ordinary SFT epoch. It should preserve the
new normal-answer ability while explicitly pushing down control-set hallucination.

## Objective Choice

The trainer now supports:

- `dpo`: classic preference optimization with an optional reference model;
- `simpo`: reference-free preference optimization using length-normalized
  log-probability margins;
- `orpo`: reference-free odds-ratio preference loss.

For the next run, use `simpo` because it is reference-free, memory-efficient,
and aligns with the length-normalized completion log-probability already used by
the project trainer.

## Data Design

`data/processed/lingoqa_preference_v4_guarded_train.jsonl`

Pairs:

- normal: 300
- text-only control: 300
- blank-image control: 300
- wrong-image control: 600
- total: 1500

The important change from v3 is that every normal sample has a normal answer
over refusal pair. This guards against blanket refusal. Controls are still
dominant through pair count and weights:

- normal weight: 0.7
- text-only weight: 1.0
- blank-image weight: 1.0
- wrong-image weight: 1.35
- counting control boost: 1.25
- spatial control boost: 1.15
- reasoning control boost: 1.10

This directly targets the observed SFT-v2 failures: counting and spatial rows
had negative per-case control gaps, and wrong-image remained high.

## Default Training Command

Use SFT-v2 as the starting adapter:

```bash
INIT_ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v4_guarded_simpo \
LOSS_TYPE=simpo \
LR=3e-6 \
DPO_BETA=0.08 \
SIMPO_GAMMA=0.05 \
CHOSEN_SFT_WEIGHT=0.08 \
EPOCHS=1 \
GRAD_ACCUM=8 \
MAX_IMAGES=5 \
MAX_PIXELS=401408 \
bash scripts/48_train_lingoqa_pref_v4_guarded.sh
```

## Success Criteria

On clean dev:

- normal F1 should stay near SFT-v2, preferably `>= 0.34`;
- wrong-image F1 should drop meaningfully below SFT-v2 `0.2965`;
- blank-image and text-only should drop or at least not increase;
- per-case gap should move above zero, ideally `>= +0.03`;
- positive-gap rate should improve over SFT-v2 `0.22`.

If normal F1 collapses, reduce preference strength:

- `LR=2e-6`
- `DPO_BETA=0.05`
- `CHOSEN_SFT_WEIGHT=0.12`

If controls remain high while normal is preserved, increase control pressure:

- `WRONG_REPEAT=3` when rebuilding v4 data;
- or train a short second preference pass with `EPOCHS=1 LR=1e-6`.

## Notes

The old 100-case control set is useful only as a historical diagnostic. The
clean dev/test split should be used for project claims because it is split by
video `segment_id`.
