# AGENTS.md -- /data/msz/opd_project

Qwen3-VL-8B OPD project for fusing the base/instruct model with pointing
experts through online distillation. The project should stay focused on OPD
and multi-teacher distillation.

## Scope

This directory owns the OPD design, prompt-pool preparation, teacher/student
launch scripts, and pointing evaluation for:

- Student/base candidates:
  - `/data/msz/models/Qwen3-VL-8B-Instruct`
  - `/data/msz/models/8b_base` once the safetensors weights finish deploying
- Expert teachers:
  - `/data/msz/models/expert3`
  - `/data/msz/models/expert4`
- Data and evaluation:
  - `/data/msz/opd_project/data/prompt_pool.jsonl`
  - `/data/msz/opd_project/data/eval_robopoint_500.jsonl`

## Current Findings

- `Qwen3-VL-8B-Instruct`, `expert3`, and `expert4` are complete 4-shard
  safetensors checkpoints with matching weight keys, so they are compatible
  teacher/student candidates.
- `/data/msz/models/8b_base` currently has config/tokenizer files but no model
  safetensors shards. Treat it as unavailable until the weights appear.
- The current runtime has `safetensors`, but `slime` and `sglang` were not
  present during the first inspection. Do not assume `training/run_opd.sh`
  works until those dependencies or replacements are installed.
- The existing prompt pool has useful data, but many RoboPoint prompts still
  ask for old `(x, y)` coordinates in the `0..1` range while the system prompt
  asks for `<point>[[x,y]]</point>` in the `0..1000` range. Rebuild or sanitize
  prompts before OPD training.
- `merge/merge_to_acebrain.py` is legacy/placeholder code. Do not use it as
  the fusion path.

## Project Direction

Use multi-teacher OPD as the primary fusion method:

1. The student generates responses on-policy from the prompt pool.
2. `expert3` and `expert4` score the student's sampled responses with
   token-level log probabilities.
3. The student is optimized toward the expert teacher distribution.
4. `Qwen3-VL-8B-Instruct` is also used as a reference/keepalive model so the
   fused model does not lose general VLM behavior.
5. General VQA replay is mixed in to further reduce forgetting.

## Required Data Cleanup

Before any OPD run, regenerate or sanitize `prompt_pool.jsonl`:

- Every pointing prompt should require exactly this output family:
  `<point>[[x1,y1],[x2,y2],...]</point>`.
- Coordinates must be normalized integers in `[0, 1000]`.
- Remove user text that says coordinates are between `0` and `1`.
- Remove user text that asks for "a list of tuples" unless it is rewritten to
  the `<point>` format.
- Preserve `gt_points` for evaluation items.
- Keep a replay split for general VQA, but do not force replay answers into
  pointing format.

Recommended pointing system prompt:

```text
You are a helpful vision-language assistant. When the user asks for a location,
answer with coordinates in the range 0 to 1000. Your answer must be formatted
as "<point>[[x1,y1],[x2,y2],...]</point>". Return only the point tag.
```

## Evaluation First

Before training, run the same evaluation script for:

- `Qwen3-VL-8B-Instruct`
- `expert3`
- `expert4`
- any later OPD checkpoint

Report at least:

- `acc@50_per_point`, `acc@100_per_point`, `acc@150_per_point`
- per-point and per-sample mean distance
- `precision@100`, `recall@100`, `f1@100`
- strict `<point>` format accuracy
- average predicted point count
- a small general VQA replay check

The current baseline snapshot for `Qwen3-VL-8B-Instruct` on
`eval_robopoint_500.jsonl` was approximately:

- per-point mean distance: `117.25`
- `acc@50_per_point`: `0.2806`
- `acc@100_per_point`: `0.5588`
- `acc@150_per_point`: `0.7384`

Use these numbers only as orientation; rerun after prompt cleanup.

## Teacher Aggregation

Start simple, then add complexity only if metrics justify it:

- Phase A: route each prompt to a single teacher if its source clearly matches
  one expert's strength.
- Phase B: ask both `expert3` and `expert4` for token log-probs and aggregate
  them with weighted logsumexp.
- Phase C: learn or tune per-source teacher weights from validation metrics.

Keep `Qwen3-VL-8B-Instruct` as a reference distribution. The OPD objective
should balance expert imitation against reference preservation.

## Training Recipe

Initial hyperparameters:

- student init: `/data/msz/models/Qwen3-VL-8B-Instruct`
- optional alternate student init: `/data/msz/models/8b_base` after weights land
- freeze vision tower: true
- train visual merger/MLP: true
- train LLM: true
- learning rate: `5e-7` to `1e-6`
- temperature: `0.6` to `0.7`
- top-p: `0.95`
- max new tokens: `128` or `256`
- group size: `4`
- general replay ratio: start at `0.10`
- reference KL: enabled
- expert KL: enabled
- format reward: optional, small, and only for strict `<point>` compliance

Do not rely on pure `reward=0` GRPO unless the implementation clearly confirms
that teacher token log-probs are actually used in the loss. A zero reward path
without a real teacher KL term will not distill the experts.

## Implementation Notes

- Fix all placeholder paths before running:
  - `SLIME_ROOT="/path/to/slime"` is not valid.
  - teacher model paths in scripts should point to `expert3` and/or `expert4`.
- If `sglang` is unavailable or incompatible on MACA, implement teacher
  log-prob extraction directly with Transformers as a slower but simpler
  fallback.
- Use `safetensors` APIs for safetensors files. Do not use `torch.load` on
  safetensors checkpoints.
- Avoid touching unrelated training code under `/data/msz/point` unless a
  change is explicitly needed for OPD.
- Keep generated outputs under ignored output/log/result directories.
- When changing scripts, preserve MACA-related environment workarounds from the
  parent project where distributed training is involved.

## Success Criteria

An OPD checkpoint is useful only if it improves pointing while preserving the
base model's general behavior:

- `acc@100_per_point` improves over both the Instruct baseline and the weaker
  expert on the cleaned evaluation set.
- strict `<point>` format accuracy is at least `95%`.
- average predicted point count is not wildly lower than the ground-truth count.
- general VQA replay does not show obvious collapse or formatting leakage.
- training logs show no persistent NaN/Inf loss, grad norm, or parameter issues.
