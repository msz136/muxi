# OPD Preflight Report - 2026-05-15

## Scope

This preflight intentionally did not run evaluation. It checked data shape,
cleaned the prompt pool, configured equal expert loss weights, and ran minimal
teacher/student sanity checks.

## AceBrain Data-Line Pattern Used

The local `C:\Users\mousongzhe\pryer` AceBrain line suggests the following
engineering pattern:

- keep a registry/config layer for source paths;
- normalize each sample into one canonical schema before training;
- check media presence and required fields before batching;
- filter or fallback around bad samples instead of letting one corrupt sample
  crash a run;
- print deterministic source/count statistics;
- keep original raw artifacts and write clean derived artifacts.

The OPD cleaner follows the same pattern for JSONL VLM prompts.

## Data Cleaning Outputs

Cleaner:

- `/data/msz/opd_project/scripts/clean_opd_data.py`

Clean artifacts:

- `/data/msz/opd_project/data/prompt_pool_clean.jsonl`
- `/data/msz/opd_project/data/eval_robopoint_500_clean.jsonl`
- `/data/msz/opd_project/data/cleaning_stats.json`
- `/data/msz/opd_project/data/cleaning_report.md`

Results:

- `prompt_pool.jsonl`: `37,780` rows -> `32,857` clean rows
- dropped `4,923` rows for missing local images
- removed `27,780` old-format RoboPoint conflicts
- clean source counts:
  - `robopoint`: `27,780`
  - `general_vqa_replay`: `4,869`
  - `sharerobot_affordance`: `208`
- `eval_robopoint_500.jsonl`: `500` rows -> `500` clean rows

Post-clean validation:

- missing image count: `0`
- old `0..1` tuple/list-format conflict count: `0`
- bad `gt_points`: `0`
- schema errors: `0`

## OPD Config

Config:

- `/data/msz/opd_project/configs/opd_multiteacher_55.yaml`

Key choices:

- student: `/data/msz/models/8b_base`
- reference: `/data/msz/models/Qwen3-VL-8B-Instruct`
- teachers:
  - `expert3`: weight `0.5`
  - `expert4`: weight `0.5`
- reference KL coefficient: `0.05`
- expert KL coefficient: `1.0`
- evaluation disabled in this config
- prompt batch size: `8`
- group size: `4`
- response batch per update: `32`

## Runtime Environment Note

Direct Python model imports require MACA environment variables:

```bash
export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH
```

Without these, importing Transformers/Torch can fail inside Triton MACA backend
discovery.

## Teacher Forward Checks

Script:

- `/data/msz/opd_project/training/check_teacher_forward.py`

Commands used one clean prompt and `max_new_tokens=32`.

Results:

- `expert3`: forward/generation OK
- `expert4`: forward/generation OK

Observation:

- both experts generated a JSON / `point_2d` style response on the first
  prompt, not strict `<point>...</point>`.
- This is not a crash, but the future OPD loss should keep a small strict-format
  constraint and cleaned prompts should continue to request only point tags.

## 8b_base Smoke

Script:

- `/data/msz/opd_project/training/opd_smoke_test.py`

Result:

- model: `/data/msz/models/8b_base`
- data: `/data/msz/opd_project/data/prompt_pool_clean.jsonl`
- sample: `object_ref/1329876244-b55b742_cam02_obj35-base_cabinet-top_on`
- target: `<point>[[148,492],[202,481],[172,433],[234,483],[253,431],[214,415],[280,454]]</point>`
- loss: `1.663355`
- finite gradient tensors: `399`
- max absolute gradient: `2.765625e+00`
- status: forward/backward OK
