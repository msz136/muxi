# OPD v1 Evaluation Summary - 2026-05-20

## Model Set

| name | path |
| --- | --- |
| base | `/data/msz/models/8b_base` |
| obj_expert | `/data/msz/models/expert_obj_v1` |
| reg_expert | `/data/msz/models/expert_reg_v1` |
| opd_v1 | `/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529` |

## General Heldout Eval

Eval set: `/data/msz/point/data_eval/general_robo2vlm_heldout200_seed20260520.jsonl`

Notes:
- 200 heldout `robo2vlm-1` general VQA samples.
- Excluded media used by `/data/msz/point/data_opd/opd_mix_v1_2048_mediaok_seed20260520.jsonl`.
- An initial mixed attempt with `embspatial` was rejected because its remote image URLs returned HTTP 403 during model-side image loading.

| model_name | rc | num_samples | normalized_exact | relaxed_match | mean_token_f1 | option_samples |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 200 | 0.000000 | 0.295000 | 0.000000 | 200 |
| obj_expert | 0 | 200 | 0.045000 | 0.245000 | 0.071667 | 200 |
| reg_expert | 0 | 200 | 0.065000 | 0.310000 | 0.159091 | 200 |
| opd_v1 | 0 | 200 | 0.055000 | 0.295000 | 0.135000 | 200 |

Local detailed outputs: `report/eval_general_vqa_heldout200_20260520_151919/`

## Domain Region/Object Eval

Eval outputs: `report/eval_domain_four_models_v1_20260520_152926/`

| split | model_name | rc | num_samples | format_ok | format_rate | mean_iou_parseable | mean_iou_all | iou_at_0_3_all | iou_at_0_5_all | mean_center_error | mean_coord_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_object | base | 0 | 1251 | 196 | 0.156675 | 0.125033 | 0.019590 | 0.021583 | 0.009592 | 215.999777 | 176.341837 |
| latest_object | obj_expert | 0 | 1251 | 1251 | 1.000000 | 0.398283 | 0.398283 | 0.547562 | 0.474820 | 110.719908 | 71.961431 |
| latest_object | reg_expert | 0 | 1251 | 1251 | 1.000000 | 0.104216 | 0.104216 | 0.110312 | 0.031175 | 163.966553 | 108.932254 |
| latest_object | opd_v1 | 0 | 1251 | 1251 | 1.000000 | 0.173187 | 0.173187 | 0.227018 | 0.152678 | 197.358188 | 137.532774 |
| latest_region | base | 0 | 144 | 42 | 0.291667 | 0.093813 | 0.027362 | 0.020833 | 0.000000 | 163.678191 | 118.410714 |
| latest_region | obj_expert | 0 | 144 | 144 | 1.000000 | 0.111445 | 0.111445 | 0.111111 | 0.027778 | 137.265902 | 92.451389 |
| latest_region | reg_expert | 0 | 144 | 144 | 1.000000 | 0.236819 | 0.236819 | 0.347222 | 0.173611 | 96.033732 | 61.828125 |
| latest_region | opd_v1 | 0 | 144 | 144 | 1.000000 | 0.145121 | 0.145121 | 0.208333 | 0.090278 | 137.645307 | 92.493056 |

## OPD v1 Grounding Eval

| split | model_name | num_samples | format_ok | format_rate | mean_iou_all | iou_at_0_3_all | iou_at_0_5_all | mean_center_error | mean_coord_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_object | opd_online_mix_v1_2048 | 1251 | 1251 | 1.000000 | 0.173187 | 0.227018 | 0.152678 | 197.358188 | 137.532774 |
| latest_region | opd_online_mix_v1_2048 | 144 | 144 | 1.000000 | 0.145121 | 0.208333 | 0.090278 | 137.645307 | 92.493056 |

Local detailed outputs: `report/eval_opd_online_mix_v1_2048_20260520_145224/`

## Previous Latest Region/Object Matrix

The earlier base/expert latest region/object comparison outputs are copied to:

`report/eval_latest_region_object_matrix_20260520_104104/`

Key `mean_iou_all` values:

| split | base | object_expert | region_predbox | region_solution | opd_v1 |
| --- | --- | --- | --- | --- | --- |
| latest_object | 0.041636 | 0.398283 | 0.133069 | 0.104216 | 0.173187 |
| latest_region | 0.048248 | 0.111445 | 0.216611 | 0.236819 | 0.145121 |
