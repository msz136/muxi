# Current Model Assets And Eval Summary

## Asset Inventory

| Asset | Type | Remote path | Status |
|---|---|---|---|
| `general_obj_expert_200k` | 200k domain expert | `/data/msz/models/seed0_general_obj_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | completed |
| `general_reasoning_expert_200k` | 200k domain expert | `/data/msz/models/seed0_general_reasoning_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | completed |
| `region_expert_200k` | 200k domain expert | `/data/msz/models/seed0_region_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | completed |
| `robopoint_expert_200k` | 200k domain expert | `/data/msz/models/seed0_robopoint_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | completed |
| `spatial_rel_expert_200k` | 200k domain expert | `/data/msz/models/seed0_spatial_rel_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | completed |
| `opd5000_ckpt3000` | OPD fused checkpoint | `/data/msz/models/opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts_maxnew64_prefix64_veto1to0s500_maxnorm5_20260529_154544/checkpoint-3000` | evaluated |
| `opd5000_ckpt4000` | OPD fused checkpoint | `/data/msz/models/opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts_maxnew64_prefix64_veto1to0s500_maxnorm5_20260529_154544/checkpoint-4000` | evaluated |
| `opd5000_ckpt5000` | OPD fused checkpoint | `/data/msz/models/opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts_maxnew64_prefix64_veto1to0s500_maxnorm5_20260529_154544/checkpoint-5000` | evaluated |

## Expert Training Metrics

These are training metrics from each expert's `trainer_state.json`, not raw-holdout eval metrics. The five 200k experts have not been independently evaluated on `raw_holdout_eval_v1_10k`.

| Expert | Global step | Train loss | Last logged loss | Last grad norm | Runtime sec | Samples/s |
|---|---:|---:|---:|---:|---:|---:|
| `general_obj_expert_200k` | 3125 | 0.6652 | 0.6139 | 2.2909 | 11522.4 | 8.679 |
| `general_reasoning_expert_200k` | 3125 | 0.2321 | 0.3260 | 2.5257 | 11553.2 | 8.656 |
| `region_expert_200k` | 3125 | 0.7448 | 0.6806 | 2.0177 | 11508.9 | 8.689 |
| `robopoint_expert_200k` | 3125 | 0.7246 | 0.6356 | 1.5123 | 11887.3 | 8.412 |
| `spatial_rel_expert_200k` | 3125 | 0.7092 | 0.7401 | 2.2947 | 11505.1 | 8.692 |

## OPD Checkpoint Eval, By Format

Eval set: `/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl`.

Box metrics are computed on the 6100 box rows, point metrics on the 1400 point rows, and text metrics on the 2500 text rows.

| Checkpoint | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | **0.4761** | **0.6156** | 0.5233 | **0.3380** | 126.1 | 0.7414 | 0.8629 | 0.8716 |
| 4000 | 0.4747 | 0.6113 | 0.5207 | 0.3374 | 126.4 | 0.7379 | 0.8650 | **0.8784** |
| 5000 | 0.4759 | 0.6123 | **0.5238** | 0.3370 | **125.7** | **0.7493** | **0.8729** | 0.8724 |

## OPD Checkpoint Eval, By Pool

For box pools the metric is IoU mean. For `grounding_point` the metric is Hit@50. For `keepalive_vqa` the metric is text exact.

| Pool | n | ckpt3000 | ckpt4000 | ckpt5000 | Best |
|---|---:|---:|---:|---:|---|
| `refcoco` IoU | 1100 | 0.7343 | 0.7337 | **0.7358** | 5000 |
| `flickr30k_entities` IoU | 900 | 0.7079 | 0.7097 | **0.7101** | 5000 |
| `semantic_nav_box` IoU | 800 | 0.2921 | 0.2894 | **0.2924** | 5000 |
| `visual_genome_object` IoU | 1100 | 0.3197 | 0.3172 | **0.3211** | 5000 |
| `visual_genome_region` IoU | 1100 | **0.3925** | 0.3915 | 0.3912 | 3000 |
| `visual_genome_relationship` IoU | 1100 | **0.4019** | 0.3987 | 0.3974 | 3000 |
| `grounding_point` Hit@50 | 1400 | 0.7414 | 0.7379 | **0.7493** | 5000 |
| `keepalive_vqa` Text exact | 2500 | 0.8716 | **0.8784** | 0.8724 | 4000 |

## Readout

For the OPD fused model, `checkpoint-3000` is strongest on overall box IoU and high-threshold box accuracy, while `checkpoint-5000` is strongest on Box Acc@0.5, center distance, and point grounding. If selecting one general-purpose checkpoint from this run, `checkpoint-5000` is the default; if selecting specifically for strict box IoU, `checkpoint-3000` remains the cleaner candidate.
