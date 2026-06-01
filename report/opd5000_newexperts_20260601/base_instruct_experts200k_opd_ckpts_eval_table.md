# Base, Instruct, 200k Experts, And OPD Checkpoint Eval Table

Eval set: `/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl`. Box metrics are on 6100 box rows, point metrics on 1400 point rows, text metrics on 2500 text rows.

| Model | Type | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `8bbase` | base | 0.3854 | 0.5234 | 0.4087 | 0.2098 | 153.5 | 0.0000 | 0.0000 | 0.0080 |
| `8binstruct` | base | 0.4134 | 0.5577 | 0.4331 | 0.2472 | 140.1 | 0.0000 | 0.0000 | 0.0076 |
| `general_reasoning_expert_200k` | expert | 0.4429 | 0.5795 | 0.4797 | 0.3028 | 139.2 | 0.7036 | 0.8429 | 0.8800 |
| `robopoint_expert_200k` | expert | 0.4491 | 0.5870 | 0.4861 | 0.3061 | 134.4 | 0.7964 | 0.8986 | 0.8540 |
| `general_obj_expert_200k` | expert | 0.4740 | 0.6064 | 0.5167 | 0.3441 | 127.8 | 0.7171 | 0.8479 | 0.8492 |
| `region_expert_200k` | expert | 0.4682 | 0.6072 | 0.5121 | 0.3302 | 129.1 | 0.6836 | 0.8279 | 0.8424 |
| `spatial_rel_expert_200k` | expert | 0.4729 | 0.6098 | 0.5149 | 0.3403 | 128.3 | 0.6893 | 0.8243 | 0.8388 |
| `opd5000_ckpt3000` | opd | 0.4761 | 0.6156 | 0.5233 | 0.3380 | 126.1 | 0.7414 | 0.8629 | 0.8716 |
| `opd5000_ckpt4000` | opd | 0.4747 | 0.6113 | 0.5207 | 0.3374 | 126.4 | 0.7379 | 0.8650 | 0.8784 |
| `opd5000_ckpt5000` | opd | 0.4759 | 0.6123 | 0.5238 | 0.3370 | 125.7 | 0.7493 | 0.8729 | 0.8724 |

## Notes

- `8bbase` and `8binstruct` use existing eval from `/data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552`.
- The 5 experts use the new eval from `/data/msz/point/eval_raw_holdout_v1/experts200k_5models_20260601_141834`.
- OPD checkpoints use `/data/msz/point/eval_raw_holdout_v1/opd5000_newexperts_ckpts_1000_2000_3000_4000_5000_retry_20260601_104908`.
- The best general-purpose model in this table is still `opd5000_ckpt5000`; `opd5000_ckpt3000` remains slightly better on strict Box IoU / Acc@0.75.
