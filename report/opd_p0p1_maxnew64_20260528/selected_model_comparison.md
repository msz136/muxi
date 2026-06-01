| model | rows | box_iou | box_acc@0.3 | box_acc@0.5 | box_acc@0.75 | box_center_dist | point_hit@50 | point_hit@100 | text_exact | text_loose | sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 10000 | 0.3854 | 0.5234 | 0.4087 | 0.2098 | 153.5 | 0.0000 | 0.0000 | 0.0080 | 0.0080 | 1213.1 |
| 8b_instruct | 10000 | 0.4134 | 0.5577 | 0.4331 | 0.2472 | 140.1 | 0.0000 | 0.0000 | 0.0076 | 0.5200 | 1116.2 |
| offpolicy_3500steps | 10000 | 0.4681 | 0.6026 | 0.5138 | 0.3293 | 128.1 | 0.7193 | 0.8343 | 0.8732 | 0.8732 | 1228.9 |
| coldstart100_offpolicy | 10000 | 0.4283 | 0.5620 | 0.4621 | 0.2825 | 142.1 | 0.4014 | 0.5393 | 0.1032 | 0.1040 | 1259.6 |
| opd_v1_fullvocab_frombase_2500 | 10000 | 0.4704 | 0.6059 | 0.5139 | 0.3325 | 127.2 | 0.7214 | 0.8429 | 0.8744 | 0.8744 | 1246.2 |
| opd_v2_p0p1_maxnew64_2500 | 10000 | 0.4723 | 0.6074 | 0.5157 | 0.3359 | 126.2 | 0.7143 | 0.8479 | 0.8744 | 0.8744 | 1262.9 |

## Model paths
- base: /data/msz/models/8b_base
- 8b_instruct: /data/msz/models/Qwen3-VL-8B-Instruct
- offpolicy_3500steps: /data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/checkpoint-3500
- coldstart100_offpolicy: /data/msz/models/opd_offpolicy_coldstart100_p0p1_fullvocab_maxnew128_prefix64_veto1to0_zero3_mb16_accum1_20260527_132800
- opd_v1_fullvocab_frombase_2500: /data/msz/models/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526/checkpoint-2500
- opd_v2_p0p1_maxnew64_2500: /data/msz/models/opd_p0p1_studentrollout_full2500_save500_zero3_mb16_accum1_from_coldstart100_maxnew64_prefix64_veto1to0s500_maxnorm5_20260527_174410

## Metrics files
- base: /data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552/base/metrics.json
- 8b_instruct: /data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552/qwen3_vl_8b_instruct/metrics.json
- offpolicy_3500steps: /data/msz/point/eval_raw_holdout_v1/opd_ckpts_2500_3000_3500_cut3501_20260526_123449/opd_ckpt_3500/metrics.json
- coldstart100_offpolicy: /data/msz/point/eval_raw_holdout_v1/offpolicy_coldstart100_20260528_140217/offpolicy_coldstart100/metrics.json
- opd_v1_fullvocab_frombase_2500: /data/msz/point/eval_raw_holdout_v1/opd_fullvocab_studentrollout_full2500_maxnorm5_ckpts_1500_2000_2500_20260527_104507/opd_ckpt_2500/metrics.json
- opd_v2_p0p1_maxnew64_2500: /data/msz/point/eval_raw_holdout_v1/opd_p0p1_maxnew64_final2500_20260528_133416/opd_p0p1_maxnew64_final2500/metrics.json
