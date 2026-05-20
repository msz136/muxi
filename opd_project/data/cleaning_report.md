# OPD Data Cleaning Report

Cleaning policy:
- normalize all pointing prompts to strict `<point>[[x,y],...]</point>` output;
- use integer coordinates in `[0, 1000]`;
- remove old `(x, y)` / `0..1` tuple instructions;
- filter samples whose local absolute image files are missing;
- keep general replay prompts out of pointing format forcing.

## prompt_pool_clean.jsonl

- input rows: `37780`
- output rows: `32857`
- old-format conflicts before cleaning: `27780`
- dropped: `{"missing_image": 4923}`
- source before: `{"robopoint": 27780, "sharerobot_affordance": 5000, "general_vqa_replay": 5000}`
- source after: `{"robopoint": 27780, "general_vqa_replay": 4869, "sharerobot_affordance": 208}`

## eval_robopoint_500_clean.jsonl

- input rows: `500`
- output rows: `500`
- old-format conflicts before cleaning: `500`
- dropped: `{}`
- source before: `{"robopoint": 500}`
- source after: `{"robopoint": 500}`
