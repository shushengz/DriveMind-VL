# DriveMind-Instruct Data Schema

Each JSONL row contains:

- `id`: unique sample id.
- `image`: relative path to a front-view or cabin image.
- `vehicle_state`: structured vehicle state such as speed, weather, front-car distance, yaw rate, gear, and time.
- `perception`: structured perception result with objects, scene, and risk hint.
- `instruction`: user or system task instruction.
- `answer`: JSON object target output.
- `meta`: metadata including `task_type`, source, and difficulty.

Supported `meta.task_type` values:

- `risk_reasoning`
- `tool_call`
- `safety_rejection`
- `cabin_understanding`
- `personalized_service`

`answer` must be a JSON object, not free-form text.

## v2 Metadata Extension

For external benchmark data, keep the same top-level schema and add provenance under `meta`:

- `source`: `synthetic_seed`, `external_benchmark`, or `human_reviewed`.
- `benchmark_source`: examples include `intelli_cockpit_bench`, `nuscenes_qa`, `drivelm`, `drivebench`, `drive_and_act`, and `dmd`.
- `external.original_id`: original sample id from the upstream dataset.
- `external.split`: upstream or local split.
- `external.category`: upstream task/category label.
- `review_status`: optional status such as `unreviewed`, `auto_checked`, or `human_checked`.

This lets DriveMind-Instruct v2 report metrics by source instead of mixing synthetic and public samples into one opaque score.
