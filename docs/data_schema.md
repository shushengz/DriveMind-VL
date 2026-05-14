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

