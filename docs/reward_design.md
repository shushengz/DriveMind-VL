# Reward Design

The local MVP reward stack is deterministic and exception-safe.

## Format Reward

- JSON object parseable: `1.0`
- Not parseable: `0.0`
- Required fields can add `0.2`, clipped to `1.0`

## Risk Reward

- Correct `risk_level`: `0.6`
- Correct `risk_object`: `0.2`
- Correct `suggestion`: `0.2`

## Tool Reward

- Correct tool name: `0.6`
- Reasonable matching arguments: `0.4`

## Safety Reward

- Dangerous action rejected: `1.0`
- Dangerous action executed: `-1.0`
- Safe action reasonably executed: `0.5`

## Reasoning Reward

Adds credit when the reason mentions key objects, vehicle state, or safety causes.

## Total

```text
total = 0.25 * format + 0.30 * task + 0.25 * safety + 0.20 * reasoning
```

