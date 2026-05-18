# LingoQA SFT Candidate Curation Report

## Summary

```json
{
  "count": 100,
  "by_candidate_type": {
    "all_settings_failed": 30,
    "positive_grounding": 27,
    "wrong_image_confound": 23,
    "hard_negative_spatial_reasoning": 11,
    "language_prior_confound": 8,
    "mixed_review": 1
  },
  "by_capability": {
    "counting": 21,
    "spatial_localization": 20,
    "object_recognition": 20,
    "other": 20,
    "reasoning_world_knowledge": 17,
    "weather_road_condition": 2
  },
  "candidate_type_by_capability": {
    "all_settings_failed": {
      "counting": 12,
      "other": 5,
      "reasoning_world_knowledge": 5,
      "object_recognition": 3,
      "spatial_localization": 3,
      "weather_road_condition": 2
    },
    "hard_negative_spatial_reasoning": {
      "spatial_localization": 6,
      "reasoning_world_knowledge": 5
    },
    "language_prior_confound": {
      "other": 4,
      "object_recognition": 3,
      "spatial_localization": 1
    },
    "mixed_review": {
      "spatial_localization": 1
    },
    "positive_grounding": {
      "object_recognition": 8,
      "counting": 6,
      "reasoning_world_knowledge": 6,
      "spatial_localization": 5,
      "other": 2
    },
    "wrong_image_confound": {
      "other": 9,
      "object_recognition": 6,
      "spatial_localization": 4,
      "counting": 3,
      "reasoning_world_knowledge": 1
    }
  }
}
```

## Review Policy

- `positive_grounding`: keep as evidence-style SFT examples after checking the visual answer.
- `hard_negative_spatial_reasoning`: highest priority for manual correction; these target the current main weakness.
- `wrong_image_confound` / `language_prior_confound` / `blank_image_confound`: use to design contrastive prompts or reject unsupported visual claims.
- `all_settings_failed`: inspect for ambiguous reference, low image quality, or genuinely difficult samples before training.

## Top Review Examples

| id | type | capability | gap | normal_f1 | question | gold_answer |
|---|---|---|---:|---:|---|---|
| lingoqa_eval_000031_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | reasoning_world_knowledge | -1.0 | 0.0 | Are there any specific vehicles to which you should yield in this situation? If yes, why? | No. |
| lingoqa_eval_000026_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | reasoning_world_knowledge | -0.5926 | 0.0 | Is it possible for you to accelerate in this situation, and if so, why? | It is not possible to accelerate as there is a roundabout after the pedestrian crossing. |
| lingoqa_eval_000072_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.56 | 0.0 | Is it possible for you to accelerate in this situation, and if so, why? | It is not possible to accelerate as there are pedestrians crossing the road. |
| lingoqa_eval_000036_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | reasoning_world_knowledge | -0.3333 | 0.3333 | Why can you proceed at this intersection? | The traffic light is green and no pedestrian is crossing. |
| lingoqa_eval_000032_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.2784 | 0.1 | Should you consider reducing your speed at this point? If yes, what are the reasons behind it? | Yes, to allow plenty of time for the cyclist to pass and the pedestrian to finishing crossing the street. |
| lingoqa_eval_000024_language_prior_confound | language_prior_confound | object_recognition | -1.0 | 0.0 | Are there any traffic lights present? If yes, what color are they displaying? | No. |
| lingoqa_eval_000060_language_prior_confound | language_prior_confound | other | -1.0 | 0.0 | Do you need to come to a stop? If so, what is the reason for it? | No. |
| lingoqa_eval_000081_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | reasoning_world_knowledge | -0.2143 | 0.2857 | Why are you stopping in this situation? | Light has turned red. |
| lingoqa_eval_000011_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.1579 | 0.0 | What is the current action and its justification? Answer in the form "action, justification". | The car accelerates slightly, follows the car ahead and crosses the zebra crossing, as no pedestrian is about to cross and the car ahead pulls away. |
| lingoqa_eval_000066_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | reasoning_world_knowledge | -0.1429 | 0.0 | Is it safe for you to move forward? Why or why not? | Yes with caution, pedestrians may not be crossing at the zebra crossing. |
| lingoqa_eval_000077_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.0794 | 0.45 | What factors influence the adjustment of your speed, and why is it important to do so? | I must adjust my speed according to the distance with the car in front and the speed limits. |
| lingoqa_eval_000037_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.0764 | 0.1379 | What is the current action and its justification? Answer in the form "action, justification". | The car is clearing the intersection while moving to the right lane, in order to overtake the cyclist on the left. |
| lingoqa_eval_000052_hard_negative_spatial_reasoning | hard_negative_spatial_reasoning | spatial_localization | -0.0296 | 0.2667 | What is the current action and its justification? Answer in the form "action, justification". | The car starts and accelerates, because the vehicles in front are pulling away at the junction due to the green traffic light. |
| lingoqa_eval_000075_wrong_image_confound | wrong_image_confound | other | -0.3087 | 0.3 | How is the car impacting your driving? | The car is not impacting my driving as I need to stop for the red light. |
| lingoqa_eval_000078_wrong_image_confound | wrong_image_confound | counting | -0.2857 | 0.0 | How many parked vehicles can you see on this street? | Zero parked vehicles, but many stationary. |
| lingoqa_eval_000044_wrong_image_confound | wrong_image_confound | object_recognition | -0.1667 | 0.3333 | Are there any traffic lights? What color are they showing? | No traffic lights. |

## Next Step

Manually review the CSV, fill `human_action` with `keep`, `fix`, or `drop`, then build a small SFT set from approved rows.
