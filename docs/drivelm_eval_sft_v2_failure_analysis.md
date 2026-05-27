# DriveLM drivelm_eval_sft_v2 Visual-Control Failure Analysis

## Summary

```json
{
  "count": 300,
  "mean_normal_f1": 0.2871,
  "mean_text_only_f1": 0.3052,
  "mean_wrong_image_f1": 0.287,
  "mean_blank_image_f1": 0.3202,
  "mean_visual_dependency_gap": -0.1067,
  "failure_type_counts": {
    "wrong_image_invariant_low": 113,
    "wrong_image_invariant": 95,
    "blank_beats_normal": 65,
    "text_beats_normal": 25,
    "acceptable_or_mixed": 2
  },
  "failure_type_by_capability": {
    "object_recognition": {
      "wrong_image_invariant_low": 9,
      "blank_beats_normal": 5,
      "text_beats_normal": 4,
      "wrong_image_invariant": 2
    },
    "reasoning_world_knowledge": {
      "wrong_image_invariant_low": 58,
      "wrong_image_invariant": 38,
      "blank_beats_normal": 22,
      "text_beats_normal": 10,
      "acceptable_or_mixed": 2
    },
    "scene_completeness": {
      "wrong_image_invariant": 44,
      "wrong_image_invariant_low": 27,
      "blank_beats_normal": 22,
      "text_beats_normal": 2
    },
    "spatial_localization": {
      "wrong_image_invariant_low": 18,
      "blank_beats_normal": 16,
      "wrong_image_invariant": 11,
      "text_beats_normal": 9
    },
    "weather_road_condition": {
      "wrong_image_invariant_low": 1
    }
  },
  "failure_type_by_category": {
    "behavior": {
      "acceptable_or_mixed": 2,
      "text_beats_normal": 1
    },
    "perception": {
      "wrong_image_invariant": 55,
      "wrong_image_invariant_low": 41,
      "blank_beats_normal": 36,
      "text_beats_normal": 11
    },
    "planning": {
      "wrong_image_invariant_low": 22,
      "blank_beats_normal": 15,
      "wrong_image_invariant": 11,
      "text_beats_normal": 7
    },
    "prediction": {
      "wrong_image_invariant_low": 50,
      "wrong_image_invariant": 29,
      "blank_beats_normal": 14,
      "text_beats_normal": 6
    }
  }
}
```

## Worst Visual Dependency Gaps

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000040 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_eval_000057 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_eval_000070 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is green. | Yes. |
| drivelm_eval_000101 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000114 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_eval_000143 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is red. | Yes. |
| drivelm_eval_000171 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000186 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_eval_000190 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_eval_000207 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000220 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000248 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000249 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000251 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000254 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |

## Blank Image Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000004 | blank_beats_normal | spatial_localization | perception | 0.087 | 0.6452 | -0.5582 | There are two barriers, many trucks, two trailers, and one car to the back left of the ego car. | There is no traffic. | There are no objects to the back left of the ego car. |
| drivelm_eval_000009 | blank_beats_normal | scene_completeness | perception | 0.4348 | 0.7407 | -0.3059 | There are two cars, one truck, and one barrier to the back of the ego car. | There are no objects to the back. | There are two vehicles to the back of the ego car. |
| drivelm_eval_000010 | blank_beats_normal | scene_completeness | perception | 0.1333 | 0.2222 | -0.0889 | Two cars are moving. | The black sedan is moving and the brown SUV is stationary. | The black sedan is moving. |
| drivelm_eval_000025 | blank_beats_normal | spatial_localization | perception | 0.0 | 0.1818 | -0.1818 | Yes, there are some traffic elements in the front view. | No. | Yes. |
| drivelm_eval_000029 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.3333 | -0.3333 | Moving. | The traffic light is green. | The brown SUV is moving. |
| drivelm_eval_000033 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.8 | -0.8 | Brown SUV. | There is no car in front. | A brown SUV. |
| drivelm_eval_000034 | blank_beats_normal | scene_completeness | perception | 0.4444 | 0.8 | -0.3556 | Black sedan. | The black sedan is approaching the intersection. | A black sedan. |
| drivelm_eval_000035 | blank_beats_normal | scene_completeness | perception | 0.5 | 0.6667 | -0.1667 | Green light. | There is a green traffic light. | A green traffic light. |
| drivelm_eval_000040 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_eval_000061 | blank_beats_normal | object_recognition | prediction | 0.0 | 0.5 | -0.5 | Traffic light. | A railroad crossing. | green light. |
| drivelm_eval_000063 | blank_beats_normal | reasoning_world_knowledge | planning | 0.2564 | 0.4815 | -0.2251 | The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability. | The ego vehicle should accelerate and proceed through the intersection as the traffic light is green. | The ego vehicle should accelerate to overtake the black sedan. The probability of this action is high as the black sedan |
| drivelm_eval_000065 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.087 | -0.087 | No such action will lead to a collision. | The ego vehicle should slow down and stop at the intersection. | The ego vehicle should slow down and stop if it is not safe to proceed. |
| drivelm_eval_000067 | blank_beats_normal | reasoning_world_knowledge | planning | 0.2564 | 0.4255 | -0.1691 | The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability. | The ego vehicle should accelerate and proceed through the intersection as the traffic light is green. | The ego vehicle should accelerate and merge onto the right lane. The probability of this action is high as the traffic l |
| drivelm_eval_000070 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is green. | Yes. |
| drivelm_eval_000075 | blank_beats_normal | spatial_localization | planning | 0.0 | 0.1429 | -0.1429 | Back up, turn right. | The ego vehicle should slow down and stop at the intersection as it is raining. | The ego vehicle should not turn left at the intersection. |

## Text Only Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000001 | text_beats_normal | spatial_localization | perception | 0.1 | 0.4 | -0.3 | There are many barriers and one construction vehicle to the front right of the ego car. | There is no traffic. | There is no traffic. |
| drivelm_eval_000002 | text_beats_normal | spatial_localization | perception | 0.2105 | 0.5 | -0.2895 | There is one truck and one barrier to the front left of the ego car. | There is no traffic. | There is no traffic. |
| drivelm_eval_000013 | text_beats_normal | spatial_localization | perception | 0.6 | 0.9167 | -0.3167 | There are two trailers to the back right of the ego car. | There are no objects to the back right. | There are no objects to the back right. |
| drivelm_eval_000024 | text_beats_normal | scene_completeness | perception | 0.0714 | 0.2222 | -0.1508 | <c2,CAM_BACK,864.2,468.3> is at the back of <c1,CAM_BACK,1088.3,497.5>. | The traffic lights are green, so I can proceed straight ahead. | The ego vehicle is behind the black sedan. |
| drivelm_eval_000026 | text_beats_normal | object_recognition | perception | 0.186 | 0.2273 | -0.0413 | There are two traffic elements in the front view. The information of these traffic elements is [(traffic light, green, 6 | There are no traffic elements. | There is no traffic element. |
| drivelm_eval_000052 | text_beats_normal | spatial_localization | prediction | 0.2069 | 0.2759 | -0.069 | The action is to turn left, the reason is there is no safety issue. | The driver should slow down and prepare to stop as the traffic light turns red. | The driver should slow down and prepare to stop as the traffic light turns green. |
| drivelm_eval_000057 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_eval_000068 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.1818 | -0.1818 | Low. | There is no risk of collision as the black sedan is moving away from the ego vehicle. | {"task":"external_vqa","answer":"The probability of colliding with <c2,CAM_BACK,864.2,468.3> after the ego vehicle accel |
| drivelm_eval_000135 | text_beats_normal | object_recognition | prediction | 0.0 | 0.5 | -0.5 | Traffic light. | A stoplight. | There is no traffic sign. |
| drivelm_eval_000146 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.5 | -0.5 | Go straight. | The ego vehicle will stop at the red light. | The ego vehicle should continue straight. |
| drivelm_eval_000151 | text_beats_normal | reasoning_world_knowledge | behavior | 0.381 | 0.4444 | -0.0634 | The ego vehicle is going straight. The ego vehicle is driving slowly. | The ego vehicle will stop at the red light. | The ego vehicle will stop at the red light. |
| drivelm_eval_000162 | text_beats_normal | spatial_localization | perception | 0.125 | 0.4762 | -0.3512 | There are two trailers to the front right of the ego car. | There is no traffic. | There is no traffic. |
| drivelm_eval_000175 | text_beats_normal | spatial_localization | perception | 0.1176 | 0.2143 | -0.0967 | <c2,CAM_BACK,966.7,580.0> is on the back right of <c1,CAM_FRONT_LEFT,231.5,472.1>. | The traffic lights are green, the car ahead is white, and the car behind is gray. | The ego vehicle is behind the white sedan. |
| drivelm_eval_000177 | text_beats_normal | object_recognition | perception | 0.08 | 0.1299 | -0.0499 | There are many traffic elements in the front view. The information of these traffic elements is [(traffic light, unknown | There is no traffic. | There is a green light. |
| drivelm_eval_000198 | text_beats_normal | object_recognition | prediction | 0.2919 | 0.3651 | -0.0732 | Firstly notice that <c3,CAM_FRONT,864.5,293.6>; the object is a traffic sign, so the ego vehicle should continue going a | {"task":"external_vqa","answer":"The ego vehicle should notice the traffic light first, which is green. The traffic ligh | {"task":"external_vqa","answer":"The ego vehicle should first notice the green light, which is green and the ego vehicle |

## Wrong Image Invariant And Low

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000015 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000018 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000019 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000020 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000021 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000022 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000023 | wrong_image_invariant_low | spatial_localization | perception | 0.2308 | 0.2308 | 0.0 | There are two moving cars behind the ego car and two barriers in front of it. | The traffic lights are green and the road is wet. | The ego vehicle is proceeding through the intersection. |
| drivelm_eval_000028 | wrong_image_invariant_low | spatial_localization | perception | 0.2222 | 0.2222 | 0.0 | There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to  | There is a green traffic light, a railroad crossing sign, and a white truck. | There is a green traffic light. |
| drivelm_eval_000030 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Turn left. | The car is moving. | The object is moving. |
| drivelm_eval_000032 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | The black sedan is moving. | The black sedan is moving. |
| drivelm_eval_000036 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | Yes. | It is a traffic sign. |
| drivelm_eval_000037 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None, no, none. | The black sedan. | The black sedan. |
| drivelm_eval_000039 | wrong_image_invariant_low | spatial_localization | prediction | 0.25 | 0.25 | 0.0 | Turn left. | The traffic light will turn red. | The brown SUV will continue to move. |
| drivelm_eval_000041 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None, no, none. | The black sedan. | The black sedan. |
| drivelm_eval_000043 | wrong_image_invariant_low | spatial_localization | prediction | 0.25 | 0.25 | 0.0 | Turn left. | The traffic light will turn red. | The black sedan will continue straight. |

## Interpretation

- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.
- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.
- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.
