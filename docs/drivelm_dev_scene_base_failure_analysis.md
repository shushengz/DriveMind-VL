# DriveLM Scene drivelm_dev_scene_base Visual-Control Failure Analysis

## Summary

```json
{
  "count": 300,
  "mean_normal_f1": 0.2615,
  "mean_text_only_f1": 0.2689,
  "mean_wrong_image_f1": 0.2622,
  "mean_blank_image_f1": 0.2584,
  "mean_visual_dependency_gap": -0.0924,
  "failure_type_counts": {
    "wrong_image_invariant_low": 137,
    "wrong_image_invariant": 62,
    "blank_beats_normal": 39,
    "wrong_image_beats_normal": 22,
    "acceptable_or_mixed": 20,
    "text_beats_normal": 11,
    "normal_low": 9
  },
  "failure_type_by_capability": {
    "object_recognition": {
      "wrong_image_invariant_low": 8,
      "text_beats_normal": 2,
      "blank_beats_normal": 1
    },
    "reasoning_world_knowledge": {
      "wrong_image_invariant_low": 85,
      "blank_beats_normal": 25,
      "wrong_image_invariant": 23,
      "acceptable_or_mixed": 6,
      "text_beats_normal": 5,
      "normal_low": 3,
      "wrong_image_beats_normal": 3
    },
    "scene_completeness": {
      "wrong_image_invariant": 34,
      "acceptable_or_mixed": 13,
      "wrong_image_beats_normal": 13,
      "wrong_image_invariant_low": 9,
      "blank_beats_normal": 9,
      "normal_low": 1,
      "text_beats_normal": 1
    },
    "spatial_localization": {
      "wrong_image_invariant_low": 35,
      "wrong_image_beats_normal": 6,
      "normal_low": 5,
      "wrong_image_invariant": 5,
      "blank_beats_normal": 4,
      "text_beats_normal": 3,
      "acceptable_or_mixed": 1
    }
  },
  "failure_type_by_category": {
    "behavior": {
      "blank_beats_normal": 2,
      "wrong_image_invariant_low": 1
    },
    "perception": {
      "wrong_image_invariant": 39,
      "wrong_image_invariant_low": 38,
      "wrong_image_beats_normal": 17,
      "acceptable_or_mixed": 13,
      "blank_beats_normal": 13,
      "text_beats_normal": 4,
      "normal_low": 3
    },
    "planning": {
      "wrong_image_invariant_low": 31,
      "blank_beats_normal": 14,
      "wrong_image_invariant": 14,
      "text_beats_normal": 6,
      "normal_low": 4,
      "wrong_image_beats_normal": 3,
      "acceptable_or_mixed": 3
    },
    "prediction": {
      "wrong_image_invariant_low": 67,
      "blank_beats_normal": 10,
      "wrong_image_invariant": 9,
      "acceptable_or_mixed": 4,
      "wrong_image_beats_normal": 2,
      "normal_low": 2,
      "text_beats_normal": 1
    }
  }
}
```

## Worst Visual Dependency Gaps

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000013 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000028 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000037 | wrong_image_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | yes |
| drivelm_dev_scene_000044 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000045 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000058 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000061 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Stationary. | white commercial vehicle | stationary |
| drivelm_dev_scene_000080 | wrong_image_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Stationary. | silver sedan | white bus |
| drivelm_dev_scene_000085 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000091 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000093 | wrong_image_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | yes | yes |
| drivelm_dev_scene_000094 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000101 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000121 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000127 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | White bus. | black car. | white bus |

## Blank Image Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000004 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.2456 | -0.2456 | The action is to keep going at the same speed. The reason is that there is no safety issue. The probability of this acti | stop | The ego vehicle should slow down and prepare to stop as it approaches the black pickup truck. The probability of an acci |
| drivelm_dev_scene_000013 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000023 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0952 | 0.2105 | -0.1153 | The action is none, the reason is there is no safety issue. | stop and wait for the construction barriers to clear | stop the vehicle and check the surroundings |
| drivelm_dev_scene_000028 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000030 | blank_beats_normal | spatial_localization | perception | 0.0506 | 0.1707 | -0.1201 | There is a black SUV to the front left of the ego vehicle, a pedestrian to the front of the ego vehicle, a white sedan t | the road is wet | black suv, white sedan, white van, pedestrian |
| drivelm_dev_scene_000035 | blank_beats_normal | reasoning_world_knowledge | behavior | 0.1818 | 0.2222 | -0.0404 | The ego vehicle is going straight. The ego vehicle is driving fast. | wait for the pedestrian to cross the road before proceeding. | yield to the white commercial vehicle |
| drivelm_dev_scene_000044 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000045 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000046 | blank_beats_normal | spatial_localization | perception | 0.0 | 0.1818 | -0.1818 | There is one car behind the ego car. | green bus | white sedan car. |
| drivelm_dev_scene_000058 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000061 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Stationary. | white commercial vehicle | stationary |
| drivelm_dev_scene_000085 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_dev_scene_000091 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000094 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_dev_scene_000097 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.08 | -0.08 | No such action will lead to a collision. | stop | The ego vehicle should slow down and prepare to stop if it is approaching the white SUV. |

## Text Only Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000029 | text_beats_normal | spatial_localization | perception | 0.1346 | 0.2679 | -0.1333 | There is a gray SUV to the front left of the ego vehicle, a blue sedan to the front right of the ego vehicle, a pedestri | There is a pedestrian crossing the road. | pedestrians crossing the road |
| drivelm_dev_scene_000051 | text_beats_normal | reasoning_world_knowledge | planning | 0.1 | 0.1667 | -0.0667 | Accelerating and going straight. | The ego vehicle should slow down and wait for the construction site to clear before proceeding. | The ego vehicle should slow down or stop to avoid a collision with the black SUV approaching from the front. |
| drivelm_dev_scene_000070 | text_beats_normal | spatial_localization | planning | 0.1429 | 0.3448 | -0.2019 | The action is to turn left. The reason is to avoid a collision, which is high probability. | stop and wait for the traffic light to change before proceeding. | The ego vehicle should consider turning right to avoid the black sedan. |
| drivelm_dev_scene_000133 | text_beats_normal | object_recognition | perception | 0.0 | 0.0984 | -0.0984 | There are many traffic elements in the front view. The information of these traffic elements is [(traffic light, red, 91 | traffic_light | red light |
| drivelm_dev_scene_000135 | text_beats_normal | spatial_localization | planning | 0.1667 | 0.1905 | -0.0238 | Changing to the left lane. | The ego vehicle should slow down and avoid the construction zone to prevent a collision with the black sedan. | The ego vehicle should slow down and maintain a safe distance from the black sedan to avoid a collision. |
| drivelm_dev_scene_000160 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | high |
| drivelm_dev_scene_000169 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | high |
| drivelm_dev_scene_000173 | text_beats_normal | object_recognition | perception | 0.0 | 0.0612 | -0.0612 | There are many traffic elements in the front view. The information of these traffic elements is [(traffic light, unknown | traffic_element | gray roadblock, black sedan, gray suv |
| drivelm_dev_scene_000204 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Stationary. | front_right | white motorhome |
| drivelm_dev_scene_000283 | text_beats_normal | reasoning_world_knowledge | planning | 0.1818 | 0.2667 | -0.0849 | The action is to remain stationary. The reason for this action is to follow the traffic rules, with a high probability. | The ego vehicle should proceed forward as the traffic light turns green. | The ego vehicle should accelerate and continue straight ahead. |
| drivelm_dev_scene_000296 | text_beats_normal | scene_completeness | perception | 0.5 | 0.75 | -0.25 | One bus is moving. | the bus is stationary | moving |

## Wrong Image Invariant And Low

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000001 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | The ego vehicle. | <c3,CAM_FRONT,1325.8,512.5> | <c3,CAM_FRONT,1325.8,512.5> |
| drivelm_dev_scene_000002 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.1333 | 0.1333 | 0.0 | Keep going at the same speed. | stop and wait for the traffic light to change | wait for the leading vehicle to pass before proceeding. |
| drivelm_dev_scene_000003 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None. | <c4,CAM_FRONT,1080.0,465.8> | white bus |
| drivelm_dev_scene_000009 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | no | no |
| drivelm_dev_scene_000011 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | The ego vehicle. | <c3,CAM_FRONT_LEFT,548.2,618.6> | traffic element |
| drivelm_dev_scene_000012 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are two barriers and one car to the back left of the ego car. | orange roadblocks | orange roadblock |
| drivelm_dev_scene_000014 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | The ego vehicle. | yellow truck | white truck |
| drivelm_dev_scene_000015 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | traffic sign | traffic sign |
| drivelm_dev_scene_000016 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Brake suddenly. | The ego vehicle should slow down and prepare to stop at the upcoming intersection. | The ego vehicle should slow down or stop to avoid a collision with the black SUV approaching from behind. |
| drivelm_dev_scene_000017 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are many cars, pedestrians, three bicycles, and two trucks behind the ego car. | black sedan. | black sedan, white truck |
| drivelm_dev_scene_000021 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | moving | moving |
| drivelm_dev_scene_000022 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Decelerate gradually without braking. | wait at the traffic light | stop |
| drivelm_dev_scene_000025 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | Please wait. | red light | red light |
| drivelm_dev_scene_000026 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Low. | high | high |
| drivelm_dev_scene_000033 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | moving | moving |

## Interpretation

- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.
- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.
- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.
