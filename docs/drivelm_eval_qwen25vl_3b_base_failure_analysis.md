# DriveLM drivelm_eval_qwen25vl_3b_base Visual-Control Failure Analysis

## Summary

```json
{
  "count": 300,
  "mean_normal_f1": 0.291,
  "mean_text_only_f1": 0.2826,
  "mean_wrong_image_f1": 0.2907,
  "mean_blank_image_f1": 0.282,
  "mean_visual_dependency_gap": -0.097,
  "failure_type_counts": {
    "wrong_image_invariant_low": 138,
    "wrong_image_invariant": 93,
    "blank_beats_normal": 57,
    "text_beats_normal": 12
  },
  "failure_type_by_capability": {
    "object_recognition": {
      "wrong_image_invariant_low": 11,
      "blank_beats_normal": 4,
      "wrong_image_invariant": 3,
      "text_beats_normal": 2
    },
    "reasoning_world_knowledge": {
      "wrong_image_invariant_low": 74,
      "blank_beats_normal": 28,
      "wrong_image_invariant": 23,
      "text_beats_normal": 5
    },
    "scene_completeness": {
      "wrong_image_invariant": 62,
      "wrong_image_invariant_low": 18,
      "blank_beats_normal": 13,
      "text_beats_normal": 2
    },
    "spatial_localization": {
      "wrong_image_invariant_low": 34,
      "blank_beats_normal": 12,
      "wrong_image_invariant": 5,
      "text_beats_normal": 3
    },
    "weather_road_condition": {
      "wrong_image_invariant_low": 1
    }
  },
  "failure_type_by_category": {
    "behavior": {
      "blank_beats_normal": 2,
      "text_beats_normal": 1
    },
    "perception": {
      "wrong_image_invariant": 68,
      "wrong_image_invariant_low": 46,
      "blank_beats_normal": 23,
      "text_beats_normal": 6
    },
    "planning": {
      "wrong_image_invariant_low": 26,
      "blank_beats_normal": 20,
      "wrong_image_invariant": 5,
      "text_beats_normal": 4
    },
    "prediction": {
      "wrong_image_invariant_low": 66,
      "wrong_image_invariant": 20,
      "blank_beats_normal": 12,
      "text_beats_normal": 1
    }
  }
}
```

## Worst Visual Dependency Gaps

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000051 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000064 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_eval_000093 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000128 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000139 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_eval_000171 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |
| drivelm_eval_000172 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |
| drivelm_eval_000173 | text_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | no |
| drivelm_eval_000201 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000214 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_eval_000218 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_eval_000248 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |
| drivelm_eval_000249 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |
| drivelm_eval_000251 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |
| drivelm_eval_000254 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | yes |

## Blank Image Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000023 | blank_beats_normal | spatial_localization | perception | 0.0909 | 0.1429 | -0.052 | There are two moving cars behind the ego car and two barriers in front of it. | the road is wet with rain | The ego vehicle is proceeding through an intersection with a green light. |
| drivelm_eval_000028 | blank_beats_normal | spatial_localization | perception | 0.0333 | 0.1875 | -0.1542 | There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to  | traffic light | brown suv, black sedan, green light |
| drivelm_eval_000035 | blank_beats_normal | scene_completeness | perception | 0.5714 | 1.0 | -0.4286 | Green light. | the traffic light is green. | green light |
| drivelm_eval_000048 | blank_beats_normal | object_recognition | prediction | 0.0227 | 0.1649 | -0.1422 | Firstly notice that <c3,CAM_FRONT,1043.2,82.2>. The object is a traffic sign, so the ego vehicle should keep going ahead | first: traffic light, second: traffic light, third: traffic light | The ego vehicle should first notice the green light, then the black sedan, and finally the brown SUV. |
| drivelm_eval_000051 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000061 | blank_beats_normal | object_recognition | prediction | 0.0 | 0.5 | -0.5 | Traffic light. | x | green light |
| drivelm_eval_000064 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | low |
| drivelm_eval_000065 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0952 | 0.25 | -0.1548 | No such action will lead to a collision. | The ego vehicle should slow down and prepare to stop at the intersection. | The ego vehicle should slow down or stop to avoid a collision with the brown SUV. |
| drivelm_eval_000067 | blank_beats_normal | reasoning_world_knowledge | planning | 0.1481 | 0.2222 | -0.0741 | The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability. | stop at the intersection | The ego vehicle should slow down and prepare to stop at the intersection. |
| drivelm_eval_000071 | blank_beats_normal | reasoning_world_knowledge | planning | 0.125 | 0.1818 | -0.0568 | Keep going at the same speed. | wait for the red light to turn green before proceeding. | wait for the green light |
| drivelm_eval_000072 | blank_beats_normal | reasoning_world_knowledge | planning | 0.2353 | 1.0 | -0.7647 | Green light. | the traffic signal that the ego vehicle should pay attention to is the green light. | green light |
| drivelm_eval_000077 | blank_beats_normal | spatial_localization | planning | 0.0 | 0.1667 | -0.1667 | Turn left. | stop | wait for the green light to turn red before proceeding. |
| drivelm_eval_000079 | blank_beats_normal | reasoning_world_knowledge | behavior | 0.0952 | 0.1333 | -0.0381 | The ego vehicle is going straight. The ego vehicle is driving fast. | wait for the traffic light to change before proceeding. | continue straight ahead |
| drivelm_eval_000093 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | yes | no |
| drivelm_eval_000108 | blank_beats_normal | spatial_localization | perception | 0.2 | 0.25 | -0.05 | <c1,CAM_BACK,991.7,603.0> is on the ego lane. | the sedan is in front of the traffic lights. | the traffic light is red. |

## Text Only Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000006 | text_beats_normal | spatial_localization | perception | 0.4444 | 0.6667 | -0.2223 | The construction vehicle to the front right of the ego car is parked. | the construction vehicle is moving | the traffic light is green |
| drivelm_eval_000024 | text_beats_normal | scene_completeness | perception | 0.1818 | 0.2222 | -0.0404 | <c2,CAM_BACK,864.2,468.3> is at the back of <c1,CAM_BACK,1088.3,497.5>. | the traffic light is green. | The ego vehicle is moving towards the intersection. |
| drivelm_eval_000026 | text_beats_normal | object_recognition | perception | 0.0 | 0.1282 | -0.1282 | There are two traffic elements in the front view. The information of these traffic elements is [(traffic light, green, 6 | traffic_element | traffic_element |
| drivelm_eval_000073 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.4 | -0.4 | Go straight. | proceed through the intersection | proceed through the intersection |
| drivelm_eval_000148 | text_beats_normal | spatial_localization | planning | 0.0769 | 0.2286 | -0.1517 | Backing up, turning left, turning right, and changing to the right lane are dangerous actions for the ego vehicle in thi | stop at the intersection | stop at the red light |
| drivelm_eval_000151 | text_beats_normal | reasoning_world_knowledge | behavior | 0.0909 | 0.1176 | -0.0267 | The ego vehicle is going straight. The ego vehicle is driving slowly. | wait for the red light to turn green before proceeding. | stop |
| drivelm_eval_000173 | text_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | no | no |
| drivelm_eval_000177 | text_beats_normal | object_recognition | perception | 0.0 | 0.0901 | -0.0901 | There are many traffic elements in the front view. The information of these traffic elements is [(traffic light, unknown | traffic_element | green light |
| drivelm_eval_000207 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 0.1429 | -0.1429 | Yes. | no | no |
| drivelm_eval_000223 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.4 | -0.4 | Go straight. | proceed through the intersection | proceeding through the intersection |
| drivelm_eval_000257 | text_beats_normal | spatial_localization | perception | 0.0678 | 0.1714 | -0.1036 | <c3,CAM_BACK,1290.0,472.5> is at the back of <c1,CAM_BACK,1568.3,505.0>, <c2,CAM_BACK,981.8,499.1> is at the back right  | the traffic light is red | The ego vehicle is moving towards the intersection. |
| drivelm_eval_000299 | text_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | high | high |

## Wrong Image Invariant And Low

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000001 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are many barriers and one construction vehicle to the front right of the ego car. | brown suv | green light |
| drivelm_eval_000002 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There is one truck and one barrier to the front left of the ego car. | brown suv | green light |
| drivelm_eval_000004 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are two barriers, many trucks, two trailers, and one car to the back left of the ego car. | brown suv | brown suv, black sedan |
| drivelm_eval_000007 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are many obstacles in front of the ego car. | brown suv | green light |
| drivelm_eval_000009 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | There are two cars, one truck, and one barrier to the back of the ego car. | brown suv | brown suv, black sedan |
| drivelm_eval_000013 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | There are two trailers to the back right of the ego car. | brown suv, black sedan | brown suv, black sedan |
| drivelm_eval_000018 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | no | no |
| drivelm_eval_000020 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | no | no |
| drivelm_eval_000022 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | no | no |
| drivelm_eval_000025 | wrong_image_invariant_low | spatial_localization | perception | 0.1818 | 0.1818 | 0.0 | Yes, there are some traffic elements in the front view. | yes | yes |
| drivelm_eval_000030 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Turn left. | moving | moving |
| drivelm_eval_000032 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | moving | moving |
| drivelm_eval_000036 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | traffic sign | traffic sign |
| drivelm_eval_000037 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None, no, none. | the black sedan would be dangerous if it were to stop abruptly. | the black sedan would affect the ego vehicle and could cause a collision if it does not stop. |
| drivelm_eval_000039 | wrong_image_invariant_low | spatial_localization | prediction | 0.0 | 0.0 | 0.0 | Turn left. | unknown | Vehicle is moving |

## Interpretation

- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.
- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.
- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.
