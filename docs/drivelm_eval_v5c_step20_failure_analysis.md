# DriveLM drivelm_eval_v5c_step20 Visual-Control Failure Analysis

## Summary

```json
{
  "count": 300,
  "mean_normal_f1": 0.2824,
  "mean_text_only_f1": 0.3139,
  "mean_wrong_image_f1": 0.2822,
  "mean_blank_image_f1": 0.3231,
  "mean_visual_dependency_gap": -0.1205,
  "failure_type_counts": {
    "wrong_image_invariant_low": 106,
    "wrong_image_invariant": 94,
    "blank_beats_normal": 73,
    "text_beats_normal": 25,
    "acceptable_or_mixed": 2
  },
  "failure_type_by_capability": {
    "object_recognition": {
      "wrong_image_invariant_low": 9,
      "text_beats_normal": 4,
      "blank_beats_normal": 4,
      "wrong_image_invariant": 3
    },
    "reasoning_world_knowledge": {
      "wrong_image_invariant_low": 57,
      "wrong_image_invariant": 35,
      "blank_beats_normal": 29,
      "text_beats_normal": 7,
      "acceptable_or_mixed": 2
    },
    "scene_completeness": {
      "wrong_image_invariant": 41,
      "blank_beats_normal": 25,
      "wrong_image_invariant_low": 25,
      "text_beats_normal": 4
    },
    "spatial_localization": {
      "wrong_image_invariant": 15,
      "blank_beats_normal": 15,
      "wrong_image_invariant_low": 14,
      "text_beats_normal": 10
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
      "wrong_image_invariant": 56,
      "blank_beats_normal": 39,
      "wrong_image_invariant_low": 34,
      "text_beats_normal": 14
    },
    "planning": {
      "wrong_image_invariant_low": 21,
      "blank_beats_normal": 19,
      "wrong_image_invariant": 10,
      "text_beats_normal": 5
    },
    "prediction": {
      "wrong_image_invariant_low": 51,
      "wrong_image_invariant": 28,
      "blank_beats_normal": 15,
      "text_beats_normal": 5
    }
  }
}
```

## Worst Visual Dependency Gaps

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000057 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No, the traffic light is green. | Yes. |
| drivelm_eval_000070 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is green. | Yes. |
| drivelm_eval_000101 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000141 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because there is no traffic coming from the left. | Yes. |
| drivelm_eval_000143 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is red. | Yes. |
| drivelm_eval_000171 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000172 | text_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_eval_000186 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | It is a road barrier. | No. |
| drivelm_eval_000190 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | It is a road barrier. | No. |
| drivelm_eval_000207 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No, the traffic light is green. | Yes. |
| drivelm_eval_000220 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because there is no traffic coming from the left. | Yes. |
| drivelm_eval_000242 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | {"task":"external_vqa","answer":"No.","" ,"reason":"No motorcycles with riders are seen in the image."} | No. |
| drivelm_eval_000248 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000249 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_eval_000251 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |

## Blank Image Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000002 | blank_beats_normal | spatial_localization | perception | 0.2857 | 0.5 | -0.2143 | There is one truck and one barrier to the front left of the ego car. | There is no traffic in front. | There is no traffic element to the front left. |
| drivelm_eval_000008 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.5 | -0.5 | Two trailers are parked. | The green light is on. | There are no trailers. |
| drivelm_eval_000009 | blank_beats_normal | scene_completeness | perception | 0.4348 | 0.7407 | -0.3059 | There are two cars, one truck, and one barrier to the back of the ego car. | There are no objects to the back. | There are two vehicles to the back of the ego car. |
| drivelm_eval_000025 | blank_beats_normal | spatial_localization | perception | 0.0 | 0.1818 | -0.1818 | Yes, there are some traffic elements in the front view. | No. | Yes. |
| drivelm_eval_000029 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.3333 | -0.3333 | Moving. | The traffic light is green. | The brown SUV is moving. |
| drivelm_eval_000033 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.8 | -0.8 | Brown SUV. | The road is wet and there is rain falling. | A brown SUV. |
| drivelm_eval_000034 | blank_beats_normal | scene_completeness | perception | 0.3077 | 0.8 | -0.4923 | Black sedan. | The black sedan is approaching the intersection from the right side. | A black sedan. |
| drivelm_eval_000035 | blank_beats_normal | scene_completeness | perception | 0.5 | 0.6667 | -0.1667 | Green light. | There is a green traffic light. | A green traffic light. |
| drivelm_eval_000047 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 0.3333 | -0.3333 | The ego vehicle. | Traffic light. | The traffic light. |
| drivelm_eval_000057 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No, the traffic light is green. | Yes. |
| drivelm_eval_000063 | blank_beats_normal | reasoning_world_knowledge | planning | 0.3019 | 0.4815 | -0.1796 | The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability. | The ego vehicle should accelerate and proceed straight ahead as it is green. The probability of the ego vehicle proceedi | The ego vehicle should accelerate to overtake the black sedan. The probability of this action is high as the black sedan |
| drivelm_eval_000065 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 0.087 | -0.087 | No such action will lead to a collision. | The ego vehicle should slow down and stop at the traffic light. | The ego vehicle should slow down and stop if it is not safe to proceed. |
| drivelm_eval_000067 | blank_beats_normal | reasoning_world_knowledge | planning | 0.2105 | 0.4255 | -0.215 | The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability. | The ego vehicle should accelerate and proceed straight ahead as the traffic light is green. | The ego vehicle should accelerate and merge onto the right lane. The probability of this action is high as the traffic l |
| drivelm_eval_000070 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No, because the traffic light is green. | Yes. |
| drivelm_eval_000071 | blank_beats_normal | reasoning_world_knowledge | planning | 0.1538 | 0.1818 | -0.028 | Keep going at the same speed. | The ego vehicle should continue straight ahead. | The ego vehicle should accelerate. |

## Text Only Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000004 | text_beats_normal | spatial_localization | perception | 0.3077 | 0.6452 | -0.3375 | There are two barriers, many trucks, two trailers, and one car to the back left of the ego car. | There is no traffic to the back. | There are no objects to the back left. |
| drivelm_eval_000010 | text_beats_normal | scene_completeness | perception | 0.1538 | 0.2667 | -0.1129 | Two cars are moving. | The black sedan is moving towards the ego vehicle. | The black sedan is moving. |
| drivelm_eval_000013 | text_beats_normal | spatial_localization | perception | 0.6 | 0.9167 | -0.3167 | There are two trailers to the back right of the ego car. | There are no objects to the back right. | There are no objects to the back right of the ego car. |
| drivelm_eval_000024 | text_beats_normal | scene_completeness | perception | 0.1765 | 0.2222 | -0.0457 | <c2,CAM_BACK,864.2,468.3> is at the back of <c1,CAM_BACK,1088.3,497.5>. | The traffic lights are ahead of the car. The car is in front of the traffic lights. | The ego vehicle is behind the black sedan. |
| drivelm_eval_000026 | text_beats_normal | object_recognition | perception | 0.1395 | 0.2273 | -0.0878 | There are two traffic elements in the front view. The information of these traffic elements is [(traffic light, green, 6 | There is no traffic element. | There is no traffic element. |
| drivelm_eval_000052 | text_beats_normal | spatial_localization | prediction | 0.2069 | 0.2295 | -0.0226 | The action is to turn left, the reason is there is no safety issue. | The driver should slow down and prepare to stop as the traffic light turns red. | The driver should slow down and prepare to stop as the traffic light turns green. |
| drivelm_eval_000061 | text_beats_normal | object_recognition | prediction | 0.0 | 0.5 | -0.5 | Traffic light. | A railroad crossing sign. | There is no traffic sign. |
| drivelm_eval_000080 | text_beats_normal | spatial_localization | perception | 0.4286 | 0.5625 | -0.1339 | There are many trucks, one car, two barriers, and one traffic cone to the back left of the ego car. | There are no objects to the back left. | There is no object to the back left. |
| drivelm_eval_000087 | text_beats_normal | spatial_localization | perception | 0.3529 | 0.7619 | -0.409 | There is one barrier in front of the ego car. | There are no objects to the front. | There is no traffic ahead. |
| drivelm_eval_000151 | text_beats_normal | reasoning_world_knowledge | behavior | 0.381 | 0.4211 | -0.0401 | The ego vehicle is going straight. The ego vehicle is driving slowly. | The ego vehicle should stop at the red light. | The ego vehicle will stop at the red light. |
| drivelm_eval_000152 | text_beats_normal | spatial_localization | perception | 0.4 | 0.6207 | -0.2207 | There are many trucks, two cars, and many barriers to the back left of the ego car. | There is no traffic to the back left. | There is no object to the back left. |
| drivelm_eval_000172 | text_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_eval_000175 | text_beats_normal | spatial_localization | perception | 0.069 | 0.2143 | -0.1453 | <c2,CAM_BACK,966.7,580.0> is on the back right of <c1,CAM_FRONT_LEFT,231.5,472.1>. | The traffic lights are green, so I can proceed straight ahead. | The ego vehicle is behind the white sedan. |
| drivelm_eval_000177 | text_beats_normal | object_recognition | perception | 0.1053 | 0.1299 | -0.0246 | There are many traffic elements in the front view. The information of these traffic elements is [(traffic light, unknown | There is a green light. | There is a green light ahead. |
| drivelm_eval_000198 | text_beats_normal | object_recognition | prediction | 0.2903 | 0.3651 | -0.0748 | Firstly notice that <c3,CAM_FRONT,864.5,293.6>; the object is a traffic sign, so the ego vehicle should continue going a | {"task":"external_vqa","answer":"The ego vehicle should first notice the traffic light, which is green. The traffic ligh | {"task":"external_vqa","answer":"The ego vehicle should first notice the green light, which is green and the ego vehicle |

## Wrong Image Invariant And Low

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_eval_000015 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000018 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000019 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000020 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000021 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | No. | No. |
| drivelm_eval_000022 | wrong_image_invariant_low | scene_completeness | perception | 0.0 | 0.0 | 0.0 | Yes. | {"task":"external_vqa","answer":"No.","" ,"reason":"No."} | No. |
| drivelm_eval_000028 | wrong_image_invariant_low | spatial_localization | perception | 0.2222 | 0.2222 | 0.0 | There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to  | There is a green traffic light, a railroad crossing sign, and a white truck. | There is a green traffic light. |
| drivelm_eval_000030 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Turn left. | The car is moving. | The object is moving. |
| drivelm_eval_000032 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | The black sedan is moving. | The black sedan is moving. |
| drivelm_eval_000036 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | Yes. | It is a traffic sign. |
| drivelm_eval_000037 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None, no, none. | The black sedan. | The black sedan. |
| drivelm_eval_000039 | wrong_image_invariant_low | spatial_localization | prediction | 0.25 | 0.25 | 0.0 | Turn left. | The traffic light will turn red. | The brown SUV will continue to move. |
| drivelm_eval_000040 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | Yes. | It is a traffic light. |
| drivelm_eval_000041 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | None, no, none. | The black sedan. | The black sedan. |
| drivelm_eval_000043 | wrong_image_invariant_low | spatial_localization | prediction | 0.25 | 0.25 | 0.0 | Turn left. | The traffic light will turn red. | The black sedan will continue straight. |

## Interpretation

- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.
- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.
- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.
