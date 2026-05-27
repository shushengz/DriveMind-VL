# DriveLM Scene drivelm_dev_scene_sft_1200 Visual-Control Failure Analysis

## Summary

```json
{
  "count": 300,
  "mean_normal_f1": 0.3782,
  "mean_text_only_f1": 0.3935,
  "mean_wrong_image_f1": 0.3627,
  "mean_blank_image_f1": 0.4185,
  "mean_visual_dependency_gap": -0.1576,
  "failure_type_counts": {
    "wrong_image_invariant_low": 82,
    "wrong_image_invariant": 82,
    "blank_beats_normal": 54,
    "acceptable_or_mixed": 35,
    "wrong_image_beats_normal": 25,
    "text_beats_normal": 17,
    "normal_low": 5
  },
  "failure_type_by_capability": {
    "object_recognition": {
      "wrong_image_invariant_low": 6,
      "blank_beats_normal": 3,
      "acceptable_or_mixed": 2
    },
    "reasoning_world_knowledge": {
      "wrong_image_invariant": 43,
      "wrong_image_invariant_low": 41,
      "blank_beats_normal": 36,
      "acceptable_or_mixed": 10,
      "wrong_image_beats_normal": 9,
      "text_beats_normal": 9,
      "normal_low": 2
    },
    "scene_completeness": {
      "wrong_image_invariant": 32,
      "acceptable_or_mixed": 14,
      "blank_beats_normal": 9,
      "wrong_image_beats_normal": 8,
      "wrong_image_invariant_low": 8,
      "text_beats_normal": 7,
      "normal_low": 2
    },
    "spatial_localization": {
      "wrong_image_invariant_low": 27,
      "acceptable_or_mixed": 9,
      "wrong_image_beats_normal": 8,
      "wrong_image_invariant": 7,
      "blank_beats_normal": 6,
      "text_beats_normal": 1,
      "normal_low": 1
    }
  },
  "failure_type_by_category": {
    "behavior": {
      "wrong_image_invariant_low": 2,
      "blank_beats_normal": 1
    },
    "perception": {
      "wrong_image_invariant": 37,
      "wrong_image_invariant_low": 26,
      "acceptable_or_mixed": 23,
      "blank_beats_normal": 17,
      "wrong_image_beats_normal": 14,
      "text_beats_normal": 7,
      "normal_low": 3
    },
    "planning": {
      "wrong_image_invariant_low": 27,
      "blank_beats_normal": 18,
      "wrong_image_invariant": 12,
      "acceptable_or_mixed": 9,
      "wrong_image_beats_normal": 6,
      "text_beats_normal": 3
    },
    "prediction": {
      "wrong_image_invariant": 33,
      "wrong_image_invariant_low": 27,
      "blank_beats_normal": 18,
      "text_beats_normal": 7,
      "wrong_image_beats_normal": 5,
      "acceptable_or_mixed": 3,
      "normal_low": 2
    }
  }
}
```

## Worst Visual Dependency Gaps

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000009 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000013 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |
| drivelm_dev_scene_000014 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | The ego vehicle. | <c3,CAM_BACK_LEFT,363.1,450.8> | <c4,CAM_BACK,838.3,464.2> |
| drivelm_dev_scene_000015 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000020 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000038 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000039 | wrong_image_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_dev_scene_000058 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |
| drivelm_dev_scene_000066 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000073 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000074 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000085 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |
| drivelm_dev_scene_000086 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000089 | wrong_image_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | No. | Yes. | Yes. |
| drivelm_dev_scene_000090 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |

## Blank Image Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000001 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 0.5 | -0.5 | The ego vehicle. | <c3,CAM_FRONT,1325.8,512.5> | Vehicle. |
| drivelm_dev_scene_000009 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000013 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |
| drivelm_dev_scene_000015 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000017 | blank_beats_normal | spatial_localization | perception | 0.4167 | 0.48 | -0.0633 | There are many cars, pedestrians, three bicycles, and two trucks behind the ego car. | There are vehicles to the back of the ego car. | There are two vehicles to the back of the ego car. |
| drivelm_dev_scene_000020 | blank_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000032 | blank_beats_normal | scene_completeness | perception | 0.2667 | 0.4 | -0.1333 | Many pedestrians are moving. | Two pedestrians are walking towards the front of the ego car. | Moving. |
| drivelm_dev_scene_000038 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000058 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |
| drivelm_dev_scene_000065 | blank_beats_normal | spatial_localization | planning | 0.0 | 0.25 | -0.25 | Reversing to the right rear. | None. | Close the window. |
| drivelm_dev_scene_000066 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000069 | blank_beats_normal | scene_completeness | perception | 0.0 | 0.2105 | -0.2105 | Many cars are parked, and three are moving. | None. | There are two cars to the back of the ego car. |
| drivelm_dev_scene_000073 | blank_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | No. | Yes. | No. |
| drivelm_dev_scene_000074 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Yes. | No. | Yes. |
| drivelm_dev_scene_000085 | blank_beats_normal | reasoning_world_knowledge | planning | 0.0 | 1.0 | -1.0 | Low. | High. | Low. |

## Text Only Beats Normal

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000012 | text_beats_normal | spatial_localization | perception | 0.2105 | 0.7407 | -0.5302 | There are two barriers and one car to the back left of the ego car. | There are orange roadblocks. | There are no objects. |
| drivelm_dev_scene_000014 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | The ego vehicle. | <c3,CAM_BACK_LEFT,363.1,450.8> | <c4,CAM_BACK,838.3,464.2> |
| drivelm_dev_scene_000036 | text_beats_normal | scene_completeness | perception | 0.1333 | 0.3636 | -0.2303 | Many pedestrians are moving, and many are standing. | Two people are walking on the sidewalk. | There are people on the roadside. |
| drivelm_dev_scene_000052 | text_beats_normal | scene_completeness | perception | 0.5714 | 0.75 | -0.1786 | One truck is moving. | It is moving. | It is moving. |
| drivelm_dev_scene_000064 | text_beats_normal | scene_completeness | perception | 0.2857 | 0.5 | -0.2143 | One truck is parked. | It is moving. | It is moving. |
| drivelm_dev_scene_000087 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0513 | 0.5 | -0.4487 | None, no, none. | The object most likely to be occluded by <c1,cam_back,840.0,506.7> would be the black business car. This object does not | The object most likely to be occluded by <c1,cam_back,840.0,506.7> is the black sedan. This object does not affect the e |
| drivelm_dev_scene_000153 | text_beats_normal | reasoning_world_knowledge | prediction | 0.0 | 1.0 | -1.0 | Stationary. | Stay still. | The future state of <c4,CAM_FRONT,844.2,567.5> is stationary. |
| drivelm_dev_scene_000157 | text_beats_normal | reasoning_world_knowledge | prediction | 0.1176 | 0.5 | -0.3824 | None, no, none. | No, it won't affect the ego vehicle. No, this object will not cause danger. | The black sedan in front of the ego vehicle is most likely to be occluded by <c3,CAM_BACK,891.7,528.3>. This object woul |
| drivelm_dev_scene_000166 | text_beats_normal | reasoning_world_knowledge | prediction | 0.3333 | 0.5 | -0.1667 | None, no, none. | None of them. | No, it won't affect the ego vehicle. No action of the ego vehicle is dangerous. |
| drivelm_dev_scene_000193 | text_beats_normal | reasoning_world_knowledge | prediction | 0.1333 | 0.5 | -0.3667 | None, no, none. | The pedestrian would not affect the ego vehicle. There is no danger. | The pedestrian would affect the ego vehicle. The action of the ego vehicle that is dangerous is to make a left turn. |
| drivelm_dev_scene_000202 | text_beats_normal | scene_completeness | perception | 0.2222 | 0.4 | -0.1778 | One motorcycle is with a rider. | It is moving. | It is moving. |
| drivelm_dev_scene_000210 | text_beats_normal | scene_completeness | perception | 0.0 | 1.0 | -1.0 | Yes. | No. | No. |
| drivelm_dev_scene_000211 | text_beats_normal | reasoning_world_knowledge | planning | 0.2778 | 0.3673 | -0.0895 | The action is to remain stationary. The reason for this action is to follow the traffic rules, which is highly probable. | The ego vehicle should accelerate and turn right. The probability of this action is high. | The ego vehicle should accelerate and pass the red pickup truck. |
| drivelm_dev_scene_000226 | text_beats_normal | scene_completeness | perception | 0.2857 | 0.5 | -0.2143 | One truck is parked. | It is moving. | It is moving. |
| drivelm_dev_scene_000245 | text_beats_normal | scene_completeness | planning | 0.0 | 0.087 | -0.087 | The lighting in this scene will affect the driver's ability to drive. | It is dark. | I'm sorry, I can't see anything. |

## Wrong Image Invariant And Low

| id | failure_type | capability | category | normal_f1 | control_max_f1 | visual_dependency_gap | gold_answer | normal_answer | blank_image_answer |
|---|---|---|---|---|---|---|---|---|---|
| drivelm_dev_scene_000002 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Keep going at the same speed. | Slow down and stop. | Slow down. |
| drivelm_dev_scene_000011 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | The ego vehicle. | <c1,CAM_BACK,535.4,481.7> | Traffic element. |
| drivelm_dev_scene_000021 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | Moving. | Moving. |
| drivelm_dev_scene_000022 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Decelerate gradually without braking. | Wait. | Stop. |
| drivelm_dev_scene_000023 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.1333 | 0.1333 | 0.0 | The action is none, the reason is there is no safety issue. | Park the car. | Close the window. |
| drivelm_dev_scene_000025 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | Please wait. | It means that the traffic light is red. | It means that there is a red light. |
| drivelm_dev_scene_000028 | wrong_image_invariant_low | reasoning_world_knowledge | prediction | 0.0 | 0.0 | 0.0 | No. | Yes. | Yes. |
| drivelm_dev_scene_000029 | wrong_image_invariant_low | spatial_localization | perception | 0.0777 | 0.0943 | -0.0166 | There is a gray SUV to the front left of the ego vehicle, a blue sedan to the front right of the ego vehicle, a pedestri | There are pedestrians, vehicles, and buildings. | There are three pedestrians. |
| drivelm_dev_scene_000030 | wrong_image_invariant_low | spatial_localization | perception | 0.0714 | 0.0732 | -0.0018 | There is a black SUV to the front left of the ego vehicle, a pedestrian to the front of the ego vehicle, a white sedan t | There are vehicles, traffic signs, traffic cones, and buildings. | There are two vehicles in the scene. |
| drivelm_dev_scene_000033 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | Moving. | Moving. |
| drivelm_dev_scene_000034 | wrong_image_invariant_low | spatial_localization | planning | 0.0 | 0.0 | 0.0 | The priority of the objects that the ego vehicle should consider, in descending order, is: <c2,CAM_FRONT_RIGHT,803.4,512 | front car. | pedestrian, black truck, silver SUV. |
| drivelm_dev_scene_000035 | wrong_image_invariant_low | reasoning_world_knowledge | behavior | 0.1429 | 0.1429 | 0.0 | The ego vehicle is going straight. The ego vehicle is driving fast. | Continue straight. | Keep driving. |
| drivelm_dev_scene_000043 | wrong_image_invariant_low | spatial_localization | perception | 0.0 | 0.0 | 0.0 | Going ahead. | Moving. | Moving. |
| drivelm_dev_scene_000045 | wrong_image_invariant_low | reasoning_world_knowledge | planning | 0.0 | 0.0 | 0.0 | Low. | High. | The probability of colliding with <c3,CAM_BACK,890.8,638.3> after the ego vehicle goes straight and keeps the same speed |
| drivelm_dev_scene_000049 | wrong_image_invariant_low | spatial_localization | planning | 0.0 | 0.0 | 0.0 | Reverse right. | None. | Close the window. |

## Interpretation

- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.
- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.
- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.
