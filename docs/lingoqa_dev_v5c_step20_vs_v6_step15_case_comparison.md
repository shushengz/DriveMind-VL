# Visual-Control Case Comparison: v5c_step20 vs v6_step15

## Summary

```json
{
  "count": 100,
  "mean_delta_normal_f1": -0.0095,
  "mean_delta_text_only_f1": -0.0193,
  "mean_delta_wrong_image_f1": 0.0005,
  "mean_delta_blank_image_f1": -0.0128,
  "mean_delta_control_max_f1": -0.015,
  "mean_delta_setting_gap": 0.0055,
  "delta_type_counts": {
    "mixed_or_neutral": 66,
    "visual_gap_regression": 11,
    "normal_regression": 10,
    "candidate_better_control_no_normal_loss": 8,
    "candidate_better_overall": 4,
    "wrong_image_control_improved": 1
  },
  "delta_type_by_capability": {
    "counting": {
      "mixed_or_neutral": 13
    },
    "object_recognition": {
      "mixed_or_neutral": 6,
      "candidate_better_control_no_normal_loss": 1
    },
    "other": {
      "mixed_or_neutral": 20,
      "visual_gap_regression": 3,
      "normal_regression": 2,
      "candidate_better_overall": 2,
      "candidate_better_control_no_normal_loss": 1
    },
    "spatial_localization": {
      "mixed_or_neutral": 27,
      "visual_gap_regression": 8,
      "normal_regression": 8,
      "candidate_better_control_no_normal_loss": 6,
      "candidate_better_overall": 2,
      "wrong_image_control_improved": 1
    }
  }
}
```

## Reading The Deltas

- Positive deltas mean `v6_step15` is higher than `v5c_step20`.
- Lower control F1 can be good only when normal F1 is preserved.
- `answer_suppression_tradeoff` means both normal and controls dropped, so the model likely became less willing or less able to answer rather than better grounded.

## Largest Normal Regressions

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | v5c_step20_normal_answer | v6_step15_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000429 | spatial_localization | normal_regression | -0.8889 | 0.0 | -0.8889 | Yes, there is a one pedestrian crossing the road ahead. | Yes, there is a pedestrian crossing the road. | No. |
| lingoqa_eval_000334 | other | normal_regression | -0.4503 | 0.0147 | -0.465 | Yes I should decelerate because the traffic light is red. | Yes, because there is a red light ahead. | No, there is no need to decrease my acceleration. |
| lingoqa_eval_000396 | spatial_localization | normal_regression | -0.2854 | -0.0334 | -0.252 | Slightly accelerating in our driving lane, the traffic light is green, 2 pedestrians just finished crossing the road, no | The car continues straight as it passes the junction. The traffic light is green and there is no pedestrian crossing the | The car continues straight ahead, as it is approaching a roundabout and needs to turn left. |
| lingoqa_eval_000368 | spatial_localization | normal_regression | -0.2057 | 0.0231 | -0.2288 | No. There are no pedestrians who want to cross and there is enough room for us to continue on behind the cyclist. | No, I do not need to come to a stop as there is no traffic in front of me. | No. |
| lingoqa_eval_000187 | spatial_localization | normal_regression | -0.1869 | 0.0091 | -0.196 | I am maintaining my speed and following the road and the cyclist ahead, as the traffic ahead is moving off and there is  | The car continues straight ahead, as it is following the cyclist and there are no traffic lights or pedestrians in the w | The car continues straight ahead, as it is following the cyclist. |
| lingoqa_eval_000258 | other | normal_regression | -0.1445 | -0.0381 | -0.1064 | We must remain stopped while the pedestrians cross, and then we can start and accelerate once they have clears the road. | I should slow down and steer to the right to avoid pedestrians crossing the road. | I should slow down and steer to the right. |
| lingoqa_eval_000297 | spatial_localization | normal_regression | -0.1333 | 0.0653 | -0.1986 | Carry on at speed, but paying attention to the cyclists in front and remaining careful. | I will accelerate as I am following two cyclists and there is no traffic ahead. | I will accelerate as I am approaching a junction with a green light. |
| lingoqa_eval_000451 | spatial_localization | normal_regression | -0.0761 | 0.0 | -0.0761 | Yes, there is a cyclist directly in front of the car in the current lane. | Yes, I see two cyclists. One is ahead of me, and the other is on my left. | Yes, I see two cyclists ahead of me. |
| lingoqa_eval_000338 | spatial_localization | normal_regression | -0.069 | 0.0 | -0.069 | The car moves slightly right, as it reaches a width restriction traffic island, then continues driving as the zebra cros | The car continues straight as it approaches the junction, as there is no traffic approaching from the left. | The car continues straight as it approaches the junction. The traffic light is green and there is no oncoming traffic. |
| lingoqa_eval_000480 | spatial_localization | normal_regression | -0.0606 | 0.0 | -0.0606 | The car maintains its speed and changes lane to the right, in order to follow its route. | The car continues straight ahead, as it is following the traffic light and the road markings. | The car accelerates, as it is following a motorcycle that is driving at a slower speed. |
| lingoqa_eval_000196 | spatial_localization | visual_gap_regression | -0.039 | 0.0175 | -0.0565 | The car is driving straight and moving to the right of its lane, in order to overtake parked vehicles on the left of the | The car accelerates, as it is overtaking a cyclist on the left side of the road. | The car accelerates as it passes the cyclist on the left, because the cyclist is moving away from the car. |
| lingoqa_eval_000448 | spatial_localization | visual_gap_regression | -0.0222 | 0.0 | -0.0222 | The car moves to the right lane, because the left lane is reserved for turning left at the next junction. | The car continues straight as it approaches the junction. The traffic light is green and there is no oncoming traffic. | The car continues straight ahead, as it is approaching a roundabout and needs to turn left. |

## Largest Control Reductions

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | v5c_step20_normal_answer | v6_step15_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000153 | object_recognition | candidate_better_control_no_normal_loss | 0.0 | -0.5 | 0.5 | Yes, green. | Yes, green. | Yes, green. |
| lingoqa_eval_000106 | other | candidate_better_control_no_normal_loss | 0.0 | -0.3334 | 0.3334 | I am following the red car. | The car in front. | The car in front. |
| lingoqa_eval_000386 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2581 | 0.2581 | Not at the moment, there is a traffic island and the cyclist veers right avoiding a road bump. | No. | No. |
| lingoqa_eval_000449 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2222 | 0.2222 | I am going around the pedestrian from the left. | I am going around the pedestrian on my left. | I am going around the pedestrian on my left. |
| lingoqa_eval_000151 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1711 | 0.1711 | Yes, there is a cyclist I am following, also there is a car and one white lorry further ahead. | No. | No. |
| lingoqa_eval_000223 | spatial_localization | candidate_better_control_no_normal_loss | -0.0158 | -0.1429 | 0.1271 | Because I am overtaking a truck parked on the left lane. | I am encroaching on the lane to my right because I am overtaking the Tesco delivery truck. | I am overtaking the Tesco delivery truck. |
| lingoqa_eval_000344 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.119 | 0.119 | We are stopping in front of a traffic light because it is red. | We accelerate as we approach the traffic lights, because we need to pass them. | We accelerate as we approach the traffic lights, because we need to pass them. |
| lingoqa_eval_000415 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1061 | 0.1061 | No, because there is a cyclist in front who is travelling at the same speed as us, and we cannot overtake because of the | Yes, because there is no traffic in front of us. | Yes, because there is no traffic in front of us. |
| lingoqa_eval_000037 | spatial_localization | wrong_image_control_improved | 0.0 | -0.0404 | 0.0404 | The car is clearing the intersection while moving to the right lane, in order to overtake the cyclist on the left. | The car accelerates as the traffic light turns green. | The car accelerates as the traffic light turns green. |
| lingoqa_eval_000258 | other | normal_regression | -0.1445 | -0.0381 | -0.1064 | We must remain stopped while the pedestrians cross, and then we can start and accelerate once they have clears the road. | I should slow down and steer to the right to avoid pedestrians crossing the road. | I should slow down and steer to the right. |
| lingoqa_eval_000396 | spatial_localization | normal_regression | -0.2854 | -0.0334 | -0.252 | Slightly accelerating in our driving lane, the traffic light is green, 2 pedestrians just finished crossing the road, no | The car continues straight as it passes the junction. The traffic light is green and there is no pedestrian crossing the | The car continues straight ahead, as it is approaching a roundabout and needs to turn left. |
| lingoqa_eval_000362 | spatial_localization | mixed_or_neutral | 0.0 | -0.0333 | 0.0333 | The car slows down slightly while following a cyclist and moves to the left of the lane, as it needs to avoid the traffi | The car accelerates as the cyclist passes by, because it needs to overtake him. | The car accelerates as the cyclist passes by, because it needs to overtake him. |

## Largest Setting-Gap Improvements

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | v5c_step20_normal_answer | v6_step15_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000235 | spatial_localization | candidate_better_overall | 0.75 | 0.0 | 0.75 | Left. | The truck is parked on the left. | Left. |
| lingoqa_eval_000296 | other | candidate_better_overall | 0.7143 | 0.0 | 0.7143 | No. | No, the traffic light is red. | No. |
| lingoqa_eval_000153 | object_recognition | candidate_better_control_no_normal_loss | 0.0 | -0.5 | 0.5 | Yes, green. | Yes, green. | Yes, green. |
| lingoqa_eval_000106 | other | candidate_better_control_no_normal_loss | 0.0 | -0.3334 | 0.3334 | I am following the red car. | The car in front. | The car in front. |
| lingoqa_eval_000386 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2581 | 0.2581 | Not at the moment, there is a traffic island and the cyclist veers right avoiding a road bump. | No. | No. |
| lingoqa_eval_000449 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2222 | 0.2222 | I am going around the pedestrian from the left. | I am going around the pedestrian on my left. | I am going around the pedestrian on my left. |
| lingoqa_eval_000151 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1711 | 0.1711 | Yes, there is a cyclist I am following, also there is a car and one white lorry further ahead. | No. | No. |
| lingoqa_eval_000223 | spatial_localization | candidate_better_control_no_normal_loss | -0.0158 | -0.1429 | 0.1271 | Because I am overtaking a truck parked on the left lane. | I am encroaching on the lane to my right because I am overtaking the Tesco delivery truck. | I am overtaking the Tesco delivery truck. |
| lingoqa_eval_000344 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.119 | 0.119 | We are stopping in front of a traffic light because it is red. | We accelerate as we approach the traffic lights, because we need to pass them. | We accelerate as we approach the traffic lights, because we need to pass them. |
| lingoqa_eval_000326 | other | candidate_better_overall | 0.1071 | 0.0 | 0.1071 | No, all vehicles are travelling at speed. | No, because there is no traffic ahead. | No. |
| lingoqa_eval_000415 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1061 | 0.1061 | No, because there is a cyclist in front who is travelling at the same speed as us, and we cannot overtake because of the | Yes, because there is no traffic in front of us. | Yes, because there is no traffic in front of us. |
| lingoqa_eval_000411 | spatial_localization | candidate_better_overall | 0.0504 | 0.0 | 0.0504 | I will continue straight and maintain my speed in order to finish overtaking the cyclist. | I will accelerate as I am approaching the traffic lights and then change gear when the light turns green. | I will accelerate as I am approaching the roundabout and then turn left. |

## Best Tradeoff Candidates

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | v5c_step20_normal_answer | v6_step15_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000235 | spatial_localization | candidate_better_overall | 0.75 | 0.0 | 0.75 | Left. | The truck is parked on the left. | Left. |
| lingoqa_eval_000296 | other | candidate_better_overall | 0.7143 | 0.0 | 0.7143 | No. | No, the traffic light is red. | No. |
| lingoqa_eval_000153 | object_recognition | candidate_better_control_no_normal_loss | 0.0 | -0.5 | 0.5 | Yes, green. | Yes, green. | Yes, green. |
| lingoqa_eval_000106 | other | candidate_better_control_no_normal_loss | 0.0 | -0.3334 | 0.3334 | I am following the red car. | The car in front. | The car in front. |
| lingoqa_eval_000386 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2581 | 0.2581 | Not at the moment, there is a traffic island and the cyclist veers right avoiding a road bump. | No. | No. |
| lingoqa_eval_000449 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.2222 | 0.2222 | I am going around the pedestrian from the left. | I am going around the pedestrian on my left. | I am going around the pedestrian on my left. |
| lingoqa_eval_000151 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1711 | 0.1711 | Yes, there is a cyclist I am following, also there is a car and one white lorry further ahead. | No. | No. |
| lingoqa_eval_000223 | spatial_localization | candidate_better_control_no_normal_loss | -0.0158 | -0.1429 | 0.1271 | Because I am overtaking a truck parked on the left lane. | I am encroaching on the lane to my right because I am overtaking the Tesco delivery truck. | I am overtaking the Tesco delivery truck. |
| lingoqa_eval_000344 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.119 | 0.119 | We are stopping in front of a traffic light because it is red. | We accelerate as we approach the traffic lights, because we need to pass them. | We accelerate as we approach the traffic lights, because we need to pass them. |
| lingoqa_eval_000326 | other | candidate_better_overall | 0.1071 | 0.0 | 0.1071 | No, all vehicles are travelling at speed. | No, because there is no traffic ahead. | No. |
| lingoqa_eval_000415 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.1061 | 0.1061 | No, because there is a cyclist in front who is travelling at the same speed as us, and we cannot overtake because of the | Yes, because there is no traffic in front of us. | Yes, because there is no traffic in front of us. |
| lingoqa_eval_000411 | spatial_localization | candidate_better_overall | 0.0504 | 0.0 | 0.0504 | I will continue straight and maintain my speed in order to finish overtaking the cyclist. | I will accelerate as I am approaching the traffic lights and then change gear when the light turns green. | I will accelerate as I am approaching the roundabout and then turn left. |
