# Visual-Control Case Comparison: sft_v2 vs v5c_step20

## Summary

```json
{
  "count": 100,
  "mean_delta_normal_f1": -0.007,
  "mean_delta_text_only_f1": -0.0107,
  "mean_delta_wrong_image_f1": -0.0224,
  "mean_delta_blank_image_f1": -0.0039,
  "mean_delta_control_max_f1": -0.0179,
  "mean_delta_setting_gap": 0.0109,
  "delta_type_counts": {
    "mixed_or_neutral": 64,
    "visual_gap_regression": 15,
    "candidate_better_overall": 10,
    "normal_regression": 5,
    "candidate_better_control_no_normal_loss": 4,
    "answer_suppression_tradeoff": 2
  },
  "delta_type_by_capability": {
    "counting": {
      "mixed_or_neutral": 8,
      "candidate_better_control_no_normal_loss": 2,
      "normal_regression": 1
    },
    "object_recognition": {
      "mixed_or_neutral": 10
    },
    "other": {
      "mixed_or_neutral": 11,
      "visual_gap_regression": 3,
      "candidate_better_overall": 3
    },
    "reasoning_world_knowledge": {
      "visual_gap_regression": 2,
      "mixed_or_neutral": 2,
      "candidate_better_overall": 1,
      "candidate_better_control_no_normal_loss": 1
    },
    "spatial_localization": {
      "mixed_or_neutral": 33,
      "visual_gap_regression": 10,
      "candidate_better_overall": 6,
      "normal_regression": 4,
      "answer_suppression_tradeoff": 2,
      "candidate_better_control_no_normal_loss": 1
    }
  }
}
```

## Reading The Deltas

- Positive deltas mean `v5c_step20` is higher than `sft_v2`.
- Lower control F1 can be good only when normal F1 is preserved.
- `answer_suppression_tradeoff` means both normal and controls dropped, so the model likely became less willing or less able to answer rather than better grounded.

## Largest Normal Regressions

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | sft_v2_normal_answer | v5c_step20_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000224 | counting | normal_regression | -1.0 | 0.0 | -1.0 | None. | None. | There are two cars parked on the left side of the road. |
| lingoqa_eval_000311 | spatial_localization | answer_suppression_tradeoff | -1.0 | -1.0 | 0.0 | Accelerating. | Accelerating. | The traffic in front is decelerating. |
| lingoqa_eval_000217 | spatial_localization | normal_regression | -0.2641 | 0.1647 | -0.4288 | The car slows down, because it approaches an intersection where the traffic lights turn red from amber. | The car accelerates as it approaches the roundabout, because the traffic light turns green. | We accelerate as we approach the roundabout, because we need to merge onto the roundabout. |
| lingoqa_eval_000470 | spatial_localization | answer_suppression_tradeoff | -0.1868 | -0.0544 | -0.1324 | The car accelerates and enters the intersection, as the traffic lights are green and the traffic in front enters the int | The car accelerates as it reaches the junction, because the traffic light turns green. | The car accelerates, as it is following a motorcycle that is turning right at the junction. |
| lingoqa_eval_000011 | spatial_localization | normal_regression | -0.1047 | -0.0386 | -0.0661 | The car accelerates slightly, follows the car ahead and crosses the zebra crossing, as no pedestrian is about to cross a | The car continues straight ahead, as it is approaching a roundabout and needs to turn left. | Slow down, there is a pedestrian crossing the road. |
| lingoqa_eval_000444 | spatial_localization | normal_regression | -0.1043 | 0.0 | -0.1043 | The car starts and moves right, because the vehicle in front is pulling away at the green traffic light and then it need | The car slows down because it is approaching a red light and there is a van in front of it. | The car slows down because it is approaching a stop line and there is a truck ahead of it. |
| lingoqa_eval_000005 | spatial_localization | normal_regression | -0.05 | 0.1545 | -0.2045 | The car starts, because the traffic lights turn green. | The car stops at a traffic light, because it is red. | The car slows down because it is approaching a roundabout with traffic lights and cyclists. The traffic light is green,  |
| lingoqa_eval_000392 | spatial_localization | visual_gap_regression | -0.0361 | 0.0635 | -0.0996 | We are stationary and then start accelerating. Although the lights are green, the van in front is stationary due to cars | The car continues straight ahead, as it is following the bus that is in front of it. | The car continues straight as it passes the bus, because the bus is stopped at the traffic light. |
| lingoqa_eval_000384 | spatial_localization | visual_gap_regression | -0.0338 | 0.0023 | -0.0361 | I am overtaking cyclists as they are slower than me and then decelerate to be careful around oncoming cyclist. | The car slows down because it is overtaking a cyclist. The cyclist is in the left lane and the car is in the right lane. | The car slows down as it approaches the cyclist. The car slows down because the cyclist is approaching the car. |
| lingoqa_eval_000065 | other | visual_gap_regression | -0.0329 | 0.0 | -0.0329 | Should pay attention to the state of the lights as they are red so we must stop. | Pay attention to the traffic lights and the cyclists. | Pay attention to the traffic lights, the cyclists and the pedestrians. |
| lingoqa_eval_000027 | spatial_localization | visual_gap_regression | -0.0016 | 0.0 | -0.0016 | You shouldn't completely stop, but you should slow down since you're overtaking 2 cyclists and steering to the right but | No, there is no traffic. | No, there is no traffic ahead. |
| lingoqa_eval_000001 | counting | mixed_or_neutral | 0.0 | 0.0 | 0.0 | Zero. | There are no cyclists. | There are no cyclists. |

## Largest Control Reductions

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | sft_v2_normal_answer | v5c_step20_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000092 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 1 | None. | There are no parked cars on the left. |
| lingoqa_eval_000112 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 0 | None. | None. |
| lingoqa_eval_000311 | spatial_localization | answer_suppression_tradeoff | -1.0 | -1.0 | 0.0 | Accelerating. | Accelerating. | The traffic in front is decelerating. |
| lingoqa_eval_000081 | reasoning_world_knowledge | candidate_better_control_no_normal_loss | 0.0 | -0.1429 | 0.1429 | Light has turned red. | I am stopped because I have to give way to a bus that is turning right. | I am stopped because I have to give way to a bus that is turning right. |
| lingoqa_eval_000428 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.0901 | 0.0901 | No, there is a cyclist in our lane, and a dust cart in the lane to our right, meaning that we cannot overtake the cyclis | Yes, the road is clear. | Yes, the road is clear. |
| lingoqa_eval_000099 | spatial_localization | candidate_better_overall | 0.1042 | -0.0667 | 0.1709 | The car starts, as the cyclist and the bus in front are pulling away. | The cyclist is overtaking the bus. The bus is slowing down to let the cyclist pass. | The cyclist is passing the bus, because he is in front of it and has a clear path. |
| lingoqa_eval_000470 | spatial_localization | answer_suppression_tradeoff | -0.1868 | -0.0544 | -0.1324 | The car accelerates and enters the intersection, as the traffic lights are green and the traffic in front enters the int | The car accelerates as it reaches the junction, because the traffic light turns green. | The car accelerates, as it is following a motorcycle that is turning right at the junction. |
| lingoqa_eval_000011 | spatial_localization | normal_regression | -0.1047 | -0.0386 | -0.0661 | The car accelerates slightly, follows the car ahead and crosses the zebra crossing, as no pedestrian is about to cross a | The car continues straight ahead, as it is approaching a roundabout and needs to turn left. | Slow down, there is a pedestrian crossing the road. |
| lingoqa_eval_000498 | spatial_localization | mixed_or_neutral | 0.0 | -0.0237 | 0.0237 | The car accelerates, in order to match the speed of the traffic in front. | The car continues straight ahead, as it is following the traffic flow. | The car continues straight ahead, as it is following the traffic flow. |
| lingoqa_eval_000478 | spatial_localization | candidate_better_overall | 0.1729 | -0.0048 | 0.1777 | The car remains stopped before accelerating gently, because a jaywalker is crossing the street behind the bus that is pu | {"task":"external_vqa","answer":"The bus is approaching the junction. The bus is red and has the number 77 on it. It is  | The bus is approaching the junction. The bus is red and has the number 77 on it. It is approaching the junction at a low |
| lingoqa_eval_000001 | counting | mixed_or_neutral | 0.0 | 0.0 | 0.0 | Zero. | There are no cyclists. | There are no cyclists. |
| lingoqa_eval_000007 | counting | mixed_or_neutral | 0.0 | 0.0 | 0.0 | 5 pedestrians crossing the road, 2 right in front of the car, 2 further away, and one to the right. | None. | None. |

## Largest Setting-Gap Improvements

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | sft_v2_normal_answer | v5c_step20_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000092 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 1 | None. | There are no parked cars on the left. |
| lingoqa_eval_000112 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 0 | None. | None. |
| lingoqa_eval_000240 | other | candidate_better_overall | 0.4242 | 0.0237 | 0.4005 | Yes because the zebra crossing is clear and there is not any pedestrian waiting to cross it. | No, I can't. | Yes, I can drive over the zebra crossing as there is no pedestrian crossing the road. |
| lingoqa_eval_000322 | other | candidate_better_overall | 0.3636 | 0.0 | 0.3636 | Yes there a cyclist in a red coat. | No. | Yes, a car. |
| lingoqa_eval_000467 | spatial_localization | candidate_better_overall | 0.3077 | 0.0 | 0.3077 | The vehicles are parked on both sides of the road. | Left. | On the left. |
| lingoqa_eval_000478 | spatial_localization | candidate_better_overall | 0.1729 | -0.0048 | 0.1777 | The car remains stopped before accelerating gently, because a jaywalker is crossing the street behind the bus that is pu | {"task":"external_vqa","answer":"The bus is approaching the junction. The bus is red and has the number 77 on it. It is  | The bus is approaching the junction. The bus is red and has the number 77 on it. It is approaching the junction at a low |
| lingoqa_eval_000099 | spatial_localization | candidate_better_overall | 0.1042 | -0.0667 | 0.1709 | The car starts, as the cyclist and the bus in front are pulling away. | The cyclist is overtaking the bus. The bus is slowing down to let the cyclist pass. | The cyclist is passing the bus, because he is in front of it and has a clear path. |
| lingoqa_eval_000081 | reasoning_world_knowledge | candidate_better_control_no_normal_loss | 0.0 | -0.1429 | 0.1429 | Light has turned red. | I am stopped because I have to give way to a bus that is turning right. | I am stopped because I have to give way to a bus that is turning right. |
| lingoqa_eval_000036 | reasoning_world_knowledge | candidate_better_overall | 0.1176 | 0.0 | 0.1176 | The traffic light is green and no pedestrian is crossing. | I have priority. | I have priority over the other vehicles. |
| lingoqa_eval_000285 | spatial_localization | candidate_better_overall | 0.1006 | 0.001 | 0.0996 | The follows the vehicle in front, as both the car and the vehicle in front are turning left at the intersection and the  | The car stops at a traffic light, because it is red. | The car continues straight as it reaches the junction, because there is no traffic coming from the left. |
| lingoqa_eval_000427 | spatial_localization | candidate_better_overall | 0.1192 | 0.0208 | 0.0984 | The cars in front are at a stop because the light is red. | {"task":"external_vqa","answer":"I see a white van in front of me, a red car on my left and a blue car on my right. I am | I see a white van in front of me, a red car on my left and a blue car on my right. I am driving at a speed of 20 km/h. |
| lingoqa_eval_000428 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.0901 | 0.0901 | No, there is a cyclist in our lane, and a dust cart in the lane to our right, meaning that we cannot overtake the cyclis | Yes, the road is clear. | Yes, the road is clear. |

## Best Tradeoff Candidates

| id | capability | delta_type | delta_normal_f1 | delta_control_max_f1 | delta_setting_gap | gold_answer | sft_v2_normal_answer | v5c_step20_normal_answer |
|---|---|---|---|---|---|---|---|---|
| lingoqa_eval_000092 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 1 | None. | There are no parked cars on the left. |
| lingoqa_eval_000112 | counting | candidate_better_control_no_normal_loss | 0.0 | -1.0 | 1.0 | 0 | None. | None. |
| lingoqa_eval_000240 | other | candidate_better_overall | 0.4242 | 0.0237 | 0.4005 | Yes because the zebra crossing is clear and there is not any pedestrian waiting to cross it. | No, I can't. | Yes, I can drive over the zebra crossing as there is no pedestrian crossing the road. |
| lingoqa_eval_000322 | other | candidate_better_overall | 0.3636 | 0.0 | 0.3636 | Yes there a cyclist in a red coat. | No. | Yes, a car. |
| lingoqa_eval_000467 | spatial_localization | candidate_better_overall | 0.3077 | 0.0 | 0.3077 | The vehicles are parked on both sides of the road. | Left. | On the left. |
| lingoqa_eval_000478 | spatial_localization | candidate_better_overall | 0.1729 | -0.0048 | 0.1777 | The car remains stopped before accelerating gently, because a jaywalker is crossing the street behind the bus that is pu | {"task":"external_vqa","answer":"The bus is approaching the junction. The bus is red and has the number 77 on it. It is  | The bus is approaching the junction. The bus is red and has the number 77 on it. It is approaching the junction at a low |
| lingoqa_eval_000099 | spatial_localization | candidate_better_overall | 0.1042 | -0.0667 | 0.1709 | The car starts, as the cyclist and the bus in front are pulling away. | The cyclist is overtaking the bus. The bus is slowing down to let the cyclist pass. | The cyclist is passing the bus, because he is in front of it and has a clear path. |
| lingoqa_eval_000081 | reasoning_world_knowledge | candidate_better_control_no_normal_loss | 0.0 | -0.1429 | 0.1429 | Light has turned red. | I am stopped because I have to give way to a bus that is turning right. | I am stopped because I have to give way to a bus that is turning right. |
| lingoqa_eval_000036 | reasoning_world_knowledge | candidate_better_overall | 0.1176 | 0.0 | 0.1176 | The traffic light is green and no pedestrian is crossing. | I have priority. | I have priority over the other vehicles. |
| lingoqa_eval_000285 | spatial_localization | candidate_better_overall | 0.1006 | 0.001 | 0.0996 | The follows the vehicle in front, as both the car and the vehicle in front are turning left at the intersection and the  | The car stops at a traffic light, because it is red. | The car continues straight as it reaches the junction, because there is no traffic coming from the left. |
| lingoqa_eval_000427 | spatial_localization | candidate_better_overall | 0.1192 | 0.0208 | 0.0984 | The cars in front are at a stop because the light is red. | {"task":"external_vqa","answer":"I see a white van in front of me, a red car on my left and a blue car on my right. I am | I see a white van in front of me, a red car on my left and a blue car on my right. I am driving at a speed of 20 km/h. |
| lingoqa_eval_000428 | spatial_localization | candidate_better_control_no_normal_loss | 0.0 | -0.0901 | 0.0901 | No, there is a cyclist in our lane, and a dust cart in the lane to our right, meaning that we cannot overtake the cyclis | Yes, the road is clear. | Yes, the road is clear. |
