# LingoQA Bad Case Taxonomy

## Summary

```json
{
  "count": 100,
  "by_primary_error_type": {
    "all_settings_failed": 39,
    "wrong_image_confound": 23,
    "language_prior_or_question_bias": 18,
    "visual_helped": 15,
    "blank_image_confound": 5
  },
  "by_capability": {
    "counting": 21,
    "object_recognition": 20,
    "other": 20,
    "spatial_localization": 20,
    "reasoning_world_knowledge": 17,
    "weather_road_condition": 2
  },
  "primary_error_type_by_capability": {
    "counting": {
      "all_settings_failed": 14,
      "visual_helped": 4,
      "wrong_image_confound": 2,
      "language_prior_or_question_bias": 1
    },
    "object_recognition": {
      "wrong_image_confound": 6,
      "all_settings_failed": 5,
      "visual_helped": 5,
      "language_prior_or_question_bias": 4
    },
    "other": {
      "language_prior_or_question_bias": 7,
      "wrong_image_confound": 6,
      "all_settings_failed": 5,
      "blank_image_confound": 1,
      "visual_helped": 1
    },
    "reasoning_world_knowledge": {
      "all_settings_failed": 7,
      "language_prior_or_question_bias": 3,
      "blank_image_confound": 3,
      "wrong_image_confound": 2,
      "visual_helped": 2
    },
    "spatial_localization": {
      "wrong_image_confound": 7,
      "all_settings_failed": 6,
      "visual_helped": 3,
      "language_prior_or_question_bias": 3,
      "blank_image_confound": 1
    },
    "weather_road_condition": {
      "all_settings_failed": 2
    }
  }
}
```

## Interpretation

- `visual_helped`: normal image/frame input beats all controls by the configured margin.
- `language_prior_or_question_bias`: text-only matches or beats normal, suggesting the question may be answerable without vision.
- `wrong_image_confound`: wrong-frame matches or beats normal, a strong warning against claiming stable visual grounding.
- `blank_image_confound`: blank image matches or beats normal, often caused by priors or answer/reference ambiguity.
- `all_settings_failed`: none of the settings reaches the pass threshold; these are the best candidates for manual review and data repair.

## Hardest Cases

| id | capability | normal_f1 | primary_error_type | secondary_flags | question | gold_answer | normal_answer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lingoqa_eval_000024 | object_recognition | 0.0 | language_prior_or_question_bias | object_recognition_failure;text_only_beats_or_ties_normal | Are there any traffic lights present? If yes, what color are they displaying? | No. | yes, yellow |
| lingoqa_eval_000031 | reasoning_world_knowledge | 0.0 | language_prior_or_question_bias | reasoning_failure;wrong_image_beats_or_ties_normal;text_only_beats_or_ties_normal | Are there any specific vehicles to which you should yield in this situation? If yes, why? | No. | yes |
| lingoqa_eval_000060 | other | 0.0 | language_prior_or_question_bias | text_only_beats_or_ties_normal | Do you need to come to a stop? If so, what is the reason for it? | No. | yes |
| lingoqa_eval_000072 | spatial_localization | 0.0 | language_prior_or_question_bias | spatial_failure;text_only_beats_or_ties_normal | Is it possible for you to accelerate in this situation, and if so, why? | It is not possible to accelerate as there are pedestrians crossing the road. | yes |
| lingoqa_eval_000044 | object_recognition | 0.0 | language_prior_or_question_bias | object_recognition_failure;text_only_beats_or_ties_normal | Are there any traffic lights? What color are they showing? | No traffic lights. | yes, red light |
| lingoqa_eval_000026 | reasoning_world_knowledge | 0.0 | language_prior_or_question_bias | reasoning_failure;text_only_beats_or_ties_normal | Is it possible for you to accelerate in this situation, and if so, why? | It is not possible to accelerate as there is a roundabout after the pedestrian crossing. | yes |
| lingoqa_eval_000021 | reasoning_world_knowledge | 0.0 | blank_image_confound | reasoning_failure;wrong_image_beats_or_ties_normal;text_only_beats_or_ties_normal | Why are you changing lane? | I am not changing lane. I am overtaking the cyclist within the current lane. | to avoid a car in front of me |
| lingoqa_eval_000019 | object_recognition | 0.0 | language_prior_or_question_bias | object_recognition_failure;text_only_beats_or_ties_normal | Are there any traffic lights present at this intersection? If so, what is the color displayed? | There are traffic lights displaying blinking amber. | yes |
| lingoqa_eval_000028 | counting | 0.0 | wrong_image_confound | counting_failure;wrong_image_beats_or_ties_normal | How many motorcyclists can you see? | I can see 1 motorcyclist. | 2 |
| lingoqa_eval_000022 | spatial_localization | 0.0 | wrong_image_confound | spatial_failure;wrong_image_beats_or_ties_normal | What is the current action and its justification? Answer in the form "action, justification". | The car maintains its speed and stays on it lane, because the lane is free of traffic and the left lane is a bus lane. | stop |
| lingoqa_eval_000078 | counting | 0.0 | wrong_image_confound | counting_failure;wrong_image_beats_or_ties_normal | How many parked vehicles can you see on this street? | Zero parked vehicles, but many stationary. | 12 |
| lingoqa_eval_000094 | other | 0.0 | wrong_image_confound | wrong_image_beats_or_ties_normal | What is the current action and its justification? Answer in the form "action, justification". | The car remains stopped and then accelerate slightly, as the traffic light turns green but pedestrians finish crossing the road at the pedes | stop |

## Strongest Control Confounds

| id | capability | visual_dependency_gap | primary_error_type | text_only_f1 | wrong_image_f1 | blank_image_f1 | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lingoqa_eval_000024 | object_recognition | -1.0 | language_prior_or_question_bias | 1.0 | 0.0 | 1.0 | Are there any traffic lights present? If yes, what color are they displaying? |
| lingoqa_eval_000031 | reasoning_world_knowledge | -1.0 | language_prior_or_question_bias | 1.0 | 1.0 | 1.0 | Are there any specific vehicles to which you should yield in this situation? If yes, why? |
| lingoqa_eval_000060 | other | -1.0 | language_prior_or_question_bias | 1.0 | 0.0 | 0.0 | Do you need to come to a stop? If so, what is the reason for it? |
| lingoqa_eval_000072 | spatial_localization | -0.5263 | language_prior_or_question_bias | 0.5263 | 0.0 | 0.0 | Is it possible for you to accelerate in this situation, and if so, why? |
| lingoqa_eval_000044 | object_recognition | -0.5 | language_prior_or_question_bias | 0.5 | 0.0 | 0.5 | Are there any traffic lights? What color are they showing? |
| lingoqa_eval_000026 | reasoning_world_knowledge | -0.4762 | language_prior_or_question_bias | 0.4762 | 0.0 | 0.0 | Is it possible for you to accelerate in this situation, and if so, why? |
| lingoqa_eval_000021 | reasoning_world_knowledge | -0.4 | blank_image_confound | 0.3333 | 0.375 | 0.4 | Why are you changing lane? |
| lingoqa_eval_000019 | object_recognition | -0.3636 | language_prior_or_question_bias | 0.3636 | 0.0 | 0.0 | Are there any traffic lights present at this intersection? If so, what is the color displayed? |
| lingoqa_eval_000028 | counting | -0.3333 | wrong_image_confound | 0.0 | 0.3333 | 0.0 | How many motorcyclists can you see? |
| lingoqa_eval_000081 | reasoning_world_knowledge | -0.3333 | wrong_image_confound | 0.0 | 0.4444 | 0.0 | Why are you stopping in this situation? |
| lingoqa_eval_000022 | spatial_localization | -0.3125 | wrong_image_confound | 0.0 | 0.3125 | 0.0 | What is the current action and its justification? Answer in the form "action, justification". |
| lingoqa_eval_000075 | other | -0.3087 | wrong_image_confound | 0.4118 | 0.6087 | 0.6087 | How is the car impacting your driving? |

## Positive Grounding Evidence

| id | capability | visual_dependency_gap | normal_f1 | control_max_f1 | question | gold_answer | normal_answer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lingoqa_eval_000013 | counting | 1.0 | 1.0 | 0.0 | How many cyclists are on the road? | 2 | 2 |
| lingoqa_eval_000064 | object_recognition | 1.0 | 1.0 | 0.0 | What is the color of traffic lights at this intersection? | Green. | green |
| lingoqa_eval_000029 | object_recognition | 0.8 | 0.8 | 0.0 | Are there any traffic lights present? If yes, what is the color of the lights? | Yes, red. | yes, red light |
| lingoqa_eval_000054 | object_recognition | 0.5 | 0.5 | 0.0 | Are there any traffic lights present? If so, what is the color displayed by the lights? | Yes, green. | yes, the traffic light is green. |
| lingoqa_eval_000005 | spatial_localization | 0.4286 | 0.4286 | 0.0 | What is the current action and its justification? Answer in the form "action, justification". | The car starts, because the traffic lights turn green. | the traffic light is green |
| lingoqa_eval_000076 | reasoning_world_knowledge | 0.3778 | 0.6 | 0.2222 | Why can you start in this instance? | Because the light turns green. | the traffic light is green |
| lingoqa_eval_000068 | counting | 0.3333 | 0.3333 | 0.0 | How many cyclists are there? | 1, in the left lane. | There are two cyclists in the image. |
| lingoqa_eval_000014 | object_recognition | 0.2857 | 0.2857 | 0.0 | Are there any traffic signals? If yes, what is the color displayed? | Yes, the traffic light is green. | yes |
| lingoqa_eval_000079 | object_recognition | 0.2857 | 0.2857 | 0.0 | Can you spot a traffic light? If yes, what color is being shown? | Yes. The traffic light is green. | yes |
| lingoqa_eval_000011 | spatial_localization | 0.2424 | 0.2424 | 0.0 | What is the current action and its justification? Answer in the form "action, justification". | The car accelerates slightly, follows the car ahead and crosses the zebra crossing, as no pedestrian is about to cross and the car ahead pul | the car is driving on the road. |
| lingoqa_eval_000063 | counting | 0.2222 | 0.2222 | 0.0 | How many pedestrians are present in this image? | There are 2 pedestrians, on the left sidewalk. | 2 |
| lingoqa_eval_000095 | spatial_localization | 0.2 | 0.2 | 0.0 | Are there any cyclists in this image? | Yes, there are 2 cyclists in the left lane. | yes |
