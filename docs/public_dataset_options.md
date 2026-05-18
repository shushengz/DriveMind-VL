# Public Dataset Options For DriveMind-VL

DriveMind-VL should not rely on synthetic data as the main evidence. Since IntelliCockpitBench currently exposes only a 7-sample public GitHub subset in this workspace, the next practical path is a hybrid public-data evaluation plan.

## Recommended Priority

| Priority | Dataset | Best Use In DriveMind-VL | Access / Size Notes | Caveat |
|---|---|---|---|---|
| P0 | LingoQA | External video VQA, visual grounding, answer truthfulness | Official repo provides benchmark instructions and a 500-question evaluation dataset link; paper reports 28K short video scenarios and 419K annotations | Mostly front-view driving, not cabin/tool control |
| P0 | DriveBench | Reliability eval with clean/corrupted/text-only settings | Official repo describes 19,200 frames and 20,498 QA pairs across 17 settings | Strong for robustness, less for cockpit services |
| P0 | Talk2CarSlim | User command grounding and referred object localization | Official repo says slim setup needs under 2GB, full nuScenes version needs 300GB+ | Requires bbox/IoU style eval; not a chat/tool-call dataset |
| P1 | Drive&Act | Cabin behavior, driver/passenger action understanding | Public site says login restrictions removed; 12h video, 5 views, NIR/Depth/RGB, 83 hierarchical labels | Activity labels must be converted to QA/state labels |
| P1 | 100-Driver | Distracted driver classification and cabin state robustness | Project page reports 470K+ images, 4 cameras, 100 drivers, 79 hours, licensing form required | Access requires form; not VQA by default |
| P1 | DMD Driver Monitoring Dataset | Driver monitoring: distraction, drowsiness, gaze, hand state | Academic/scientific access by form; 37 volunteers, 3 in-cabin cameras, RGB/IR/depth, 50+ tags | Large data package; access is not instant |
| P1 | BDD-X | Driving explanation and risk/action reasoning | Official repo for textual explanations for self-driving vehicles | Needs BDD assets; explanation style, not direct VQA |
| P2 | NuScenes-QA | Large-scale driving VQA from nuScenes annotations | Paper reports 34K scenes and 460K QA pairs | nuScenes data can be large; generated from templates |
| P2 | DriveLM | Graph VQA for perception/prediction/planning | Official repo provides DriveLM-Data over nuScenes/CARLA and evaluation pipeline | More autonomous-driving planning than cockpit |
| P2 | DADA-2000 / VRU-Accident captions | Accident/risk explanation and anticipation | DADA-2000 has 2,000 accident clips and 658K+ frames; VRU-Accident has dense caption metadata on Hugging Face | Video/risk task conversion needed; license/access must be checked |

## What To Use First

1. LingoQA: best immediate replacement for the missing IntelliCockpitBench scale. Map it to `external_vqa` and keep normal/text-only/wrong-image/blank-image controls.
2. DriveBench: best for reliability and visual-dependency analysis because it already emphasizes clean, corrupted, and text-only settings.
3. Talk2CarSlim: best for user-command grounding. It can become `external_grounding` with bbox output and IoU metric.
4. Drive&Act: best first public source for `cabin_understanding`.

## DriveMind Mapping

| DriveMind Task | Public Data Source |
|---|---|
| `external_vqa` | LingoQA, DriveBench, NuScenes-QA, DriveLM |
| `risk_reasoning` | BDD-X, DADA-2000, DriveBench risk subsets |
| `cabin_understanding` | Drive&Act, 100-Driver, DMD |
| `tool_call` | Mostly synthetic/human-reviewed DriveMind data |
| `safety_rejection` | Mostly synthetic/human-reviewed DriveMind data |
| `external_grounding` | Talk2CarSlim |

## Immediate Implementation Plan

1. Add a generic external dataset registry.
2. Add LingoQA audit/conversion scripts first.
3. Add Talk2CarSlim adapter second if download is manageable.
4. Add Drive&Act label-to-QA conversion for cabin understanding.
5. Keep IntelliCockpitBench as preferred cockpit benchmark, but do not block progress on its full release.

## Sample Size Guidance

50-100 samples are enough for the first adapter smoke test and failure taxonomy. They are not enough for a final claim. For LingoQA, the next credible targets are:

- 100 samples for first engineering report.
- 100 samples with actual videos or extracted frames for first visual-grounding report.
- 500 official benchmark answers for the main external VQA score.
- Full evaluation split if storage and video access allow.

## Sources

- IntelliCockpitBench: https://github.com/Lane315/IntelliCockpitBench
- LingoQA: https://github.com/wayveai/LingoQA
- DriveBench: https://github.com/worldbench/DriveBench
- Talk2Car: https://github.com/talk2car/Talk2Car
- Drive&Act: https://driveandact.com/
- 100-Driver: https://100-driver.github.io/
- DMD: https://www.vicomtech.org/en/news/detail/562_vicomtech-develops-the-driver-monitoring-dataset-dmd-an-opensource-driver-monitoring-system-now-available-for-academic-and-scientific-use
- BDD-X: https://github.com/JinkyuKimUCB/BDD-X-dataset
- NuScenes-QA: https://github.com/qiantianwen/NuScenes-QA
- DriveLM: https://github.com/OpenDriveLab/DriveLM
- DADA-2000 paper: https://arxiv.org/abs/1904.12634
