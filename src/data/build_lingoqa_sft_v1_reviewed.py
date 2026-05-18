"""Build conservative LingoQA SFT-v1 reviewed candidates.

This script is intentionally curation-heavy: it records the manual review and
rewrite decisions used for the SFT-v1 candidate JSONL. It does not train.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = ROOT / "data/processed/lingoqa_sft_candidates_needs_review.jsonl"
V0_KEEP = ROOT / "data/processed/lingoqa_sft_v0_keep_25.jsonl"
AGENT_REVIEW = ROOT / "outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv"
FIX_CSV = ROOT / "outputs/cases/lingoqa_sft_v0_need_fix_7.csv"
DROP_CSV = ROOT / "outputs/cases/lingoqa_sft_v0_drop_9.csv"

OUT_FIX = ROOT / "outputs/cases/lingoqa_sft_v1_fix_rewritten.csv"
OUT_REVIEWED = ROOT / "outputs/cases/lingoqa_sft_v1_reviewed_plan.csv"
OUT_JSONL = ROOT / "data/processed/lingoqa_sft_v1_balanced_candidate.jsonl"
OUT_SUMMARY = ROOT / "outputs/eval_results/lingoqa_sft_v1_balanced_candidate_summary.json"
OUT_REPORT = ROOT / "docs/lingoqa_sft_v1_review_report.md"


FIX_REWRITES: dict[str, dict[str, str]] = {
    "lingoqa_eval_000031_hard_negative_spatial_reasoning": {
        "answer": "No specific vehicle needs to be yielded to in the ego path.",
        "reason": "The current frames show the ego vehicle approaching a zebra crossing and junction area. No car, bus, or cyclist is visibly entering the ego lane with priority, but pedestrians are near the crossing, so the safe implication is to proceed slowly and keep watching the crossing.",
        "status": "rewritten_keep",
        "note": "Rewrote terse 'No' with visible road users and caution around the zebra crossing.",
    },
    "lingoqa_eval_000032_hard_negative_spatial_reasoning": {
        "answer": "Yes, reduce speed before the crossing.",
        "reason": "The ego vehicle is approaching a zebra crossing and junction. A pedestrian is at or moving across the crossing area, and a cyclist is visible near the right side of the intersection, so slowing gives both road users time and reduces conflict risk.",
        "status": "rewritten_keep",
        "note": "Grounded the slowdown in pedestrian/cyclist positions around the crossing.",
    },
    "lingoqa_eval_000066_hard_negative_spatial_reasoning": {
        "answer": "It is only safe to move forward slowly and cautiously.",
        "reason": "The ego vehicle is near a zebra crossing. Pedestrians are visible around the crossing in later frames, so the car should move only after confirming they are not entering the ego lane; the safety concern is pedestrian priority at the crossing.",
        "status": "rewritten_keep",
        "note": "Changed 'safe' to cautious movement tied to pedestrian position.",
    },
    "lingoqa_eval_000077_hard_negative_spatial_reasoning": {
        "answer": "Speed should be adjusted for the low-speed street environment.",
        "reason": "The frames show a narrow urban road with a visible 20 mph road marking, a bus-stop area, and roadside pedestrian activity. The ego action should be to maintain a modest speed so there is enough stopping margin if a pedestrian, cyclist, or vehicle appears near the stop or side street.",
        "status": "rewritten_keep",
        "note": "Original car-distance rationale was not visible; rewrote around visible speed marking and roadside context.",
    },
    "lingoqa_eval_000044_wrong_image_confound": {
        "answer": "No vehicle traffic lights are visible.",
        "reason": "The frames show a pedestrian crossing area with yellow beacon-style posts, but no red/amber/green vehicle traffic signal governing the ego lane. The safe interpretation is not to invent a traffic-light color from the image.",
        "status": "rewritten_keep",
        "note": "Kept as a negative visual-evidence sample, distinguishing beacons from vehicle traffic lights.",
    },
    "lingoqa_eval_000025_wrong_image_confound": {
        "answer": "",
        "reason": "",
        "status": "rewritten_drop",
        "note": "Dropped: 'safe to proceed' is too safety-critical and the visible signal/pedestrian context is not clear enough for a confident positive answer.",
    },
    "lingoqa_eval_000085_wrong_image_confound": {
        "answer": "No, there is no visible vehicle that the ego vehicle must yield to when going straight.",
        "reason": "The ego-facing signal is green and the visible vehicles are not crossing into the ego lane. The appropriate action is to continue straight with normal caution for the junction and any pedestrians near the crossing.",
        "status": "rewritten_keep",
        "note": "Rewrote terse negative answer with signal ownership and ego-path evidence.",
    },
}


MANUAL_KEEP: dict[str, dict[str, str]] = {
    "lingoqa_eval_000064_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "The traffic light is green.",
        "reason": "The signal visible ahead for the ego path is green across the reviewed frames, supporting a green-light answer from the current image evidence.",
    },
    "lingoqa_eval_000070_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Yes, pedestrians are crossing the road.",
        "reason": "Pedestrians are visible on and near the zebra crossing ahead of the ego vehicle, so the answer is grounded in their position across the road.",
    },
    "lingoqa_eval_000092_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "There is one parked car on the left.",
        "reason": "A dark parked vehicle is visible along the left curb beside the ego lane; the count is one for the parked car on that side.",
    },
    "lingoqa_eval_000056_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Yes, it is safe to enter with caution because the ego traffic light is green.",
        "reason": "The ego-facing traffic signal is green. A pedestrian is visible near the crossing area, so the safe implication is to enter only while continuing to monitor that pedestrian.",
    },
    "lingoqa_eval_000005_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Start moving, because the traffic light for the ego path is green.",
        "reason": "The frames show the ego vehicle at an intersection with a green signal ahead and no vehicle blocking the lane, so starting is visually supported.",
    },
    "lingoqa_eval_000061_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Enter only after the pedestrians have cleared the crossing.",
        "reason": "Pedestrians and crossing beacons are visible at the crossing ahead. The ego vehicle should not enter until the crossing is clear, even if the beacon phase permits cautious movement.",
    },
    "lingoqa_eval_000099_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Start and follow as the cyclist and bus ahead pull away.",
        "reason": "The bus is ahead in the ego lane and a cyclist is visible near the left side of the lane. The ego action is to start gently while preserving distance as those road users move away.",
    },
    "lingoqa_eval_000043_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Two pedestrians can be seen crossing the road.",
        "reason": "In the later frame, two pedestrians are visible on the crossing area ahead of the ego vehicle, so the count is two.",
    },
    "lingoqa_eval_000063_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "There are two pedestrians on the left sidewalk.",
        "reason": "Two people are visible on the left sidewalk in the reviewed frames, and they are not in the ego lane.",
    },
    "lingoqa_eval_000022_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "Maintain speed and stay in the current lane.",
        "reason": "The ego lane ahead appears clear, while the left lane is marked as a bus lane. Staying in the current lane avoids entering the bus lane unnecessarily.",
    },
    "lingoqa_eval_000047_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "There is no right-turn intersection immediately ahead.",
        "reason": "The visible road continues forward without a clear right-turn junction in front of the ego vehicle, so no right-turn positioning adjustment is supported by the image.",
    },
    "lingoqa_eval_000004_positive_grounding": {
        "role": "positive_visual_grounding",
        "answer": "You are stopping because the light is red and pedestrians are crossing.",
        "reason": "The scene shows a red signal and several pedestrians crossing in front of the ego vehicle. Stopping protects the pedestrians and obeys the signal.",
    },
    "lingoqa_eval_000026_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "No, do not accelerate.",
        "reason": "The ego vehicle is approaching a zebra crossing and a junction/roundabout area. A pedestrian is visible near the crossing, so the safe action is to keep speed low rather than accelerate.",
    },
    "lingoqa_eval_000072_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "No, it is not possible to accelerate.",
        "reason": "Pedestrians are crossing directly ahead at the zebra crossing. The ego action should be to slow or stop because accelerating would reduce safety margin for crossing pedestrians.",
    },
    "lingoqa_eval_000036_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "You can proceed because the ego traffic light is green and no pedestrian is crossing the ego path.",
        "reason": "The visible signal for the ego lane is green, and the crosswalk area ahead is not occupied in the ego path, so proceeding through the intersection is supported.",
    },
    "lingoqa_eval_000081_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "You are stopping because the traffic light ahead is red.",
        "reason": "The ego vehicle is approaching an intersection where the signal ahead is red. Stopping keeps the vehicle out of the junction until the signal permits movement.",
    },
    "lingoqa_eval_000011_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "Accelerate slightly and follow the car ahead across the zebra crossing.",
        "reason": "The lead car is moving away, the zebra crossing in the ego path is clear, and no pedestrian is about to step into the crossing, so a gentle start is visually supported.",
    },
    "lingoqa_eval_000052_hard_negative_spatial_reasoning": {
        "role": "spatial_reasoning_correction",
        "answer": "Start and accelerate gently as traffic ahead pulls away at the green light.",
        "reason": "The vehicles in front are moving through the junction and the traffic light is green. The ego action should be a cautious start while maintaining distance to the vehicle ahead.",
    },
    "lingoqa_eval_000018_all_settings_failed": {
        "role": "positive_visual_grounding",
        "answer": "Two pedestrians are crossing the road.",
        "reason": "Two pedestrians are visible on the crossing in front of the ego vehicle, so the count and location are directly supported by the frames.",
    },
    "lingoqa_eval_000035_all_settings_failed": {
        "role": "spatial_reasoning_correction",
        "answer": "Stop at the intersection.",
        "reason": "The ego vehicle reaches an intersection with a red traffic light and pedestrians crossing the road. Stopping prevents entering the crossing while pedestrians are in the path.",
    },
    "lingoqa_eval_000071_all_settings_failed": {
        "role": "spatial_reasoning_correction",
        "answer": "Accelerate only gently after the cyclist and bus ahead pull away.",
        "reason": "A cyclist is visible near the front-left of the ego lane and a bus is ahead. The ego vehicle should wait for them to move away and then follow with space.",
    },
    "lingoqa_eval_000001_all_settings_failed": {
        "role": "positive_visual_grounding",
        "answer": "There are zero cyclists visible.",
        "reason": "The reviewed frames show pedestrians and parked vehicles near the road, but no cyclist is visible in the ego scene.",
    },
    "lingoqa_eval_000058_all_settings_failed": {
        "role": "positive_visual_grounding",
        "answer": "There are no cyclists visible.",
        "reason": "Pedestrians are visible near the crossing area, but no bicycle rider is visible in the current frames.",
    },
    "lingoqa_eval_000074_all_settings_failed": {
        "role": "positive_visual_grounding",
        "answer": "No traffic light color is visible.",
        "reason": "The road ahead shows lane markings and surrounding traffic, but no visible vehicle traffic light governing the ego path, so no color should be claimed.",
    },
}

ANTI_KEEP = {
    "lingoqa_eval_000075_wrong_image_confound": (
        "The nearby car is not the main constraint; the ego vehicle needs to stop for the red light.",
        "The current frames show a red signal and pedestrians in the crossing area. The visible safety constraint is the signal and pedestrian crossing, not a vehicle cutting into the ego lane.",
    ),
    "lingoqa_eval_000049_wrong_image_confound": (
        "Yes, a green traffic light is visible.",
        "The traffic signal ahead for the ego path is green in the reviewed frames, so the answer should be based on that visible signal.",
    ),
    "lingoqa_eval_000009_wrong_image_confound": (
        "No, there is no traffic in the ego lane preventing forward movement.",
        "The ego lane ahead is clear in the current images; nearby parked or roadside vehicles do not block the ego path.",
    ),
    "lingoqa_eval_000021_wrong_image_confound": (
        "You are not changing lane; you are overtaking a cyclist within the current lane.",
        "A cyclist is visible close to the left/front of the ego vehicle while the ego lane continues ahead. The maneuver is to pass with lateral clearance rather than change lanes.",
    ),
    "lingoqa_eval_000023_wrong_image_confound": (
        "There are zero cyclists on the road.",
        "The reviewed frames show pedestrians near the crossing, but no cyclist is visible in the road scene.",
    ),
    "lingoqa_eval_000040_wrong_image_confound": (
        "No cyclists are visible.",
        "Vehicles and pedestrians are visible around the road, but there is no bicycle rider in the current images.",
    ),
    "lingoqa_eval_000062_wrong_image_confound": (
        "No buses are visible.",
        "The scene contains cyclists or pedestrians on the road, but no bus appears in the current frames.",
    ),
    "lingoqa_eval_000069_wrong_image_confound": (
        "Yes, the visible traffic lights are green.",
        "The ego-facing signal at the junction is green, so the color answer should be tied to the visible signal rather than a prior.",
    ),
    "lingoqa_eval_000090_wrong_image_confound": (
        "The road speed limit is 20.",
        "A 20 mph marking is painted in the ego lane, providing direct visual evidence for the speed limit.",
    ),
    "lingoqa_eval_000098_wrong_image_confound": (
        "No emergency vehicles are visible.",
        "The frames show ordinary road traffic and parked vehicles, but no ambulance, police, or fire vehicle is visible.",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def base_eval_id(sample_id: str) -> str:
    parts = sample_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else sample_id


def candidate_type(sample: dict[str, Any]) -> str:
    return str(sample.get("review", {}).get("candidate_type", ""))


def capability(sample: dict[str, Any]) -> str:
    return str(sample.get("meta", {}).get("external", {}).get("capability", sample.get("answer", {}).get("subcategory", "")))


def visual_gap(sample: dict[str, Any]) -> str:
    return str(sample.get("review", {}).get("visual_dependency_gap", ""))


def normal_f1(sample: dict[str, Any]) -> str:
    return str(sample.get("review", {}).get("normal_f1", ""))


def control_max_f1(sample: dict[str, Any]) -> str:
    return str(sample.get("review", {}).get("control_max_f1", ""))


def image_paths(sample: dict[str, Any]) -> list[str]:
    external = sample.get("meta", {}).get("external", {})
    paths = external.get("image_paths")
    if isinstance(paths, list) and paths:
        return [str(p) for p in paths]
    return [str(sample.get("image", ""))]


def make_sample(sample: dict[str, Any], role: str, answer: str, reason: str, review_note: str, source_status: str) -> dict[str, Any]:
    out = json.loads(json.dumps(sample, ensure_ascii=False))
    refs = list(out.get("answer", {}).get("references", []))
    if answer and answer not in refs:
        refs.insert(0, answer)
    out["answer"] = {
        "task": "external_vqa",
        "answer": answer,
        "references": refs,
        "category": "LingoQA",
        "subcategory": capability(sample),
        "reason": reason,
    }
    external = out.setdefault("meta", {}).setdefault("external", {})
    external["image_paths"] = image_paths(sample)
    out["meta"].update(
        {
            "task_type": "external_vqa",
            "source": "lingoqa_sft_v1",
            "benchmark_source": "lingoqa",
            "capability": capability(sample),
            "sft_v1_role": role,
            "curation_status": "reviewed_keep_sft_v1",
            "prompt_variant": "spatial",
            "frame_strategy": "first_middle_last",
            "max_images": 3,
            "source_status": source_status,
        }
    )
    out["review"] = {
        "final_decision": "keep_sft_v1",
        "review_note": review_note,
        "preferred_objective": "normal_vqa_sft",
    }
    return out


def main() -> None:
    candidates = read_jsonl(CANDIDATES)
    by_id = {row["id"]: row for row in candidates}
    v0_keep_ids = {row["id"] for row in read_jsonl(V0_KEEP)}
    agent_rows = {row["id"]: row for row in read_csv(AGENT_REVIEW)}
    fix_rows = {row["id"]: row for row in read_csv(FIX_CSV)}
    drop_ids = {row["id"] for row in read_csv(DROP_CSV)}

    fix_out = []
    for sample_id, fix in fix_rows.items():
        sample = by_id[sample_id]
        rw = FIX_REWRITES[sample_id]
        fix_out.append(
            {
                "id": sample_id,
                "candidate_type": fix["candidate_type"],
                "capability": fix["capability"],
                "question": fix["question"],
                "original_gold_answer": fix["gold_answer"],
                "rewritten_answer": rw["answer"],
                "rewritten_reason": rw["reason"],
                "rewrite_status": rw["status"],
                "rewrite_note": rw["note"],
                "image": fix["image"],
                "image_paths": fix["image_paths"],
                "source_fix_csv": str(FIX_CSV.relative_to(ROOT)),
            }
        )

    reviewed_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def add_keep(sample_id: str, role: str, answer: str, reason: str, note: str, source_status: str) -> None:
        sample = by_id[sample_id]
        used_ids.add(sample_id)
        jsonl_rows.append(make_sample(sample, role, answer, reason, note, source_status))

    for sample_id, payload in MANUAL_KEEP.items():
        source_status = "unreviewed_positive_manual_review"
        if sample_id in v0_keep_ids:
            source_status = "v0_keep_rewritten"
        if candidate_type(by_id[sample_id]) == "all_settings_failed":
            source_status = "all_settings_failed_manual_promote"
        add_keep(sample_id, payload["role"], payload["answer"], payload["reason"], "Manual image review accepted with grounded rewrite.", source_status)

    for sample_id, (answer, reason) in ANTI_KEEP.items():
        add_keep(sample_id, "anti_hallucination_counterfactual", answer, reason, "Kept as capped anti-hallucination visual-evidence sample.", "v0_keep_anti_capped_rewritten")

    for sample_id, rw in FIX_REWRITES.items():
        if rw["status"] != "rewritten_keep":
            continue
        sample = by_id[sample_id]
        add_keep(sample_id, "high_quality_fix_rewritten", rw["answer"], rw["reason"], rw["note"], "v0_fix_rewritten")

    for sample in candidates:
        sample_id = sample["id"]
        ctype = candidate_type(sample)
        action = agent_rows.get(sample_id, {}).get("human_action", "unreviewed")
        source_status = "unreviewed"
        if sample_id in v0_keep_ids:
            source_status = "v0_keep"
        if sample_id in fix_rows:
            source_status = "v0_fix"
        if sample_id in drop_ids:
            source_status = "v0_drop"
        if sample_id in used_ids:
            built = next(row for row in jsonl_rows if row["id"] == sample_id)
            decision = "keep_sft_v1"
            role = built["meta"]["sft_v1_role"]
            final_answer = built["answer"]["answer"]
            final_reason = built["answer"]["reason"]
            pref = "normal_vqa_sft"
            eligible = "true"
            note = built["review"]["review_note"]
            manual_rewrite = "true"
            source_status = built["meta"]["source_status"]
        elif sample_id in fix_rows and FIX_REWRITES[sample_id]["status"] == "rewritten_drop":
            decision = "drop_sft_v1"
            role = "high_quality_fix_rewritten"
            final_answer = ""
            final_reason = ""
            pref = "normal_vqa_sft"
            eligible = "false"
            note = FIX_REWRITES[sample_id]["note"]
            manual_rewrite = "true"
        elif sample_id in drop_ids or ctype in {"language_prior_confound", "mixed_review"}:
            decision = "drop_sft_v1"
            role = "exclude"
            final_answer = ""
            final_reason = ""
            pref = "none"
            eligible = "false"
            note = agent_rows.get(sample_id, {}).get("human_note", "Dropped by conservative v1 policy.")
            manual_rewrite = "false"
        elif ctype == "wrong_image_confound":
            decision = "keep_preference_later"
            role = "anti_hallucination_counterfactual"
            final_answer = ""
            final_reason = ""
            pref = "preference_or_rejection_later"
            eligible = "false"
            note = "Wrong-image confound not selected for ordinary SFT cap; better reserved for preference/rejection data."
            manual_rewrite = "false"
        else:
            decision = "needs_human_review"
            role = {
                "positive_grounding": "positive_visual_grounding",
                "hard_negative_spatial_reasoning": "spatial_reasoning_correction",
                "all_settings_failed": "manual_review_needed",
            }.get(ctype, "manual_review_needed")
            final_answer = ""
            final_reason = ""
            pref = "normal_vqa_sft_after_review"
            eligible = "false"
            note = "Not promoted because local image evidence was missing or not reviewed in this pass."
            manual_rewrite = "true"

        reviewed_rows.append(
            {
                "id": sample_id,
                "eval_case_id": base_eval_id(sample_id),
                "source_status": source_status,
                "candidate_type": ctype,
                "capability": capability(sample),
                "sft_v1_role": role,
                "sft_v1_action": action,
                "ordinary_sft_eligible": eligible,
                "preferred_objective": pref,
                "manual_rewrite_required": manual_rewrite,
                "final_decision": decision,
                "final_answer": final_answer,
                "final_reason": final_reason,
                "review_note": note,
                "priority_bucket": sample.get("review", {}).get("priority_bucket", ""),
                "visual_dependency_gap": visual_gap(sample),
                "normal_f1": normal_f1(sample),
                "control_max_f1": control_max_f1(sample),
                "question": sample.get("instruction", ""),
                "image": sample.get("image", ""),
                "image_paths": json.dumps(image_paths(sample), ensure_ascii=False),
            }
        )

    write_csv(
        OUT_FIX,
        fix_out,
        [
            "id",
            "candidate_type",
            "capability",
            "question",
            "original_gold_answer",
            "rewritten_answer",
            "rewritten_reason",
            "rewrite_status",
            "rewrite_note",
            "image",
            "image_paths",
            "source_fix_csv",
        ],
    )

    reviewed_fields = [
        "id",
        "eval_case_id",
        "source_status",
        "candidate_type",
        "capability",
        "sft_v1_role",
        "sft_v1_action",
        "ordinary_sft_eligible",
        "preferred_objective",
        "manual_rewrite_required",
        "final_decision",
        "final_answer",
        "final_reason",
        "review_note",
        "priority_bucket",
        "visual_dependency_gap",
        "normal_f1",
        "control_max_f1",
        "question",
        "image",
        "image_paths",
    ]
    write_csv(OUT_REVIEWED, reviewed_rows, reviewed_fields)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisions = Counter(row["final_decision"] for row in reviewed_rows)
    by_role = Counter(row["meta"]["sft_v1_role"] for row in jsonl_rows)
    by_capability = Counter(row["meta"]["capability"] for row in jsonl_rows)
    by_source = Counter(row["meta"]["source_status"] for row in jsonl_rows)
    by_pref = Counter(row["review"]["preferred_objective"] for row in jsonl_rows)
    target = {
        "total_50_80": 50 <= len(jsonl_rows) <= 80,
        "positive_visual_grounding_30_40": 30 <= by_role.get("positive_visual_grounding", 0) <= 40,
        "spatial_reasoning_correction_20_30": 20 <= by_role.get("spatial_reasoning_correction", 0) <= 30,
        "anti_hallucination_counterfactual_10_15": 10 <= by_role.get("anti_hallucination_counterfactual", 0) <= 15,
        "high_quality_fix_rewritten_7_15": 7 <= by_role.get("high_quality_fix_rewritten", 0) <= 15,
    }
    contamination_risk = {
        "risk": "high",
        "reason": "The available candidate pool is mined from the 100-case LingoQA eval/control subset; using it for training can contaminate later evaluation on the same cases. Hold out a fresh LingoQA split for post-SFT evaluation.",
    }
    summary = {
        "input_files": [
            str(CANDIDATES.relative_to(ROOT)),
            str(V0_KEEP.relative_to(ROOT)),
            str(AGENT_REVIEW.relative_to(ROOT)),
            str(FIX_CSV.relative_to(ROOT)),
            str(DROP_CSV.relative_to(ROOT)),
        ],
        "missing_requested_inputs_in_local_workspace": [
            "docs/lingoqa_sft_v0_failure_analysis.md",
            "docs/lingoqa_sft_v1_data_plan.md",
            "outputs/cases/lingoqa_sft_v1_candidate_plan.csv",
            "outputs/eval_results/lingoqa_sft_v1_candidate_plan_summary.json",
            "src/data/build_lingoqa_sft_v1_plan.py",
        ],
        "total_samples": len(jsonl_rows),
        "reviewed_plan_rows": len(reviewed_rows),
        "by_sft_v1_role": dict(by_role),
        "by_capability": dict(by_capability),
        "by_source_status": dict(by_source),
        "by_preferred_objective": dict(by_pref),
        "reviewed_plan_decisions": dict(decisions),
        "positive_visual_grounding": by_role.get("positive_visual_grounding", 0),
        "spatial_reasoning_correction": by_role.get("spatial_reasoning_correction", 0),
        "anti_hallucination_counterfactual": by_role.get("anti_hallucination_counterfactual", 0),
        "high_quality_fix_rewritten": by_role.get("high_quality_fix_rewritten", 0),
        "target_ratio_status": target,
        "target_ratio_reached": all(target.values()),
        "eval_contamination_risk": contamination_risk,
        "recommend_smoke_training": False,
        "smoke_training_recommendation_reason": f"Do not train yet: the ordinary SFT set has only {len(jsonl_rows)} reviewed samples, below the 50-80 target, and is still positive/spatial-light.",
        "ssh_server_verification": "not_verified: SSH publickey/password auth was unavailable in this run, so outputs were built from the local project workspace.",
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# LingoQA SFT-v1 Review Report

## Scope

This pass rebuilt a conservative SFT-v1 reviewed plan from the locally available project files because the requested server-only v1 candidate plan was not present in the workspace and SSH authentication was unavailable during this run.

## Inputs Read

- `docs/lingoqa_sft_v0_build_report.md`
- `docs/lingoqa_sft_v0_adapter_eval_report.md`
- `docs/lingoqa_best_strategy_visual_control_report.md`
- `docs/lingoqa_bad_case_analysis.md`
- `docs/lingoqa_sft_candidate_curation.md`
- `outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv`
- `outputs/cases/lingoqa_sft_v0_need_fix_7.csv`
- `outputs/cases/lingoqa_sft_v0_drop_9.csv`
- `data/processed/lingoqa_sft_candidates_needs_review.jsonl`
- `data/processed/lingoqa_sft_v0_keep_25.jsonl`
- `src/train_qwen25vl_lora.py`
- `src/eval/base_infer_qwen25vl.py`

Requested but missing locally:

- `docs/lingoqa_sft_v0_failure_analysis.md`
- `docs/lingoqa_sft_v1_data_plan.md`
- `outputs/cases/lingoqa_sft_v1_candidate_plan.csv`
- `outputs/eval_results/lingoqa_sft_v1_candidate_plan_summary.json`
- `src/data/build_lingoqa_sft_v1_plan.py`

## Manual Review Policy

Positive and spatial samples were promoted only when the visible frames supported the answer and the final answer/reason could be rewritten with explicit visual evidence. Missing-image rows were not promoted. Wrong-image confounds were capped and rewritten as current-image visual-evidence samples; the remainder were marked for later preference/rejection use.

## Fix-7 Decisions

| id | decision | note |
|---|---|---|
| lingoqa_eval_000031_hard_negative_spatial_reasoning | rewritten_keep | No yield vehicle visible; rewrote with zebra-crossing pedestrian caution. |
| lingoqa_eval_000032_hard_negative_spatial_reasoning | rewritten_keep | Slowing grounded in pedestrian/cyclist positions near crossing. |
| lingoqa_eval_000066_hard_negative_spatial_reasoning | rewritten_keep | Safe only with slow cautious movement after checking crossing pedestrians. |
| lingoqa_eval_000077_hard_negative_spatial_reasoning | rewritten_keep | Replaced unsupported car-distance rationale with visible 20 mph/bus-stop context. |
| lingoqa_eval_000044_wrong_image_confound | rewritten_keep | Negative traffic-light answer grounded by distinguishing crossing beacons from vehicle lights. |
| lingoqa_eval_000025_wrong_image_confound | rewritten_drop | Dropped because safe-to-proceed answer is not visually clear enough. |
| lingoqa_eval_000085_wrong_image_confound | rewritten_keep | No yield vehicle visible; green ego signal and clear ego path noted. |

## Output Distribution

- Total ordinary SFT JSONL samples: {len(jsonl_rows)}
- By role: `{dict(by_role)}`
- By capability: `{dict(by_capability)}`
- By source status: `{dict(by_source)}`
- Reviewed plan decisions: `{dict(decisions)}`

## Target Ratio Check

- total 50-80: {target["total_50_80"]}
- positive_visual_grounding 30-40: {target["positive_visual_grounding_30_40"]}
- spatial_reasoning_correction 20-30: {target["spatial_reasoning_correction_20_30"]}
- anti_hallucination_counterfactual 10-15: {target["anti_hallucination_counterfactual_10_15"]}
- high_quality_fix_rewritten 7-15: {target["high_quality_fix_rewritten_7_15"]}

The target ratio is **not reached**. The set is too small and still short on positive visual grounding and spatial correction. This is preferable to padding with unreviewed or weakly grounded examples.

## Contamination Risk

Eval contamination risk is **high** because the available candidate pool is mined from the 100-case LingoQA eval/control subset. If this JSONL is trained, later evaluation should use a fresh LingoQA split or clearly report that the original 100-case control set is contaminated.

## Recommendation

Do **not** start smoke training yet. First recover the server-side v1 candidate plan, review the missing positive rows with images available, and mine additional spatial correction examples from a clean split.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
