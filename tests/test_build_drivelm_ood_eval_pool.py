from src.eval.build_drivelm_ood_eval_pool import classify_question, validate_group


def row(setting, paths, labels):
    return {"id": "d1", "mode": "strict_visual", "setting": setting, "image_paths": paths, "image_labels": labels, "prompt": "question", "question": "question", "gold": "answer"}


def test_valid_control_group_passes_structural_checks():
    group = {
        "normal": row("normal", ["front.jpg"], ["[CAM_FRONT]"]),
        "text_only": row("text_only", [], []),
        "wrong_image": row("wrong_image", ["other.jpg"], ["[CAM_FRONT]"]),
        "blank_image": row("blank_image", ["blank.jpg"], ["[CAM_FRONT]"]),
    }
    assert validate_group(group) == []


def test_invalid_visual_controls_are_rejected():
    group = {
        "normal": row("normal", ["front.jpg"], ["[CAM_FRONT]"]),
        "text_only": row("text_only", ["front.jpg"], []),
        "wrong_image": row("wrong_image", ["front.jpg"], ["[CAM_FRONT]"]),
        "blank_image": row("blank_image", ["front.jpg"], ["[CAM_FRONT]"]),
    }
    errors = validate_group(group)
    assert "text_only_has_images" in errors
    assert "wrong_image_matches_normal" in errors
    assert "blank_image_not_placeholder" in errors


def test_question_taxonomy():
    assert classify_question("What is at <c2,CAM_BACK,12,14>?") == "object_token"
    assert classify_question("How many cars are visible?") == "counting"
