from src.data.visual_control_formatter import MODE_AUGMENTED, MODE_STRICT, build_variants, parse_question_cameras


def lingo_row(idx="a", seg="s1"):
    return {
        "id": idx,
        "instruction": "What color is the traffic light?",
        "image": f"{seg}/0.jpg",
        "perception": {"objects": ["secret"]},
        "answer": {"answer": "green"},
        "meta": {"benchmark_source": "lingoqa", "external": {"image_paths": [f"{seg}/0.jpg", f"{seg}/1.jpg"], "capability": "object"}},
    }


def drivelm_row(question="What is visible in CAM_BACK?", idx="d1"):
    return {
        "id": idx,
        "instruction": question,
        "perception": {"objects": {"<c3,CAM_BACK,1,2>": {"Category": "Vehicle"}}},
        "answer": {"answer": "a vehicle"},
        "meta": {"benchmark_source": "drivelm", "external": {"image_paths": ["x/CAM_FRONT/a.jpg", "x/CAM_FRONT_LEFT/a.jpg", "x/CAM_FRONT_RIGHT/a.jpg", "x/CAM_BACK/a.jpg"]}},
    }


def test_strict_visual_mode_drops_perception():
    variants = build_variants(lingo_row(), [lingo_row(), lingo_row("b", "s2")], "lingoqa", MODE_STRICT)
    assert "perception" not in variants["normal"]["metadata"]
    assert variants["normal"]["mode"] == MODE_STRICT


def test_text_only_has_no_images():
    variants = build_variants(lingo_row(), [lingo_row(), lingo_row("b", "s2")], "lingoqa", MODE_STRICT)
    assert variants["text_only"]["image_paths"] == []


def test_blank_image_uses_placeholder():
    variants = build_variants(lingo_row(), [lingo_row(), lingo_row("b", "s2")], "lingoqa", MODE_STRICT, blank_image="blank.jpg")
    assert variants["blank_image"]["image_paths"] == ["blank.jpg", "blank.jpg"]


def test_wrong_image_differs_from_original():
    variants = build_variants(lingo_row(), [lingo_row(), lingo_row("b", "s2")], "lingoqa", MODE_STRICT)
    assert variants["wrong_image"]["image_paths"] != variants["normal"]["image_paths"]


def test_drivelm_camera_selection_from_question_and_object_token():
    assert parse_question_cameras("Use <c3,CAM_BACK,1,2> please")[0] == "CAM_BACK"
    assert parse_question_cameras("Look at CAM_FRONT_LEFT")[0] == "CAM_FRONT_LEFT"
    variants = build_variants(drivelm_row("Use <c3,CAM_BACK,1,2> please"), [drivelm_row(), drivelm_row(idx="d2")], "drivelm", MODE_STRICT)
    assert variants["normal"]["image_labels"] == ["[CAM_BACK]"]
    assert "objects" not in str(variants["normal"]["metadata"])


def test_output_fields_complete_and_augmented_marked():
    variants = build_variants(lingo_row(), [lingo_row(), lingo_row("b", "s2")], "lingoqa", MODE_AUGMENTED)
    row = variants["normal"]
    for key in ["id", "dataset", "mode", "setting", "question", "gold", "image_paths", "image_labels", "prompt", "metadata"]:
        assert key in row
    assert row["mode"] == MODE_AUGMENTED
    assert "perception" in row["metadata"]
