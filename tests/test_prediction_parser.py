
from src.eval.prediction_parser import parse_prediction


def test_valid_json_answer_reason():
    out = parse_prediction('{"answer":"yes","reason":"because"}')
    assert out["answer_text"] == "yes"
    assert out["reason_text"] == "because"
    assert out["parse_success"] is True
    assert out["format_type"] == "json"


def test_json_like_single_quotes():
    out = parse_prediction("{'answer': 'no', 'reason': 'none'}")
    assert out["answer_text"] == "no"
    assert out["format_type"] == "json_like"


def test_answer_only_json():
    out = parse_prediction('{"answer":"stop"}')
    assert out["answer_text"] == "stop"
    assert out["reason_text"] == ""


def test_plain_text():
    out = parse_prediction("The car should stop.")
    assert out["answer_text"] == "The car should stop."
    assert out["format_type"] == "plain_text"


def test_damaged_json_falls_back_failed():
    out = parse_prediction('{"answer": }')
    assert out["parse_success"] is False
    assert out["answer_text"] == '{"answer": }'
