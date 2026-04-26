from app.ml.scoring.scoring import compute_score

NO_ISSUES = {
    "blurry": 0,
    "low_light": 0,
    "cluttered": 0,
    "bad_framing": 0,
    "tilted": 0,
    "poor_space_visibility": 0,
    "watermark": 0,
}


def test_no_defects_gives_score_100():
    result = compute_score(NO_ISSUES, "bedroom")
    assert result["score"] == 100


def test_no_defects_gives_grade_a():
    result = compute_score(NO_ISSUES, "bedroom")
    assert result["grade"] == "A"


def test_score_decreases_with_defect():
    issues = {**NO_ISSUES, "blurry": 1}
    result = compute_score(issues, "bedroom")
    assert result["score"] < 100


def test_score_minimum_is_zero():
    all_issues = {k: 1 for k in NO_ISSUES}
    result = compute_score(all_issues, "bedroom")
    assert result["score"] >= 0


def test_score_maximum_is_100():
    result = compute_score(NO_ISSUES, "bedroom")
    assert result["score"] <= 100


def test_all_defects_give_grade_f():
    all_issues = {k: 1 for k in NO_ISSUES}
    result = compute_score(all_issues, "bedroom")
    assert result["grade"] == "F"


def test_room_type_penalty_applied():
    # exterior has a -5 penalty vs neutral room
    base = compute_score(NO_ISSUES, "bedroom")["score"]
    penalised = compute_score(NO_ISSUES, "exterior")["score"]
    assert penalised < base


def test_room_type_bonus_applied():
    # Add a defect so the base score is below 100 and the +5 bonus is visible
    issues = {**NO_ISSUES, "tilted": 1}
    base = compute_score(issues, "bedroom")["score"]
    boosted = compute_score(issues, "bathroom")["score"]
    assert boosted > base
