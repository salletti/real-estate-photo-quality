from app.ml.suggestions.suggestions import generate_suggestions

NO_ISSUES = {
    "blurry": 0,
    "low_light": 0,
    "cluttered": 0,
    "bad_framing": 0,
    "tilted": 0,
    "poor_space_visibility": 0,
    "watermark": 0,
}


def test_no_issues_returns_empty_list():
    result = generate_suggestions(NO_ISSUES)
    assert result == []


def test_single_issue_returns_suggestions():
    issues = {**NO_ISSUES, "blurry": 1}
    result = generate_suggestions(issues)
    assert len(result) > 0


def test_multiple_issues_combine_suggestions():
    issues = {**NO_ISSUES, "blurry": 1, "low_light": 1}
    combined = generate_suggestions(issues)
    only_blurry = generate_suggestions({**NO_ISSUES, "blurry": 1})
    only_low_light = generate_suggestions({**NO_ISSUES, "low_light": 1})
    assert len(combined) == len(only_blurry) + len(only_low_light)


def test_suggestions_are_strings():
    issues = {**NO_ISSUES, "watermark": 1}
    result = generate_suggestions(issues)
    assert all(isinstance(s, str) for s in result)
