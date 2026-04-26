from __future__ import annotations

PENALTIES: dict[str, int] = {
    "blurry": -25,
    "poor_space_visibility": -25,
    "bad_framing": -15,
    "low_light": -15,
    "cluttered": -15,
    "tilted": -10,
    "watermark": -10,
}

ROOM_TYPE_ADJUSTMENTS: dict[str, int] = {
    "bathroom": +5,
    "attic": +5,
    "exterior": -5,
    "garden": -5,
    "pool": -5,
}

GRADES: list[tuple[int, str]] = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (45, "D"),
    (30, "E"),
    (0,  "F"),
]


def _get_grade(score: int) -> str:
    for threshold, grade in GRADES:
        if score >= threshold:
            return grade
    return "F"


def compute_score(labels: dict[str, int], room_type: str) -> dict[str, int | str]:
    score = 100
    score += sum(PENALTIES[label] for label, active in labels.items() if active and label in PENALTIES)
    score += ROOM_TYPE_ADJUSTMENTS.get(room_type, 0)
    score = max(0, min(100, score))

    return {"score": score, "grade": _get_grade(score)}
