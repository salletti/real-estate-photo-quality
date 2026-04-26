SUGGESTIONS: dict[str, list[str]] = {
    "blurry": [
        "Improve image sharpness",
        "Use a tripod or stabilize the camera",
    ],
    "low_light": [
        "Increase brightness",
        "Take the photo during the day",
    ],
    "cluttered": [
        "Declutter the space",
        "Remove personal items",
    ],
    "bad_framing": [
        "Improve framing",
        "Center important elements",
    ],
    "tilted": [
        "Straighten the image",
    ],
    "poor_space_visibility": [
        "Take the photo from further away",
        "Show the entire room",
    ],
    "watermark": [
        "Remove the watermark",
        "Use original image without logo",
    ],
}


def generate_suggestions(issues: dict[str, int]) -> list[str]:
    return [
        suggestion
        for issue, active in issues.items()
        if active and issue in SUGGESTIONS
        for suggestion in SUGGESTIONS[issue]
    ]
