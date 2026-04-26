import os

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

ISSUE_LABELS_FR = {
    "blurry": "flou",
    "low_light": "manque de luminosité",
    "cluttered": "espace encombré",
    "bad_framing": "mauvais cadrage",
    "tilted": "image penchée",
    "poor_space_visibility": "mauvaise visibilité de l'espace",
    "watermark": "filigrane visible",
}


def _build_prompt(active_issues: list[str]) -> str:
    items = "\n".join(f"- {ISSUE_LABELS_FR[issue]}" for issue in active_issues)
    return (
        f"Une photo immobilière présente les défauts suivants :\n{items}\n\n"
        "Donne des conseils simples, concrets et naturels en français pour améliorer cette photo. "
        "Réponds en 3 à 5 phrases maximum."
    )


def generate_human_suggestions(issues: dict[str, int]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    active_issues = [label for label, active in issues.items() if active and label in ISSUE_LABELS_FR]

    if not active_issues:
        return "Cette photo ne présente aucun défaut détecté."

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": _build_prompt(active_issues)}],
        max_tokens=300,
        temperature=0.7,
    )

    return response.choices[0].message.content
