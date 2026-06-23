"""
utils.py – Core logic for e-K Cha Label?
Covers: OCR, NLP cleaning, harmful-ingredient detection,
        nutritional parsing, health scoring, ML classification,
        grade assignment and AI explanation generation.
"""

from __future__ import annotations
import re
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR  (EasyOCR)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(image: Image.Image, languages: list[str] | None = None) -> str:
    """Run EasyOCR on a PIL image and return joined text."""
    import easyocr

    if languages is None:
        languages = ["en"]

    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    img_np = np.array(image)
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return " ".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLP  (spaCy)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove stop-words, keep only alphabetic tokens."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha and len(token.text) > 2
        ]
        return " ".join(tokens)
    except Exception:
        # Fallback: simple regex cleaning if spaCy unavailable
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Harmful-ingredient detection
# ─────────────────────────────────────────────────────────────────────────────

HARMFUL_INGREDIENTS: list[str] = [
    "aspartame",
    "msg",
    "monosodium glutamate",
    "sodium benzoate",
    "trans fat",
    "tartrazine",
    "sodium nitrite",
    "high fructose corn syrup",
    "bha",
    "bht",
    "potassium bromate",
    "saccharin",
    "acesulfame",
    "carrageenan",
    "partially hydrogenated",
    "artificial colour",
    "artificial color",
    "artificial flavour",
    "artificial flavor",
]

# Aliases that map back to canonical names
_ALIASES: dict[str, str] = {
    "monosodium glutamate": "msg",
    "partially hydrogenated": "trans fat",
    "artificial colour": "artificial colour",
    "artificial color": "artificial colour",
    "artificial flavour": "artificial flavour",
    "artificial flavor": "artificial flavour",
}


def detect_harmful_ingredients(text: str) -> list[str]:
    """Return list of harmful ingredients found in *text* (lowercased)."""
    text_lower = text.lower()
    found: set[str] = set()
    for ingredient in HARMFUL_INGREDIENTS:
        if ingredient in text_lower:
            canonical = _ALIASES.get(ingredient, ingredient)
            found.add(canonical)
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nutritional value parsing
# ─────────────────────────────────────────────────────────────────────────────

_NUTRIENT_PATTERNS: dict[str, list[str]] = {
    "calories":   [r"(?:energy|calories?|kcal)[^\d]*(\d+(?:\.\d+)?)"],
    "sugar":      [r"(?:sugar)[^\d]*(\d+(?:\.\d+)?)"],
    "fat":        [r"(?:total\s*fat|fat)[^\d]*(\d+(?:\.\d+)?)"],
    "sodium":     [r"(?:sodium|salt)[^\d]*(\d+(?:\.\d+)?)"],
    "protein":    [r"(?:protein)[^\d]*(\d+(?:\.\d+)?)"],
    "carbs":      [r"(?:carbohydrate|carbs?)[^\d]*(\d+(?:\.\d+)?)"],
    "fibre":      [r"(?:dietary\s*fibre|fibre|fiber)[^\d]*(\d+(?:\.\d+)?)"],
    "saturated":  [r"(?:saturated\s*fat|saturates)[^\d]*(\d+(?:\.\d+)?)"],
}


def parse_nutritional_values(raw_text: str) -> dict[str, float]:
    """Extract numeric nutritional values from OCR text using regex."""
    text_lower = raw_text.lower()
    result: dict[str, float] = {k: 0.0 for k in _NUTRIENT_PATTERNS}

    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    result[nutrient] = float(match.group(1))
                    break
                except ValueError:
                    pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Health scoring  (weighted formula)
# ─────────────────────────────────────────────────────────────────────────────

# Penalty thresholds (per 100 g / ml)
_THRESHOLDS = {
    "sugar":    {"low": 5,  "high": 22.5},   # g
    "fat":      {"low": 3,  "high": 17.5},   # g
    "sodium":   {"low": 0.3,"high": 1.5},    # g
    "calories": {"low": 40, "high": 400},    # kcal
}

WEIGHTS = {"sugar": 0.30, "fat": 0.25, "additives": 0.25, "calories": 0.20}


def _penalty(value: float, low: float, high: float) -> float:
    """Return a 0-1 penalty where 0 = healthy and 1 = very unhealthy."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def compute_health_score(nutrition: dict[str, float], additive_count: int) -> int:
    """
    Compute health score 0-100 (higher = healthier).
    Uses weighted penalties for sugar, fat, calories, and additive count.
    """
    sugar_pen    = _penalty(nutrition.get("sugar", 0),
                            _THRESHOLDS["sugar"]["low"], _THRESHOLDS["sugar"]["high"])
    fat_pen      = _penalty(nutrition.get("fat", 0),
                            _THRESHOLDS["fat"]["low"], _THRESHOLDS["fat"]["high"])
    calorie_pen  = _penalty(nutrition.get("calories", 0),
                            _THRESHOLDS["calories"]["low"], _THRESHOLDS["calories"]["high"])
    additive_pen = min(additive_count / 5.0, 1.0)   # 5+ additives → max penalty

    total_penalty = (
        WEIGHTS["sugar"]     * sugar_pen +
        WEIGHTS["fat"]       * fat_pen +
        WEIGHTS["additives"] * additive_pen +
        WEIGHTS["calories"]  * calorie_pen
    )

    score = int(round((1 - total_penalty) * 100))
    return max(0, min(100, score))


def get_grade(score: int) -> str:
    """Map 0-100 score to Nutri-score style A-E grade."""
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "E"


def classify_health(score: int) -> str:
    """Map score to risk label."""
    if score >= 70: return "✅ Healthy"
    if score >= 40: return "⚠️ Moderate"
    return "🚨 Unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# 6. AI Explanation (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    score: int,
    risk: str,
    harmful: list[str],
    nutrition: dict[str, float],
) -> str:
    """Generate a markdown-formatted, plain-language health summary."""

    lines: list[str] = []

    # Overall verdict
    if score >= 70:
        lines.append(
            f"### ✅ Overall: This product appears **relatively healthy** (score {score}/100)\n"
        )
        lines.append(
            "The nutritional profile looks acceptable. Enjoy in moderation as part of a balanced diet."
        )
    elif score >= 40:
        lines.append(
            f"### ⚠️ Overall: This product has **moderate health concerns** (score {score}/100)\n"
        )
        lines.append(
            "It is suitable for occasional consumption but should not form a major part of your daily diet."
        )
    else:
        lines.append(
            f"### 🚨 Overall: This product is **NOT recommended** for regular consumption (score {score}/100)\n"
        )
        lines.append(
            "High levels of sugar, fat, or harmful additives make this product a poor dietary choice."
        )

    lines.append("\n---\n")

    # Sugar
    sugar = nutrition.get("sugar", 0)
    if sugar > 22.5:
        lines.append(f"🔴 **High sugar content** ({sugar}g/100g) — well above the recommended limit of 22.5g. "
                     "Regular consumption significantly raises risk of diabetes and tooth decay.")
    elif sugar > 5:
        lines.append(f"🟡 **Moderate sugar** ({sugar}g/100g) — keep overall daily sugar intake below 30g.")
    elif sugar > 0:
        lines.append(f"🟢 **Low sugar** ({sugar}g/100g) — within healthy limits.")

    # Fat
    fat = nutrition.get("fat", 0)
    if fat > 17.5:
        lines.append(f"🔴 **High fat content** ({fat}g/100g) — exceeds recommended threshold. "
                     "Watch total daily fat intake.")
    elif fat > 3:
        lines.append(f"🟡 **Moderate fat** ({fat}g/100g).")
    elif fat > 0:
        lines.append(f"🟢 **Low fat** ({fat}g/100g).")

    # Sodium
    sodium = nutrition.get("sodium", 0)
    if sodium > 1.5:
        lines.append(f"🔴 **High sodium** ({sodium}g/100g) — may contribute to hypertension.")
    elif sodium > 0.3:
        lines.append(f"🟡 **Moderate sodium** ({sodium}g/100g).")

    # Additives
    if harmful:
        lines.append(
            f"\n⚠️ **{len(harmful)} potentially harmful additive(s) detected:** "
            + ", ".join(h.title() for h in harmful) + "."
        )
        if "trans fat" in harmful or "sodium benzoate" in harmful or "potassium bromate" in harmful:
            lines.append("🔴 One or more **high-risk additives** found. These have been linked to serious "
                         "health conditions including cardiovascular disease and certain cancers.")
    else:
        lines.append("\n🟢 **No major harmful additives detected** in the extracted text.")

    # Recommendation
    lines.append("\n---\n")
    lines.append("**💡 Recommendation:**")
    if score >= 70:
        lines.append("This product is a reasonable choice. Maintain variety in your diet.")
    elif score >= 40:
        lines.append(
            "Limit intake to a few servings per week. Pair with whole foods such as fruits, "
            "vegetables, and whole grains."
        )
    else:
        lines.append(
            "Consider healthier alternatives. If you must consume this product, keep portions "
            "small and infrequent. Consult a dietitian for personalised guidance."
        )

    lines.append(
        "\n> *This analysis is AI-generated for educational purposes. "
        "Always consult a qualified nutritionist for personal dietary advice.*"
    )

    return "\n\n".join(lines)
