"""
utils.py – Core logic for e-K Cha Label?

Covers: OCR (Tesseract), text cleaning, dataset-driven ingredient
detection, nutritional parsing, health scoring, grading and
AI-style explanation generation.

The harmful / safe ingredient knowledge now lives in an external
dataset (``ingredients_dataset.csv``) instead of being hard-coded,
so it can be edited without touching the code.
"""

from __future__ import annotations

import os
import re
import functools
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "ingredients_dataset.csv")


@functools.lru_cache(maxsize=1)
def load_ingredient_db(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the ingredient knowledge base from CSV.

    Expected columns:
        ingredient, aliases, category, classification, risk_level, description

    ``classification`` is one of: harmful | caution | safe
    ``risk_level``     is one of: High | Moderate | Low | Safe
    """
    df = pd.read_csv(path)
    # Normalise text columns
    for col in ["ingredient", "aliases", "category", "classification", "risk_level", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df["ingredient"] = df["ingredient"].str.lower()
    df["classification"] = df["classification"].str.lower()
    return df


def _search_terms(row: pd.Series) -> list[str]:
    """All matchable terms for a dataset row: the name plus any aliases."""
    terms = [row["ingredient"]]
    if row["aliases"]:
        terms += [a.strip().lower() for a in row["aliases"].split("|") if a.strip()]
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR  (Tesseract via pytesseract)
# ─────────────────────────────────────────────────────────────────────────────

# UI language codes -> Tesseract 3-letter codes
_TESS_LANG_MAP = {"en": "eng", "ne": "nep", "hi": "hin"}


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Clean up a label photo so Tesseract reads it accurately.

    Steps: grayscale -> upscale small images -> auto-contrast ->
    sharpen -> binarise (Otsu threshold).  Uses OpenCV when available
    for adaptive thresholding, otherwise a pure-PIL/NumPy fallback.
    """
    gray = ImageOps.grayscale(image)

    # Upscale small images – Tesseract likes characters ~30px tall.
    w, h = gray.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)

    arr = np.array(gray)

    try:
        import cv2  # optional, gives the best binarisation

        denoised = cv2.fastNlMeansDenoising(arr, h=10)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11,
        )
        return Image.fromarray(binary)
    except Exception:
        # NumPy Otsu fallback
        hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
        total = arr.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b = w_b = max_var = threshold = 0.0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var, threshold = var_between, t
        binary = (arr > threshold).astype(np.uint8) * 255
        return Image.fromarray(binary)


def extract_text(
    image: Image.Image,
    languages: Optional[list[str]] = None,
    preprocess: bool = True,
) -> str:
    """
    Run Tesseract OCR on a PIL image and return the extracted text.

    ``languages`` are UI codes (en/ne/hi); they are mapped to Tesseract
    language packs. Requires the system ``tesseract-ocr`` binary plus the
    relevant language data (see packages.txt for Streamlit Cloud).
    """
    import pytesseract

    if languages is None:
        languages = ["en"]

    tess_langs = "+".join(_TESS_LANG_MAP.get(l, "eng") for l in languages) or "eng"

    work_img = _preprocess_for_ocr(image) if preprocess else image

    # --oem 3 = default LSTM engine, --psm 6 = assume a uniform block of text.
    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(work_img, lang=tess_langs, config=config)
    except pytesseract.TesseractError:
        # Requested language pack might be missing – retry in English only.
        text = pytesseract.image_to_string(work_img, lang="eng", config=config)

    # Tidy whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "and", "for", "with", "per", "from", "this", "that", "are", "was",
    "may", "contains", "contain", "ingredients", "ingredient", "product",
    "value", "values", "serving", "size", "total", "net", "weight",
}


def clean_text(text: str) -> str:
    """Lowercase, strip non-letters and drop short/stop words."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        tokens = [
            tok.lemma_
            for tok in doc
            if not tok.is_stop and tok.is_alpha and len(tok.text) > 2
        ]
        return " ".join(tokens)
    except Exception:
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
        return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ingredient detection (dataset-driven)
# ─────────────────────────────────────────────────────────────────────────────

def detect_ingredients(text: str) -> list[dict]:
    """
    Match dataset ingredients against *text* (raw OCR text works best).

    Returns a list of dicts, one per matched ingredient:
        {ingredient, category, classification, risk_level, description}
    Longer terms are matched first so multi-word names win over substrings.
    """
    db = load_ingredient_db()
    text_lower = " " + text.lower() + " "
    matched: dict[str, dict] = {}

    # Build (term, row) pairs sorted by term length, longest first.
    candidates: list[tuple[str, pd.Series]] = []
    for _, row in db.iterrows():
        for term in _search_terms(row):
            if term:
                candidates.append((term, row))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    for term, row in candidates:
        name = row["ingredient"]
        if name in matched:
            continue
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text_lower):
            matched[name] = {
                "ingredient": name,
                "category": row["category"],
                "classification": row["classification"],
                "risk_level": row["risk_level"],
                "description": row["description"],
            }

    # Sort: harmful first, then caution, then safe; alphabetical within group.
    order = {"harmful": 0, "caution": 1, "safe": 2}
    return sorted(
        matched.values(),
        key=lambda d: (order.get(d["classification"], 3), d["ingredient"]),
    )


def detect_harmful_ingredients(text: str) -> list[str]:
    """
    Backwards-compatible helper: names of detected harmful/caution
    ingredients (i.e. anything that is not classified 'safe').
    """
    return [
        d["ingredient"]
        for d in detect_ingredients(text)
        if d["classification"] in ("harmful", "caution")
    ]


def count_concerns(detected: list[dict]) -> tuple[int, int, int]:
    """Return (harmful_count, caution_count, safe_count) from detected list."""
    harmful = sum(1 for d in detected if d["classification"] == "harmful")
    caution = sum(1 for d in detected if d["classification"] == "caution")
    safe = sum(1 for d in detected if d["classification"] == "safe")
    return harmful, caution, safe


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nutritional value parsing
# ─────────────────────────────────────────────────────────────────────────────

# Each pattern captures (value, unit). The keyword-to-number gap is bounded
# so that additive names in the ingredient list (e.g. "sodium benzoate",
# "salt") cannot accidentally grab a distant number from elsewhere on the
# label. A required mass/energy unit further disambiguates the value.
_NUTRIENT_PATTERNS: dict[str, list[str]] = {
    "calories":  [r"(?:energy|calories?)[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(kcal|cal|kj)?"],
    "sugar":     [r"sugars?\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)",
                  r"sugars?\b[^\d\n]{0,6}(\d+(?:\.\d+)?)()"],
    "fat":       [r"(?:total\s*fat|fat)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)",
                  r"(?:total\s*fat|fat)\b[^\d\n]{0,6}(\d+(?:\.\d+)?)()"],
    "sodium":    [r"(?:sodium|salt)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)"],
    "protein":   [r"protein\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "carbs":     [r"(?:carbohydrates?|carbs?)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "fibre":     [r"(?:dietary\s*fibre|fibre|fiber)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "saturated": [r"(?:saturated\s*fat|saturates)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
}


def parse_nutritional_values(raw_text: str) -> dict[str, float]:
    """
    Extract numeric nutritional values (per 100 g/ml) from OCR text.

    Values reported in mg are converted to grams. Calories are left as kcal.
    """
    text_lower = raw_text.lower()
    result: dict[str, float] = {k: 0.0 for k in _NUTRIENT_PATTERNS}

    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            if unit == "mg" and nutrient != "calories":
                value /= 1000.0
            result[nutrient] = value
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Health scoring
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "sugar":    {"low": 5,   "high": 22.5},  # g / 100g
    "fat":      {"low": 3,   "high": 17.5},  # g / 100g
    "sodium":   {"low": 0.3, "high": 1.5},   # g / 100g
    "calories": {"low": 40,  "high": 400},   # kcal / 100g
}

WEIGHTS = {"sugar": 0.30, "fat": 0.25, "additives": 0.25, "calories": 0.20}


def _penalty(value: float, low: float, high: float) -> float:
    """0 = healthy, 1 = very unhealthy (linear between thresholds)."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def compute_health_score(nutrition: dict[str, float], additive_count: int) -> int:
    """
    Compute a 0–100 health score (higher = healthier) from weighted
    penalties for sugar, fat, calories and additive count.
    """
    sugar_pen = _penalty(nutrition.get("sugar", 0),
                         _THRESHOLDS["sugar"]["low"], _THRESHOLDS["sugar"]["high"])
    fat_pen = _penalty(nutrition.get("fat", 0),
                       _THRESHOLDS["fat"]["low"], _THRESHOLDS["fat"]["high"])
    calorie_pen = _penalty(nutrition.get("calories", 0),
                           _THRESHOLDS["calories"]["low"], _THRESHOLDS["calories"]["high"])
    additive_pen = min(additive_count / 5.0, 1.0)  # 5+ additives → max penalty

    total_penalty = (
        WEIGHTS["sugar"] * sugar_pen
        + WEIGHTS["fat"] * fat_pen
        + WEIGHTS["additives"] * additive_pen
        + WEIGHTS["calories"] * calorie_pen
    )

    score = int(round((1 - total_penalty) * 100))
    return max(0, min(100, score))


def get_grade(score: int) -> str:
    """Map score to a Nutri-Score-style A–E grade."""
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "E"


def classify_health(score: int) -> str:
    """Map score to a risk label."""
    if score >= 70:
        return "✅ Healthy"
    if score >= 40:
        return "⚠️ Moderate"
    return "🚨 Unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# 6. AI-style explanation (rule-based, dataset-aware)
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    score: int,
    risk: str,
    detected: list[dict],
    nutrition: dict[str, float],
) -> str:
    """Generate a markdown plain-language health summary."""
    harmful_names = [d["ingredient"] for d in detected
                     if d["classification"] in ("harmful", "caution")]
    high_risk = [d["ingredient"] for d in detected if d["risk_level"] == "High"]

    lines: list[str] = []

    if score >= 70:
        lines.append(f"### ✅ Overall: relatively healthy (score {score}/100)")
        lines.append("The nutritional profile looks acceptable. Enjoy in moderation "
                     "as part of a balanced diet.")
    elif score >= 40:
        lines.append(f"### ⚠️ Overall: moderate health concerns (score {score}/100)")
        lines.append("Suitable for occasional consumption, but it should not form a "
                     "major part of your daily diet.")
    else:
        lines.append(f"### 🚨 Overall: not recommended for regular use (score {score}/100)")
        lines.append("High levels of sugar, fat or concerning additives make this a "
                     "poor everyday choice.")

    lines.append("---")

    sugar = nutrition.get("sugar", 0)
    if sugar > 22.5:
        lines.append(f"🔴 **High sugar** ({sugar} g/100g) — well above the 22.5 g guideline. "
                     "Frequent intake raises the risk of diabetes and tooth decay.")
    elif sugar > 5:
        lines.append(f"🟡 **Moderate sugar** ({sugar} g/100g) — keep total daily sugar below ~30 g.")
    elif sugar > 0:
        lines.append(f"🟢 **Low sugar** ({sugar} g/100g) — within healthy limits.")

    fat = nutrition.get("fat", 0)
    if fat > 17.5:
        lines.append(f"🔴 **High fat** ({fat} g/100g) — exceeds the recommended threshold.")
    elif fat > 3:
        lines.append(f"🟡 **Moderate fat** ({fat} g/100g).")
    elif fat > 0:
        lines.append(f"🟢 **Low fat** ({fat} g/100g).")

    sodium = nutrition.get("sodium", 0)
    if sodium > 1.5:
        lines.append(f"🔴 **High sodium** ({sodium} g/100g) — may contribute to high blood pressure.")
    elif sodium > 0.3:
        lines.append(f"🟡 **Moderate sodium** ({sodium} g/100g).")

    if harmful_names:
        lines.append(f"⚠️ **{len(harmful_names)} additive(s) of concern detected:** "
                     + ", ".join(n.title() for n in harmful_names) + ".")
        if high_risk:
            lines.append("🔴 One or more **high-risk additives** found ("
                         + ", ".join(n.title() for n in high_risk)
                         + "). These have been linked to serious health conditions.")
    else:
        lines.append("🟢 **No additives of concern** detected in the extracted text.")

    lines.append("---")
    lines.append("**💡 Recommendation:**")
    if score >= 70:
        lines.append("A reasonable choice — keep variety in your diet.")
    elif score >= 40:
        lines.append("Limit to a few servings per week and pair with whole foods such as "
                     "fruit, vegetables and whole grains.")
    else:
        lines.append("Consider healthier alternatives. If you do consume it, keep portions "
                     "small and infrequent, and consult a dietitian for guidance.")

    lines.append("> *This analysis is generated automatically for educational purposes. "
                 "Always consult a qualified nutritionist for personal dietary advice.*")

    return "\n\n".join(lines)
