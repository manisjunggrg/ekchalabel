"""
utils.py – e-K Cha Label?

Single-engine approach: Google Gemini Flash Vision reads the label image
and returns structured JSON (product name, ingredients, nutrition table).
No traditional OCR, no regex parsing, no zone splitting needed.

Ingredient intelligence comes from `eatsafe_master_database.csv` (8.7k items).
"""

from __future__ import annotations

import os
import re
import json
import base64
import functools
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eatsafe_master_database.csv")

_RISK_TO_CLASS = {"avoid": "harmful", "limit": "caution", "safe": "safe"}
_RISK_TO_LEVEL = {"avoid": "High", "limit": "Moderate", "safe": "Safe"}
_NOVA_LABEL = {
    1: "NOVA 1 · Unprocessed",
    2: "NOVA 2 · Processed culinary ingredient",
    3: "NOVA 3 · Processed food",
    4: "NOVA 4 · Ultra-processed",
}


def nova_label(group) -> str:
    try:
        return _NOVA_LABEL.get(int(group), f"NOVA {group}")
    except (ValueError, TypeError):
        return "NOVA n/a"


@functools.lru_cache(maxsize=1)
def load_ingredient_db(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ingredient"] = df["ingredient"].fillna("").astype(str).str.strip().str.lower()
    df["additive_risk"] = df["additive_risk"].fillna("limit").astype(str).str.lower()
    df["reason"] = df["reason"].fillna("").astype(str).str.strip()
    for col in ["gemini_rating", "processing_penalty", "nutrition_impact", "whole_food_bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce").fillna(0).astype(int)
    df["classification"] = df["additive_risk"].map(_RISK_TO_CLASS).fillna("caution")
    df["risk_level"] = df["additive_risk"].map(_RISK_TO_LEVEL).fillna("Moderate")
    return df[df["ingredient"] != ""].drop_duplicates(subset="ingredient").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gemini Vision – structured label extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are a food label reader. Analyse this product label image and
extract ALL information you can see. Return ONLY valid JSON (no markdown
fences, no explanation) with exactly this structure:

{
  "product_name": "string or empty",
  "ingredients": ["ingredient1", "ingredient2", ...],
  "nutrition_facts": [
    {"nutrient": "Calories", "value": 320, "unit": "kcal"},
    {"nutrient": "Total Fat", "value": 14.0, "unit": "g"},
    ...
  ],
  "raw_text": "all text visible on the label as a single string"
}

Rules:
- For ingredients: list every individual ingredient you can read, split
  by commas. Keep names simple (e.g. "wheat flour" not "WHEAT FLOUR (66%)").
- For nutrition_facts: extract EVERY row from the nutrition table.
  Include the nutrient name exactly as printed, its numeric value, and
  the unit (g, mg, kcal, %, IU, mcg etc). If a value is 0, still include it.
- For raw_text: transcribe everything you can read on the label.
- If a section is not visible, use an empty list.
- Return ONLY the JSON object. No other text."""


def _get_gemini_key() -> Optional[str]:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def gemini_key_configured() -> bool:
    return bool(_get_gemini_key())


def _image_to_base64(image: Image.Image, max_side: int = 2000) -> str:
    """Resize if needed, encode to base64 JPEG."""
    img = image.copy()
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_label_data(image: Image.Image) -> dict:
    """
    Send the label image to Gemini Flash and get structured extraction.

    Returns dict with keys:
        product_name, ingredients, nutrition_facts, raw_text
    """
    import time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    key = _get_gemini_key()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Add it in Settings → Secrets on Streamlit Cloud, "
            "or set the GOOGLE_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Retry up to 3 times with backoff on rate limits.
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                [_EXTRACTION_PROMPT, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            break  # success
        except ResourceExhausted as e:
            last_err = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))  # 5s, 10s
                continue
            raise RuntimeError(
                "⏳ **Gemini rate limit reached** (free tier: 15 requests/min).  \n"
                "Wait ~60 seconds and try again, or upgrade to a paid key for higher limits."
            ) from e

    raw = response.text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            data = json.loads(match.group())
        else:
            raise RuntimeError(f"Gemini returned invalid JSON:\n{raw[:500]}")

    # Normalise
    data.setdefault("product_name", "")
    data.setdefault("ingredients", [])
    data.setdefault("nutrition_facts", [])
    data.setdefault("raw_text", "")

    # Ensure ingredients is a flat list of strings
    data["ingredients"] = [
        str(i).strip() for i in data["ingredients"]
        if str(i).strip()
    ]

    # Ensure nutrition_facts entries have the right shape
    cleaned_nf = []
    for item in data["nutrition_facts"]:
        if isinstance(item, dict) and "nutrient" in item:
            try:
                val = float(item.get("value", 0))
            except (ValueError, TypeError):
                val = 0.0
            cleaned_nf.append({
                "nutrient": str(item["nutrient"]).strip(),
                "value": val,
                "unit": str(item.get("unit", "")).strip(),
            })
    data["nutrition_facts"] = cleaned_nf
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ingredient matching (dataset-driven)
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _build_lookup() -> dict[str, dict]:
    """Build a lowercase name → metadata lookup from the dataset."""
    db = load_ingredient_db()
    lookup: dict[str, dict] = {}
    for row in db.itertuples(index=False):
        lookup[row.ingredient] = {
            "ingredient": row.ingredient,
            "classification": row.classification,
            "risk_level": row.risk_level,
            "category": nova_label(row.nova_group),
            "nova_group": int(row.nova_group),
            "gemini_rating": float(row.gemini_rating),
            "artificial_flag": bool(row.artificial_flag),
            "processing_penalty": float(row.processing_penalty),
            "nutrition_impact": float(row.nutrition_impact),
            "whole_food_bonus": float(row.whole_food_bonus),
            "description": row.reason,
        }
    return lookup


def match_ingredients(ingredient_list: list[str]) -> list[dict]:
    """
    Match a list of ingredient names (from Gemini) against the dataset.

    Uses exact match first, then substring search for multi-word names.
    Returns metadata dicts for every match, sorted harmful → caution → safe.
    """
    lookup = _build_lookup()
    db = load_ingredient_db()
    all_names = set(db["ingredient"].tolist())
    matched: dict[str, dict] = {}

    for raw_name in ingredient_list:
        name = raw_name.strip().lower()
        if not name or len(name) < 3:
            continue

        # 1. Exact match
        if name in lookup and name not in matched:
            matched[name] = lookup[name]
            continue

        # 2. Substring match: find the longest dataset name inside this ingredient
        best: Optional[str] = None
        best_len = 0
        for db_name in all_names:
            if len(db_name) < 4:
                continue  # skip very short to avoid noise
            if db_name in name and len(db_name) > best_len:
                best = db_name
                best_len = len(db_name)
        if best and best not in matched:
            matched[best] = lookup[best]

    order = {"harmful": 0, "caution": 1, "safe": 2}
    return sorted(
        matched.values(),
        key=lambda d: (order.get(d["classification"], 3), d["gemini_rating"]),
    )


def count_concerns(detected: list[dict]) -> tuple[int, int, int]:
    h = sum(1 for d in detected if d["classification"] == "harmful")
    c = sum(1 for d in detected if d["classification"] == "caution")
    s = sum(1 for d in detected if d["classification"] == "safe")
    return h, c, s


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nutrition helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map common nutrient label names to canonical keys for scoring.
_NUTRIENT_ALIASES: dict[str, str] = {}
for _key, _aliases in {
    "calories": ["calories", "energy", "kcal", "cal"],
    "sugar": ["sugar", "sugars", "total sugars", "of which sugars", "total sugar"],
    "fat": ["total fat", "fat", "total fats"],
    "sodium": ["sodium", "salt", "na"],
    "protein": ["protein", "proteins"],
    "carbs": ["carbohydrate", "carbohydrates", "total carbohydrate", "total carbohydrates", "carbs"],
    "fibre": ["dietary fibre", "dietary fiber", "fibre", "fiber", "total dietary fiber"],
    "saturated": ["saturated fat", "saturated fats", "saturates", "of which saturates"],
    "trans_fat": ["trans fat", "trans fats", "trans"],
    "cholesterol": ["cholesterol"],
}.items():
    for a in _aliases:
        _NUTRIENT_ALIASES[a.lower()] = _key


def nutrition_to_scoring_dict(nutrition_facts: list[dict]) -> dict[str, float]:
    """
    Convert the Gemini nutrition_facts list into a flat dict with
    canonical keys and values normalised to grams (mg → g).
    """
    result: dict[str, float] = {}
    for item in nutrition_facts:
        name = item["nutrient"].lower().strip()
        canonical = _NUTRIENT_ALIASES.get(name)
        if not canonical:
            # Try partial match
            for alias, key in _NUTRIENT_ALIASES.items():
                if alias in name:
                    canonical = key
                    break
        if not canonical:
            continue
        if canonical in result:
            continue  # keep first match (usually the main value)

        value = item["value"]
        unit = item["unit"].lower().strip()
        if unit == "mg":
            value /= 1000.0
        elif unit in ("mcg", "µg"):
            value /= 1_000_000.0
        result[canonical] = value

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scoring
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "sugar":    {"low": 5,   "high": 22.5},
    "fat":      {"low": 3,   "high": 17.5},
    "sodium":   {"low": 0.3, "high": 1.5},
    "calories": {"low": 40,  "high": 400},
}


def _penalty(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _nutrition_score(nutrition: dict[str, float]) -> Optional[float]:
    if not any(nutrition.get(k, 0) > 0 for k in ("sugar", "fat", "sodium", "calories")):
        return None
    pen = (
        0.35 * _penalty(nutrition.get("sugar", 0), *_THRESHOLDS["sugar"].values())
        + 0.30 * _penalty(nutrition.get("fat", 0), *_THRESHOLDS["fat"].values())
        + 0.20 * _penalty(nutrition.get("calories", 0), *_THRESHOLDS["calories"].values())
        + 0.15 * _penalty(nutrition.get("sodium", 0), *_THRESHOLDS["sodium"].values())
    )
    return (1 - pen) * 100


def _ingredient_score(detected: list[dict]) -> Optional[float]:
    if not detected:
        return None
    ratings = [d["gemini_rating"] for d in detected]
    proc = np.mean([d["processing_penalty"] for d in detected])
    bonus = np.mean([d["whole_food_bonus"] for d in detected])
    base = (np.mean(ratings) / 5.0) * 100
    return float(np.clip(base - proc * 12 + bonus * 8, 0, 100))


def compute_score(detected: list[dict], nutrition: dict[str, float]) -> int:
    ing = _ingredient_score(detected)
    nut = _nutrition_score(nutrition)
    if ing is not None and nut is not None:
        base = 0.60 * ing + 0.40 * nut
    elif ing is not None:
        base = ing
    elif nut is not None:
        base = nut
    else:
        base = 50.0
    avoid_count = sum(1 for d in detected if d["classification"] == "harmful")
    base -= min(avoid_count * 4, 20)
    return int(round(max(0, min(100, base))))


def get_grade(score: int) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "E"


def classify_health(score: int) -> str:
    if score >= 70: return "✅ Healthy"
    if score >= 40: return "⚠️ Moderate"
    return "🚨 Unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Explanation
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    score: int, risk: str, detected: list[dict], nutrition: dict[str, float],
) -> str:
    avoid = [d for d in detected if d["classification"] == "harmful"]
    limit = [d for d in detected if d["classification"] == "caution"]
    worst = sorted(avoid + limit, key=lambda d: d["gemini_rating"])[:5]

    lines: list[str] = []
    if score >= 70:
        lines.append(f"### ✅ Relatively healthy (score {score}/100)")
        lines.append("Ingredients are mostly whole / minimally processed.")
    elif score >= 40:
        lines.append(f"### ⚠️ Moderate concerns (score {score}/100)")
        lines.append("Suitable for occasional consumption.")
    else:
        lines.append(f"### 🚨 Not recommended for regular use (score {score}/100)")
        lines.append("Heavily processed ingredients and/or poor nutrition.")
    lines.append("---")

    if detected:
        avg = np.mean([d["gemini_rating"] for d in detected])
        lines.append(f"**Ingredient quality:** {avg:.1f}/5 average across "
                     f"{len(detected)} matched ingredients.")

    sugar = nutrition.get("sugar", 0)
    if sugar > 22.5:
        lines.append(f"🔴 **High sugar** ({sugar:.1f} g)")
    elif sugar > 5:
        lines.append(f"🟡 **Moderate sugar** ({sugar:.1f} g)")

    fat = nutrition.get("fat", 0)
    if fat > 17.5:
        lines.append(f"🔴 **High fat** ({fat:.1f} g)")
    elif fat > 3:
        lines.append(f"🟡 **Moderate fat** ({fat:.1f} g)")

    sodium = nutrition.get("sodium", 0)
    if sodium > 1.5:
        lines.append(f"🔴 **High sodium** ({sodium:.2f} g)")
    elif sodium > 0.3:
        lines.append(f"🟡 **Moderate sodium** ({sodium:.2f} g)")

    if avoid:
        lines.append(f"⛔ **{len(avoid)} 'avoid' additive(s):** "
                     + ", ".join(d["ingredient"].title() for d in avoid[:6]))
    if worst:
        lines.append("**Most concerning:**")
        for d in worst:
            lines.append(f"- **{d['ingredient'].title()}** ({d['gemini_rating']:.1f}/5) — {d['description']}")
    if not avoid and not limit:
        lines.append("🟢 **No flagged additives** detected.")

    lines.append("---")
    lines.append("**💡 Recommendation:**")
    if score >= 70:
        lines.append("A reasonable choice — maintain variety.")
    elif score >= 40:
        lines.append("Limit to a few servings per week; pair with whole foods.")
    else:
        lines.append("Prefer less-processed alternatives.")
    lines.append("> *Educational analysis. Consult a nutritionist for personal advice.*")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LabelAnalysis:
    __slots__ = (
        "product_name", "raw_text",
        "extracted_ingredients", "nutrition_facts",
        "detected", "harmful", "nutrition",
        "score", "grade", "risk",
        "h_cnt", "c_cnt", "s_cnt",
    )

    def __init__(self):
        self.product_name: str = ""
        self.raw_text: str = ""
        self.extracted_ingredients: list[str] = []
        self.nutrition_facts: list[dict] = []
        self.detected: list[dict] = []
        self.harmful: list[str] = []
        self.nutrition: dict[str, float] = {}
        self.score: int = 50
        self.grade: str = "C"
        self.risk: str = "⚠️ Moderate"
        self.h_cnt = self.c_cnt = self.s_cnt = 0


def analyze_label(image: Image.Image) -> LabelAnalysis:
    """Full pipeline: Gemini extraction → dataset match → score."""
    data = extract_label_data(image)

    a = LabelAnalysis()
    a.product_name = data.get("product_name", "")
    a.raw_text = data.get("raw_text", "")
    a.extracted_ingredients = data.get("ingredients", [])
    a.nutrition_facts = data.get("nutrition_facts", [])

    a.nutrition = nutrition_to_scoring_dict(a.nutrition_facts)
    a.detected = match_ingredients(a.extracted_ingredients)
    a.harmful = [d["ingredient"] for d in a.detected
                 if d["classification"] in ("harmful", "caution")]
    a.h_cnt, a.c_cnt, a.s_cnt = count_concerns(a.detected)

    a.score = compute_score(a.detected, a.nutrition)
    a.grade = get_grade(a.score)
    a.risk = classify_health(a.score)
    return a
