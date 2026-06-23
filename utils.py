"""
utils.py – e-K Cha Label?

OCR engine: Surya OCR (v0.16.x) — deep-learning, 90+ languages, line-level
bounding boxes. Far more accurate than Tesseract on photographed labels.

Pipeline: Surya OCR → zone split → nutrition parsing → dataset match → score.
"""

from __future__ import annotations

import os
import re
import functools
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

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
# 2. Surya OCR
# ─────────────────────────────────────────────────────────────────────────────

# Cached model objects — loaded once via st.cache_resource in app.py.
_MODELS: dict = {}


def load_surya_models():
    """Load Surya detection + recognition models. Call once and cache."""
    if _MODELS:
        return _MODELS
    # Minimise memory: force CPU, small batch.
    os.environ.setdefault("TORCH_DEVICE", "cpu")
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "4")
    os.environ.setdefault("DETECTOR_BATCH_SIZE", "2")

    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.model import load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    _MODELS["det_model"] = load_det_model()
    _MODELS["det_processor"] = load_det_processor()
    _MODELS["rec_model"] = load_rec_model()
    _MODELS["rec_processor"] = load_rec_processor()
    return _MODELS


class OCRLine:
    """A single text line with bounding box."""
    __slots__ = ("text", "bbox", "confidence")

    def __init__(self, text: str, bbox: tuple | None = None, confidence: float = 0.0):
        self.text = text
        self.bbox = bbox        # (x1, y1, x2, y2) corners
        self.confidence = confidence


class OCRResult:
    __slots__ = ("lines", "annotated_image")

    def __init__(self):
        self.lines: list[OCRLine] = []
        self.annotated_image: Optional[Image.Image] = None

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines if ln.text.strip())


def _preprocess(image: Image.Image) -> Image.Image:
    """Light enhancement — Surya handles most preprocessing internally."""
    from PIL import ImageEnhance
    img = image.copy()
    w, h = img.size
    if max(w, h) < 1500:
        scale = 1500 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


def run_surya_ocr(image: Image.Image, langs: list[str] | None = None,
                  preprocess: bool = True) -> OCRResult:
    """Run Surya OCR and return lines with bounding boxes."""
    from surya.ocr import run_ocr

    models = load_surya_models()
    work = _preprocess(image) if preprocess else image
    if langs is None:
        langs = ["en"]

    predictions = run_ocr(
        [work], [langs],
        models["det_model"], models["det_processor"],
        models["rec_model"], models["rec_processor"],
    )

    result = OCRResult()
    if predictions:
        for text_line in predictions[0].text_lines:
            bbox_raw = text_line.bbox          # [x1,y1,x2,y2]
            bbox = tuple(bbox_raw) if bbox_raw else None
            conf = getattr(text_line, "confidence", 0.0)
            result.lines.append(OCRLine(text_line.text, bbox, conf))
    # Sort top to bottom.
    result.lines.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0))
    return result


# ── Nutrition-line detection (for overlay colours) ───────────────────────────

_NUTRITION_KEYWORDS = {
    "energy", "calories", "kcal", "cal", "fat", "saturated", "trans",
    "carbohydrate", "sugar", "sugars", "fibre", "fiber", "protein",
    "sodium", "salt", "cholesterol", "serving", "nutrition", "daily",
    "value", "vitamin", "iron", "calcium", "potassium", "amount",
}


def _is_nutrition_line(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _NUTRITION_KEYWORDS) and bool(re.search(r"\d", text))


# ── Draw overlay ─────────────────────────────────────────────────────────────

def draw_ocr_overlay(image: Image.Image, ocr_result: OCRResult) -> Image.Image:
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  max(10, int(image.height * 0.013)))
    except Exception:
        font = ImageFont.load_default()

    # Scale factor: if we preprocessed up, bboxes are in the scaled space.
    w_orig, h_orig = image.size
    if ocr_result.lines and ocr_result.lines[0].bbox:
        max_x = max((ln.bbox[2] for ln in ocr_result.lines if ln.bbox), default=w_orig)
        scale_x = w_orig / max_x if max_x > w_orig * 1.1 else 1.0
        max_y = max((ln.bbox[3] for ln in ocr_result.lines if ln.bbox), default=h_orig)
        scale_y = h_orig / max_y if max_y > h_orig * 1.1 else 1.0
    else:
        scale_x = scale_y = 1.0

    for ln in ocr_result.lines:
        if not ln.bbox or not ln.text.strip():
            continue
        x1, y1, x2, y2 = [c * s for c, s in zip(ln.bbox, [scale_x, scale_y, scale_x, scale_y])]
        is_nut = _is_nutrition_line(ln.text)
        fill = (34, 197, 94, 40) if is_nut else (59, 130, 246, 40)
        outline = (34, 197, 94) if is_nut else (59, 130, 246)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# 3. Zone splitting
# ─────────────────────────────────────────────────────────────────────────────

_INGREDIENT_HEADERS = re.compile(
    r"(?:ingredients|composition|made\s+(?:from|with))\s*[:\-]?", re.IGNORECASE)
_NUTRITION_HEADERS = re.compile(
    r"(?:nutrition\s*(?:facts|information|value|per)|amount\s+per|daily\s+value|"
    r"per\s+(?:serving|\d+\s*[gm]l?))", re.IGNORECASE)
_NUTRITION_ROW = re.compile(
    r"(?:energy|calories?|kcal|total\s*fat|trans\s*fat|saturated|cholesterol|"
    r"sodium|salt|carbohydrate|sugar|fibre|fiber|protein|vitamin|calcium|"
    r"iron|potassium|daily\s*value|serving|amount)[^\n]{0,30}\d", re.IGNORECASE)


class TextZones:
    __slots__ = ("nutrition", "ingredients", "other", "full")

    def __init__(self, nutrition: str, ingredients: str, other: str, full: str):
        self.nutrition = nutrition
        self.ingredients = ingredients
        self.other = other
        self.full = full

    @property
    def for_ingredient_matching(self) -> str:
        parts = []
        if self.ingredients.strip():
            parts.append(self.ingredients)
        if self.other.strip():
            parts.append(self.other)
        return "\n".join(parts) if parts else self.full

    @property
    def for_nutrition_parsing(self) -> str:
        return self.nutrition if self.nutrition.strip() else self.full


def split_text_zones(text: str) -> TextZones:
    lines = text.split("\n")
    nut, ing, other = [], [], []
    zone = "other"
    for line in lines:
        s = line.strip()
        if not s:
            if zone == "ingredients":
                zone = "other"
            continue
        if _INGREDIENT_HEADERS.search(s):
            zone = "ingredients"
            after = re.split(r"[:;\-]\s*", s, maxsplit=1)
            if len(after) > 1 and after[1].strip():
                ing.append(after[1].strip())
            continue
        if _NUTRITION_HEADERS.search(s):
            zone = "nutrition"
            continue
        if zone == "ingredients":
            ing.append(s)
            continue
        if _NUTRITION_ROW.search(s):
            nut.append(s)
            zone = "nutrition"
            continue
        if zone == "nutrition":
            zone = "other"
        other.append(s)
    return TextZones("\n".join(nut), "\n".join(ing), "\n".join(other), text)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nutrition parsing
# ─────────────────────────────────────────────────────────────────────────────

_NUTRIENT_PATTERNS = {
    "calories":  [r"(?:energy|calories?)[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(kcal|cal|kj)?"],
    "sugar":     [r"sugars?\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "fat":       [r"(?:total\s*fat|fat)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "sodium":    [r"(?:sodium|salt)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)"],
    "protein":   [r"protein\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "carbs":     [r"(?:carbohydrates?|carbs?)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "fibre":     [r"(?:dietary\s*fibre|fibre|fiber)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "saturated": [r"(?:saturated\s*fat|saturates)\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "trans_fat": [r"trans\s*fat\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
    "cholesterol": [r"cholesterol\b[^\d\n]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)?"],
}


def parse_nutritional_values(raw_text: str) -> dict[str, float]:
    text_lower = raw_text.lower()
    result = {k: 0.0 for k in _NUTRIENT_PATTERNS}
    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, text_lower)
            if not m:
                continue
            try:
                value = float(m.group(1))
            except (ValueError, IndexError):
                continue
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            if unit == "mg" and nutrient != "calories":
                value /= 1000.0
            result[nutrient] = value
            break
    return result


def nutrition_for_display(nutrition: dict[str, float]) -> list[dict]:
    """Convert flat scoring dict to displayable list with units."""
    display = []
    for key, value in nutrition.items():
        if value > 0:
            unit = "kcal" if key == "calories" else "g"
            label = key.replace("_", " ").title()
            display.append({"nutrient": label, "value": value, "unit": unit})
    return display


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ingredient matching
# ─────────────────────────────────────────────────────────────────────────────

_SHORT_ALLOW = {"msg", "bha", "bht", "egg", "soy", "oat", "nut", "b12", "b9", "d3"}

_NUTRIENT_NOISE = {
    "sugar", "sugars", "sodium", "cholesterol", "fat", "trans fat",
    "saturated fat", "total fat", "calories", "protein", "carbohydrate",
    "carbohydrates", "fibre", "fiber", "energy",
    "powder", "soup", "flavors", "flavor", "colour", "color",
    "extract", "concentrate", "blend",
}


def _matchable(term: str) -> bool:
    t = term.strip().lower()
    return len(t) >= 4 or t in _SHORT_ALLOW


@functools.lru_cache(maxsize=1)
def _build_index():
    db = load_ingredient_db()
    matchers = []
    meta = {}
    for row in db.itertuples(index=False):
        name = row.ingredient
        meta[name] = {
            "ingredient": name,
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
        if _matchable(name) and name not in _NUTRIENT_NOISE:
            matchers.append((name, re.compile(r"\b" + re.escape(name) + r"\b")))
    matchers.sort(key=lambda x: len(x[0]), reverse=True)
    return matchers, meta


def detect_ingredients(text: str) -> list[dict]:
    matchers, meta = _build_index()
    haystack = " " + text.lower() + " "
    found = {}
    claimed = []
    for name, pattern in matchers:
        if name in found:
            continue
        for mt in pattern.finditer(haystack):
            s, e = mt.span()
            if any(cs <= s and e <= ce for cs, ce in claimed):
                continue
            found[name] = meta[name]
            claimed.append((s, e))
            break
    order = {"harmful": 0, "caution": 1, "safe": 2}
    return sorted(found.values(),
                  key=lambda d: (order.get(d["classification"], 3), d["gemini_rating"]))


def count_concerns(detected):
    h = sum(1 for d in detected if d["classification"] == "harmful")
    c = sum(1 for d in detected if d["classification"] == "caution")
    s = sum(1 for d in detected if d["classification"] == "safe")
    return h, c, s


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scoring
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "sugar": {"low": 5, "high": 22.5},
    "fat": {"low": 3, "high": 17.5},
    "sodium": {"low": 0.3, "high": 1.5},
    "calories": {"low": 40, "high": 400},
}


def _penalty(value, low, high):
    if value <= low: return 0.0
    if value >= high: return 1.0
    return (value - low) / (high - low)


def _nutrition_score(nutrition):
    if not any(nutrition.get(k, 0) > 0 for k in ("sugar", "fat", "sodium", "calories")):
        return None
    pen = (0.35 * _penalty(nutrition.get("sugar", 0), 5, 22.5)
           + 0.30 * _penalty(nutrition.get("fat", 0), 3, 17.5)
           + 0.20 * _penalty(nutrition.get("calories", 0), 40, 400)
           + 0.15 * _penalty(nutrition.get("sodium", 0), 0.3, 1.5))
    return (1 - pen) * 100


def _ingredient_score(detected):
    if not detected: return None
    ratings = [d["gemini_rating"] for d in detected]
    proc = np.mean([d["processing_penalty"] for d in detected])
    bonus = np.mean([d["whole_food_bonus"] for d in detected])
    base = (np.mean(ratings) / 5.0) * 100
    return float(np.clip(base - proc * 12 + bonus * 8, 0, 100))


def compute_score(detected, nutrition):
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


def get_grade(score):
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "E"


def classify_health(score):
    if score >= 70: return "✅ Healthy"
    if score >= 40: return "⚠️ Moderate"
    return "🚨 Unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Explanation
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(score, risk, detected, nutrition):
    avoid = [d for d in detected if d["classification"] == "harmful"]
    limit = [d for d in detected if d["classification"] == "caution"]
    worst = sorted(avoid + limit, key=lambda d: d["gemini_rating"])[:5]
    lines = []
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

    sugar = nutrition.get("sugar", 0)
    if sugar > 22.5: lines.append(f"🔴 **High sugar** ({sugar:.1f} g)")
    elif sugar > 5: lines.append(f"🟡 **Moderate sugar** ({sugar:.1f} g)")
    fat = nutrition.get("fat", 0)
    if fat > 17.5: lines.append(f"🔴 **High fat** ({fat:.1f} g)")
    elif fat > 3: lines.append(f"🟡 **Moderate fat** ({fat:.1f} g)")
    sodium = nutrition.get("sodium", 0)
    if sodium > 1.5: lines.append(f"🔴 **High sodium** ({sodium:.2f} g)")
    elif sodium > 0.3: lines.append(f"🟡 **Moderate sodium** ({sodium:.2f} g)")

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
    if score >= 70: lines.append("A reasonable choice — maintain variety.")
    elif score >= 40: lines.append("Limit to a few servings per week; pair with whole foods.")
    else: lines.append("Prefer less-processed alternatives.")
    lines.append("> *Educational analysis. Consult a nutritionist for personal advice.*")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LabelAnalysis:
    __slots__ = (
        "ocr_result", "zones", "detected", "harmful", "nutrition",
        "nutrition_display", "score", "grade", "risk",
        "h_cnt", "c_cnt", "s_cnt",
    )

    def __init__(self):
        self.ocr_result: Optional[OCRResult] = None
        self.zones: Optional[TextZones] = None
        self.detected: list[dict] = []
        self.harmful: list[str] = []
        self.nutrition: dict[str, float] = {}
        self.nutrition_display: list[dict] = []
        self.score: int = 50
        self.grade: str = "C"
        self.risk: str = "⚠️ Moderate"
        self.h_cnt = self.c_cnt = self.s_cnt = 0


def analyze_label(image: Image.Image, langs=None, preprocess=True) -> LabelAnalysis:
    a = LabelAnalysis()
    a.ocr_result = run_surya_ocr(image, langs, preprocess)
    raw_text = a.ocr_result.text
    a.zones = split_text_zones(raw_text)
    a.nutrition = parse_nutritional_values(a.zones.for_nutrition_parsing)
    a.nutrition_display = nutrition_for_display(a.nutrition)
    a.detected = detect_ingredients(a.zones.for_ingredient_matching)
    a.harmful = [d["ingredient"] for d in a.detected
                 if d["classification"] in ("harmful", "caution")]
    a.h_cnt, a.c_cnt, a.s_cnt = count_concerns(a.detected)
    a.score = compute_score(a.detected, a.nutrition)
    a.grade = get_grade(a.score)
    a.risk = classify_health(a.score)
    a.ocr_result.annotated_image = draw_ocr_overlay(image, a.ocr_result)
    return a
