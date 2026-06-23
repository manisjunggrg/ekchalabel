"""
utils.py – Core logic for e-K Cha Label?

Pipeline: OCR (multi-engine) -> text cleaning -> dataset-driven ingredient
matching -> nutrition parsing -> data-driven health score -> grade + verdict.

Ingredient intelligence comes from `eatsafe_master_database.csv` (≈8.7k
ingredients) with columns:
    ingredient, nova_group, additive_risk, artificial_flag,
    gemini_rating, reason, processing_penalty, nutrition_impact,
    whole_food_bonus
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

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eatsafe_master_database.csv")

# additive_risk -> our display classification / risk level
_RISK_TO_CLASS = {"avoid": "harmful", "limit": "caution", "safe": "safe"}
_RISK_TO_LEVEL = {"avoid": "High", "limit": "Moderate", "safe": "Safe"}

_NOVA_LABEL = {
    1: "NOVA 1 · Unprocessed / minimally processed",
    2: "NOVA 2 · Processed culinary ingredient",
    3: "NOVA 3 · Processed food",
    4: "NOVA 4 · Ultra-processed",
}

# Very short ingredient names that would false-match common label text
# (e.g. "fat" in "Total Fat"). Only these short tokens are allowed to match.
_SHORT_ALLOW = {"msg", "bha", "bht", "egg", "soy", "oat", "nut", "b12", "b9", "d3"}


def nova_label(group) -> str:
    try:
        return _NOVA_LABEL.get(int(group), f"NOVA {group}")
    except (ValueError, TypeError):
        return "NOVA n/a"


@functools.lru_cache(maxsize=1)
def load_ingredient_db(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and normalise the EatSafe ingredient database."""
    df = pd.read_csv(path)
    df["ingredient"] = df["ingredient"].fillna("").astype(str).str.strip().str.lower()
    df["additive_risk"] = df["additive_risk"].fillna("limit").astype(str).str.lower()
    df["reason"] = df["reason"].fillna("").astype(str).str.strip()

    for col in ["gemini_rating", "processing_penalty", "nutrition_impact", "whole_food_bonus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce").fillna(0).astype(int)

    df["classification"] = df["additive_risk"].map(_RISK_TO_CLASS).fillna("caution")
    df["risk_level"] = df["additive_risk"].map(_RISK_TO_LEVEL).fillna("Moderate")
    df = df[df["ingredient"] != ""].drop_duplicates(subset="ingredient")
    return df.reset_index(drop=True)


def _matchable(term: str) -> bool:
    """Allow a term to be matched only if it's long enough or whitelisted."""
    t = term.strip().lower()
    if not t:
        return False
    if len(t) >= 4:
        return True
    return t in _SHORT_ALLOW


@functools.lru_cache(maxsize=1)
def _build_index():
    """Compile a word-boundary regex per matchable ingredient (cached)."""
    df = load_ingredient_db()
    matchers: list[tuple[str, "re.Pattern"]] = []
    meta: dict[str, dict] = {}
    for row in df.itertuples(index=False):
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
        if _matchable(name):
            matchers.append((name, re.compile(r"\b" + re.escape(name) + r"\b")))
    # Longest names first so multi-word ingredients win over substrings.
    matchers.sort(key=lambda x: len(x[0]), reverse=True)
    return matchers, meta


# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR  (multi-engine: PaddleOCR → EasyOCR → Tesseract)
# ─────────────────────────────────────────────────────────────────────────────

# UI codes mapped per engine
_TESS_LANG = {"en": "eng", "ne": "nep", "hi": "hin"}
_PADDLE_LANG = {"en": "en", "ne": "ne", "hi": "hi"}

# Cached engine objects (model loading is expensive)
_EASY_READERS: dict[tuple, object] = {}
_PADDLE_OCR: dict[str, object] = {}


def _enhance_colour(image: Image.Image) -> Image.Image:
    """Light enhancement for deep-learning OCR (keeps colour, no binarising)."""
    img = image
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return img.filter(ImageFilter.SHARPEN)


def _binarise_for_tesseract(image: Image.Image) -> Image.Image:
    """Heavy preprocessing Tesseract needs: grayscale + denoise + threshold."""
    gray = ImageOps.grayscale(image)
    w, h = gray.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    arr = np.array(gray)
    try:
        import cv2
        arr = cv2.fastNlMeansDenoising(arr, h=10)
        binary = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        return Image.fromarray(binary)
    except Exception:
        thr = int(arr.mean())
        return Image.fromarray((arr > thr).astype(np.uint8) * 255)


def _ocr_paddle(image: Image.Image, languages: list[str], preprocess: bool) -> str:
    from paddleocr import PaddleOCR

    lang = "en"
    for l in languages:
        if l in ("hi", "ne"):
            lang = _PADDLE_LANG[l]
            break
    if lang not in _PADDLE_OCR:
        _PADDLE_OCR[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    ocr = _PADDLE_OCR[lang]

    work = _enhance_colour(image) if preprocess else image
    result = ocr.ocr(np.array(work), cls=True)
    lines: list[str] = []
    for page in result or []:
        for entry in page or []:
            try:
                lines.append(entry[1][0])
            except (IndexError, TypeError):
                pass
    return "\n".join(lines)


def _ocr_easy(image: Image.Image, languages: list[str], preprocess: bool) -> str:
    import easyocr

    langs = tuple(languages) if languages else ("en",)
    if langs not in _EASY_READERS:
        try:
            _EASY_READERS[langs] = easyocr.Reader(list(langs), gpu=False, verbose=False)
        except Exception:
            _EASY_READERS[langs] = easyocr.Reader(["en"], gpu=False, verbose=False)
    reader = _EASY_READERS[langs]

    work = _enhance_colour(image) if preprocess else image
    results = reader.readtext(np.array(work), detail=0, paragraph=True)
    return "\n".join(results)


def _ocr_tesseract(image: Image.Image, languages: list[str], preprocess: bool) -> str:
    import pytesseract

    tess = "+".join(_TESS_LANG.get(l, "eng") for l in languages) or "eng"
    work = _binarise_for_tesseract(image) if preprocess else image
    try:
        return pytesseract.image_to_string(work, lang=tess, config="--oem 3 --psm 6")
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(work, lang="eng", config="--oem 3 --psm 6")


def _ocrspace_key() -> Optional[str]:
    """Read the OCR.space API key from env var or Streamlit secrets."""
    key = os.environ.get("OCR_SPACE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("OCR_SPACE_API_KEY")  # set in Cloud → Settings → Secrets
    except Exception:
        return None


def _ocr_ocrspace(image: Image.Image, languages: list[str], preprocess: bool) -> str:
    """
    Cloud OCR via the OCR.space API. Ideal for Streamlit Community Cloud:
    accurate on photos, runs server-side, negligible app memory.

    Needs an API key (free tier, no card) in env `OCR_SPACE_API_KEY` or in
    Streamlit secrets. Falls back to the public demo key if none is set.
    """
    import io
    import requests

    key = _ocrspace_key() or "helloworld"  # demo key: heavily rate-limited

    work = _enhance_colour(image) if preprocess else image
    # Keep the upload comfortably under the 1 MB free-tier limit.
    if max(work.size) > 2000:
        s = 2000 / max(work.size)
        work = work.resize((int(work.size[0] * s), int(work.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=80)
    buf.seek(0)

    # OCR Engine 2 reads photographed text best (Latin scripts).
    resp = requests.post(
        "https://api.ocr.space/parse/image",
        files={"label.jpg": ("label.jpg", buf, "image/jpeg")},
        data={"language": "eng", "OCREngine": 2, "scale": True,
              "isOverlayRequired": False, "apikey": key},
        timeout=40,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("OCRExitCode") not in (1, 2):
        raise RuntimeError(data.get("ErrorMessage") or "OCR.space request failed")
    return "\n".join(r.get("ParsedText", "") for r in data.get("ParsedResults", []))


_ENGINES = {
    "ocrspace": _ocr_ocrspace,
    "paddleocr": _ocr_paddle,
    "easyocr": _ocr_easy,
    "tesseract": _ocr_tesseract,
}
# Streamlit-Cloud-friendly order: cloud OCR (light + accurate) → local deep
# engines (if installed) → Tesseract fallback.
_AUTO_ORDER = ["ocrspace", "paddleocr", "easyocr", "tesseract"]


def available_engines() -> list[str]:
    """Return OCR engines that can actually run in this environment."""
    found = []
    # Cloud engine: needs `requests`; usable with a key (or demo key).
    try:
        import requests  # noqa: F401
        found.append("ocrspace")
    except Exception:
        pass
    for name, mod in [("paddleocr", "paddleocr"), ("easyocr", "easyocr"),
                      ("tesseract", "pytesseract")]:
        try:
            __import__(mod)
            found.append(name)
        except Exception:
            pass
    return found


def ocrspace_key_configured() -> bool:
    """True if a non-demo OCR.space key is set (for UI hints)."""
    return bool(_ocrspace_key())


def extract_text(
    image: Image.Image,
    languages: Optional[list[str]] = None,
    engine: str = "auto",
    preprocess: bool = True,
) -> str:
    """
    Extract text from a label image.

    engine: "auto" (try OCR.space → PaddleOCR → EasyOCR → Tesseract), or one
            of "ocrspace" / "paddleocr" / "easyocr" / "tesseract".
    On Streamlit Cloud, "ocrspace" (cloud) is the most reliable accurate
    option; local deep engines are heavy and Tesseract is weak on photos.
    """
    if languages is None:
        languages = ["en"]

    order = _AUTO_ORDER if engine == "auto" else [engine]
    last_err: Optional[Exception] = None
    for name in order:
        fn = _ENGINES.get(name)
        if fn is None:
            continue
        try:
            text = fn(image, languages, preprocess)
            if text and text.strip():
                return _tidy(text)
        except Exception as e:  # missing package / runtime error → try next
            last_err = e
            continue
    if last_err and engine != "auto":
        raise last_err
    return ""


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "and", "for", "per", "from", "this", "that", "are", "was", "may",
    "contains", "contain", "ingredients", "ingredient", "product", "value",
    "values", "serving", "size", "total", "net", "weight",
}


def clean_text(text: str) -> str:
    """Lowercase, keep alphabetic tokens, drop short/stop words."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        return " ".join(
            tok.lemma_ for tok in doc
            if not tok.is_stop and tok.is_alpha and len(tok.text) > 2
        )
    except Exception:
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        return " ".join(t for t in text.split() if len(t) > 2 and t not in _STOPWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ingredient detection (dataset-driven)
# ─────────────────────────────────────────────────────────────────────────────

def detect_ingredients(text: str) -> list[dict]:
    """
    Match dataset ingredients against raw OCR text; returns metadata dicts.

    Matchers are processed longest-first and claim character spans, so a
    shorter fragment fully inside an already-matched longer term (e.g.
    "flavor" inside "artificial flavor") is suppressed.
    """
    matchers, meta = _build_index()
    haystack = " " + text.lower() + " "
    found: dict[str, dict] = {}
    claimed: list[tuple[int, int]] = []
    for name, pattern in matchers:
        if name in found:
            continue
        for mt in pattern.finditer(haystack):
            s, e = mt.span()
            if any(cs <= s and e <= ce for cs, ce in claimed):
                continue  # nested inside a longer accepted match
            found[name] = meta[name]
            claimed.append((s, e))
            break
    order = {"harmful": 0, "caution": 1, "safe": 2}
    return sorted(
        found.values(),
        key=lambda d: (order.get(d["classification"], 3), d["gemini_rating"]),
    )


def detect_harmful_ingredients(text: str) -> list[str]:
    """Names of detected ingredients that are not classified 'safe'."""
    return [d["ingredient"] for d in detect_ingredients(text)
            if d["classification"] in ("harmful", "caution")]


def count_concerns(detected: list[dict]) -> tuple[int, int, int]:
    """(harmful, caution, safe) counts from a detected list."""
    h = sum(1 for d in detected if d["classification"] == "harmful")
    c = sum(1 for d in detected if d["classification"] == "caution")
    s = sum(1 for d in detected if d["classification"] == "safe")
    return h, c, s


# ─────────────────────────────────────────────────────────────────────────────
# 4. Nutritional value parsing
# ─────────────────────────────────────────────────────────────────────────────

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
    """Extract numeric nutritional values (per 100 g/ml). mg → g."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Health scoring  (data-driven)
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
    """0–100 from nutrition numbers, or None if nothing parsed."""
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
    """0–100 from matched ingredients' dataset ratings, or None."""
    if not detected:
        return None
    ratings = [d["gemini_rating"] for d in detected]          # 0–5
    proc = np.mean([d["processing_penalty"] for d in detected])
    bonus = np.mean([d["whole_food_bonus"] for d in detected])
    base = (np.mean(ratings) / 5.0) * 100
    return float(np.clip(base - proc * 12 + bonus * 8, 0, 100))


def compute_score(detected: list[dict], nutrition: dict[str, float]) -> int:
    """
    Blended 0–100 health score (higher = healthier).

    Uses dataset ingredient ratings (NOVA, processing, whole-food bonus)
    plus parsed nutrition. Extra penalty applied for 'avoid' additives.
    """
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


# Backwards-compatible wrapper (old signature)
def compute_health_score(nutrition: dict[str, float], additive_count: int = 0) -> int:
    score = _nutrition_score(nutrition)
    base = 50.0 if score is None else score
    base -= min(additive_count * 4, 20)
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
# 6. Verdict / explanation
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(score: int, risk: str, detected: list[dict],
                         nutrition: dict[str, float]) -> str:
    """Markdown plain-language summary using dataset signals."""
    flagged = [d for d in detected if d["classification"] in ("harmful", "caution")]
    avoid = [d for d in detected if d["classification"] == "harmful"]
    ultra = [d for d in detected if d["nova_group"] == 4]
    worst = sorted(flagged, key=lambda d: d["gemini_rating"])[:3]

    lines: list[str] = []
    if score >= 70:
        lines.append(f"### ✅ Overall: relatively healthy (score {score}/100)")
        lines.append("The ingredients are mostly minimally processed and the nutrition "
                     "looks acceptable. Enjoy as part of a balanced diet.")
    elif score >= 40:
        lines.append(f"### ⚠️ Overall: moderate concerns (score {score}/100)")
        lines.append("Fine for occasional consumption, but it shouldn't be a daily staple.")
    else:
        lines.append(f"### 🚨 Overall: not recommended for regular use (score {score}/100)")
        lines.append("Heavily processed ingredients and/or poor nutrition make this a weak "
                     "everyday choice.")
    lines.append("---")

    if detected:
        avg_rating = np.mean([d["gemini_rating"] for d in detected])
        lines.append(f"**Ingredient quality:** average dataset rating "
                     f"**{avg_rating:.1f}/5** across {len(detected)} recognised ingredient(s).")
    if ultra:
        lines.append(f"🏭 **{len(ultra)} ultra-processed (NOVA 4) ingredient(s)** detected: "
                     + ", ".join(d["ingredient"].title() for d in ultra[:6])
                     + ("…" if len(ultra) > 6 else "") + ".")

    sugar = nutrition.get("sugar", 0)
    if sugar > 22.5:
        lines.append(f"🔴 **High sugar** ({sugar} g/100g) — above the 22.5 g guideline.")
    elif sugar > 5:
        lines.append(f"🟡 **Moderate sugar** ({sugar} g/100g).")
    fat = nutrition.get("fat", 0)
    if fat > 17.5:
        lines.append(f"🔴 **High fat** ({fat} g/100g).")
    elif fat > 3:
        lines.append(f"🟡 **Moderate fat** ({fat} g/100g).")
    sodium = nutrition.get("sodium", 0)
    if sodium > 1.5:
        lines.append(f"🔴 **High sodium** ({sodium} g/100g).")

    if avoid:
        lines.append(f"⛔ **{len(avoid)} 'avoid' additive(s):** "
                     + ", ".join(d["ingredient"].title() for d in avoid[:6])
                     + ("…" if len(avoid) > 6 else "") + ".")
    if worst:
        lines.append("**Most concerning ingredients:**")
        for d in worst:
            lines.append(f"- **{d['ingredient'].title()}** ({d['gemini_rating']:.1f}/5) — {d['description']}")
    if not flagged:
        lines.append("🟢 **No 'limit' or 'avoid' additives** detected in the extracted text.")

    lines.append("---")
    lines.append("**💡 Recommendation:**")
    if score >= 70:
        lines.append("A reasonable choice — keep variety in your diet.")
    elif score >= 40:
        lines.append("Limit to a few servings per week; pair with whole foods.")
    else:
        lines.append("Prefer less-processed alternatives. Keep portions small and infrequent.")
    lines.append("> *Educational analysis only. Consult a qualified nutritionist for personal advice.*")

    return "\n\n".join(lines)
