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


def _ocr_tesseract_lines(image: Image.Image, languages: list[str], preprocess: bool) -> list[OCRLine]:
    """
    Run Tesseract and return OCRLine objects with bounding boxes.
    Uses pytesseract.image_to_data() which gives per-word bbox + confidence.
    Words are grouped back into lines using Tesseract's own line/block numbers.
    """
    import pytesseract

    tess = "+".join(_TESS_LANG.get(l, "eng") for l in languages) or "eng"
    work = _binarise_for_tesseract(image) if preprocess else image

    try:
        df = pytesseract.image_to_data(
            work, lang=tess,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        # Fallback: plain string wrapped as lines without bboxes.
        try:
            plain = pytesseract.image_to_string(work, lang=tess, config="--oem 3 --psm 6")
        except Exception:
            plain = pytesseract.image_to_string(work, lang="eng", config="--oem 3 --psm 6")
        return [OCRLine(l.strip(), engine="tesseract")
                for l in plain.splitlines() if l.strip()]

    # Group words into lines keyed by (block_num, par_num, line_num).
    from collections import defaultdict
    line_words: dict[tuple, list[dict]] = defaultdict(list)
    n = len(df["text"])
    for i in range(n):
        word = df["text"][i].strip()
        conf = int(df["conf"][i]) if df["conf"][i] != "-1" else 0
        if not word or conf < 0:
            continue
        key = (df["block_num"][i], df["par_num"][i], df["line_num"][i])
        line_words[key].append({
            "text": word,
            "conf": conf,
            "left": df["left"][i],
            "top": df["top"][i],
            "width": df["width"][i],
            "height": df["height"][i],
        })

    # Build OCRLine per group, scaling bbox back to original image coords.
    # Tesseract runs on the preprocessed (potentially upscaled) image;
    # we need to map back to original dimensions.
    orig_w, orig_h = image.size
    proc_w, proc_h = work.size if hasattr(work, "size") else (orig_w, orig_h)
    sx = orig_w / proc_w if proc_w else 1.0
    sy = orig_h / proc_h if proc_h else 1.0

    lines_out: list[OCRLine] = []
    for key in sorted(line_words):
        words = line_words[key]
        text = " ".join(w["text"] for w in words)
        avg_conf = sum(w["conf"] for w in words) / len(words)
        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)
        bbox = (left * sx, top * sy, (right - left) * sx, (bottom - top) * sy)
        lines_out.append(OCRLine(text, bbox=bbox, confidence=avg_conf, engine="tesseract"))

    return lines_out


def _ocrspace_key() -> Optional[str]:
    """Read the OCR.space API key from env var or Streamlit secrets."""
    key = os.environ.get("OCR_SPACE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("OCR_SPACE_API_KEY")
    except Exception:
        return None


# ── OCR result container ─────────────────────────────────────────────────────

class OCRLine:
    """A single line recognised by OCR, with its bounding box."""
    __slots__ = ("text", "bbox", "confidence", "engine")

    def __init__(self, text: str, bbox: tuple | None = None,
                 confidence: float = 0.0, engine: str = ""):
        self.text = text
        self.bbox = bbox          # (x, y, w, h) in image pixels
        self.confidence = confidence
        self.engine = engine

    def __repr__(self):
        return f"OCRLine({self.text!r}, bbox={self.bbox})"


class OCRResult:
    """Full OCR output: raw text + per-line bounding boxes + annotated image."""
    def __init__(self):
        self.lines: list[OCRLine] = []
        self.annotated_image: Optional[Image.Image] = None
        self._scale: float = 1.0  # preprocessing scale factor

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines if ln.text.strip())


# ── Nutrition-keyword set for colour-coding boxes ────────────────────────────

_NUTRITION_KEYWORDS = {
    "energy", "calories", "kcal", "cal", "kj", "fat", "saturated",
    "carbohydrate", "carbohydrates", "sugar", "sugars", "fibre", "fiber",
    "protein", "sodium", "salt", "cholesterol", "serving", "amount",
    "daily", "value", "nutrition", "nutritional", "facts", "per",
    "total", "trans", "vitamin", "iron", "calcium", "potassium",
}


def _is_nutrition_line(text: str) -> bool:
    """Heuristic: does the line look like part of a nutrition table?"""
    words = set(re.findall(r"[a-z]+", text.lower()))
    # Nutrition lines typically contain a keyword + a number
    has_keyword = bool(words & _NUTRITION_KEYWORDS)
    has_number = bool(re.search(r"\d", text))
    return has_keyword and has_number


# ── Better preprocessing ─────────────────────────────────────────────────────

def _prepare_for_cloud_ocr(image: Image.Image) -> tuple[Image.Image, float]:
    """
    Aggressively enhance a photographed label for cloud OCR.
    Returns (enhanced_image, scale_factor).
    """
    from PIL import ImageEnhance

    img = image.copy()
    w, h = img.size

    # Upscale to ~2500px on the long side for maximum text resolution.
    target = 2500
    if max(w, h) < target:
        scale = target / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        scale = 1.0

    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img, scale


def _shrink_for_upload(image: Image.Image, max_bytes: int = 1_000_000) -> bytes:
    """JPEG-encode, shrinking quality until it fits under the API size limit."""
    import io
    for quality in (88, 75, 60, 45):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
        # Also try resizing down
        if quality <= 60:
            w, h = image.size
            image = image.resize((int(w * 0.8), int(h * 0.8)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=40)
    return buf.getvalue()


# ── OCR.space dual-engine with bounding boxes ────────────────────────────────

def _ocrspace_call(
    image_bytes: bytes,
    engine_id: int,
    key: str,
    overlay: bool = True,
    detect_orientation: bool = False,
    filetype: str = "JPG",
) -> dict:
    """Single OCR.space API call. Returns the raw JSON response."""
    import io
    import requests

    data: dict = {
        "language": "eng",
        "OCREngine": engine_id,
        "scale": True,
        "isOverlayRequired": "true" if overlay else "false",
        "detectOrientation": "true" if detect_orientation else "false",
        "filetype": filetype,
        "apikey": key,
    }
    resp = requests.post(
        "https://api.ocr.space/parse/image",
        files={"label.jpg": ("label.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        data=data,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _plain_text_from_ocrspace(data: dict) -> str:
    """Extract the plain ParsedText from an OCR.space response (no overlay needed)."""
    parts: list[str] = []
    for pr in data.get("ParsedResults", []):
        t = pr.get("ParsedText", "")
        if t:
            parts.append(t)
    return "\n".join(parts)


def _parse_ocrspace_overlay(data: dict, engine_label: str) -> list[OCRLine]:
    """Parse OCR.space overlay JSON into OCRLine objects."""
    lines_out: list[OCRLine] = []
    for pr in data.get("ParsedResults", []):
        overlay = pr.get("TextOverlay", {})
        for api_line in overlay.get("Lines", []):
            words = api_line.get("Words", [])
            if not words:
                continue
            text = " ".join(w["WordText"] for w in words)
            # Build a bounding box that encloses all words on this line.
            xs = [w["Left"] for w in words]
            ys = [w["Top"] for w in words]
            x2s = [w["Left"] + w["Width"] for w in words]
            y2s = [w["Top"] + w["Height"] for w in words]
            bbox = (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))
            conf = np.mean([w.get("Confidence", 0) for w in words]) if words else 0
            lines_out.append(OCRLine(text, bbox, confidence=conf, engine=engine_label))
    return lines_out


def _normalise_line(text: str) -> str:
    """Strip punctuation/spaces for fuzzy duplicate detection."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _merge_lines(lines1: list[OCRLine], lines2: list[OCRLine]) -> list[OCRLine]:
    """
    Merge OCR lines from two engines.

    Engine 2 (photo) is the base. Lines from Engine 1 (table) are added
    unless they are *normalised-text* duplicates of an already-accepted line.
    Normalisation strips punctuation and spaces so "Total Fat 24g" and
    "Total Fat 24 g" are treated as the same line, but "Fat 24g" and
    "Saturated Fat 8g" are kept as distinct.

    For nutrition lines specifically we prefer Engine 1 (table engine) because
    it handles column-aligned numbers better — if both engines produced a line
    for the same y-region, keep the Engine-1 version.
    """
    # Build a fast normalised-text lookup from lines1 (E2 base).
    seen_norm: dict[str, OCRLine] = {}
    for ln in lines1:
        seen_norm[_normalise_line(ln.text)] = ln

    merged: list[OCRLine] = list(lines1)

    for ln in lines2:
        key = _normalise_line(ln.text)
        if not key:
            continue
        if key in seen_norm:
            # If E1 version is a nutrition line and existing isn't, prefer E1.
            existing = seen_norm[key]
            if _is_nutrition_line(ln.text) and not _is_nutrition_line(existing.text):
                # Replace existing with better E1 version.
                try:
                    idx = merged.index(existing)
                    merged[idx] = ln
                    seen_norm[key] = ln
                except ValueError:
                    pass
            continue  # duplicate — skip
        merged.append(ln)
        seen_norm[key] = ln

    # Sort top-to-bottom by vertical bbox position.
    merged.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0))
    return merged


def _prepare_table_crop(image: Image.Image) -> Image.Image:
    """
    High-contrast greyscale version of the image, tuned for OCR of
    nutrition-facts tables (helps when the table has a white/light background).
    """
    from PIL import ImageEnhance
    # Convert to greyscale, boost contrast aggressively, then back to RGB
    # so OCR.space receives a standard colour image.
    grey = ImageOps.grayscale(image)
    grey = ImageOps.autocontrast(grey, cutoff=2)
    grey = ImageEnhance.Contrast(grey).enhance(1.8)
    grey = grey.filter(ImageFilter.SHARPEN)
    # Upscale if small — tables need resolution.
    w, h = grey.size
    if max(w, h) < 2000:
        scale = 2000 / max(w, h)
        grey = grey.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return grey.convert("RGB")


def _ocr_ocrspace_with_overlay(
    image: Image.Image, languages: list[str], preprocess: bool,
) -> OCRResult:
    """
    Run OCR.space Engine 2 (photo text) and Engine 1 (structured table text),
    merge results, and return bounding boxes for annotation.

    Strategy:
    - Engine 2 with overlay → good for curved/photographed text; gives bboxes.
    - Engine 1 with overlay → good for table columns; gives bboxes.
    - Engine 1 plain-text on table-enhanced image → extra pass to catch
      nutrition rows that the overlay mode misses due to column alignment.
    - orientation detection enabled so skewed labels are auto-rotated.
    """
    key = _ocrspace_key() or "helloworld"

    work, scale = _prepare_for_cloud_ocr(image) if preprocess else (image, 1.0)
    img_bytes = _shrink_for_upload(work)

    # Also prepare a table-optimised version of the image.
    table_work = _prepare_table_crop(image)
    table_bytes = _shrink_for_upload(table_work)

    result = OCRResult()
    result._scale = scale

    # ── Pass 1: Engine 2 with overlay (photo / curved text) ──────────────
    lines2: list[OCRLine] = []
    try:
        data2 = _ocrspace_call(img_bytes, engine_id=2, key=key,
                               overlay=True, detect_orientation=True)
        if data2.get("OCRExitCode") in (1, 2):
            lines2 = _parse_ocrspace_overlay(data2, "E2-photo")
    except Exception:
        pass

    # ── Pass 2: Engine 1 with overlay on standard image (table columns) ──
    lines1_overlay: list[OCRLine] = []
    try:
        data1 = _ocrspace_call(img_bytes, engine_id=1, key=key,
                               overlay=True, detect_orientation=True)
        if data1.get("OCRExitCode") in (1, 2):
            lines1_overlay = _parse_ocrspace_overlay(data1, "E1-table")
    except Exception:
        pass

    # ── Pass 3: Engine 1 plain-text on high-contrast table image ─────────
    # This pass has no bboxes but often extracts table rows that the overlay
    # mode misses when columns are close together or text is very small.
    extra_lines: list[OCRLine] = []
    try:
        data1t = _ocrspace_call(table_bytes, engine_id=1, key=key,
                                overlay=False, detect_orientation=False)
        if data1t.get("OCRExitCode") in (1, 2):
            plain = _plain_text_from_ocrspace(data1t)
            for raw_line in plain.splitlines():
                raw_line = raw_line.strip()
                if raw_line:
                    extra_lines.append(OCRLine(raw_line, bbox=None,
                                               confidence=0, engine="E1-table-plain"))
    except Exception:
        pass

    # ── Merge: E2 base → add E1 overlay → add E1 plain extras ───────────
    merged = _merge_lines(lines2, lines1_overlay)
    # For plain-text extras, only add lines NOT already in merged set.
    seen_norm = {_normalise_line(ln.text) for ln in merged}
    for ln in extra_lines:
        key_n = _normalise_line(ln.text)
        if key_n and key_n not in seen_norm:
            merged.append(ln)
            seen_norm.add(key_n)
    # Re-sort: lines without bbox go after positioned lines.
    merged.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else float("inf")))

    result.lines = merged
    return result


def _ocr_ocrspace(image: Image.Image, languages: list[str], preprocess: bool) -> str:
    """Backwards-compatible wrapper returning plain text."""
    return _ocr_ocrspace_with_overlay(image, languages, preprocess).text


# ── Draw bounding boxes on image ─────────────────────────────────────────────

def draw_ocr_overlay(
    image: Image.Image,
    ocr_result: OCRResult,
    box_width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on the image. Returns an annotated copy.

    Colours:
        🟩 Green  — nutrition-related lines (energy, fat, sugar …)
        🟦 Blue   — other text (ingredients, brand, etc.)
    Boxes are scaled back from the preprocessing resolution to the original.
    """
    from PIL import ImageDraw, ImageFont

    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    scale = ocr_result._scale

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  max(10, int(image.height * 0.012)))
    except Exception:
        font = ImageFont.load_default()

    for ln in ocr_result.lines:
        if not ln.bbox or not ln.text.strip():
            continue
        x, y, w, h = ln.bbox
        # Scale bbox back to original image coordinates.
        x, y, w, h = x / scale, y / scale, w / scale, h / scale
        is_nut = _is_nutrition_line(ln.text)
        colour = (34, 197, 94) if is_nut else (59, 130, 246)       # green / blue
        fill_bg = (34, 197, 94, 35) if is_nut else (59, 130, 246, 35)

        # Draw filled semi-transparent rectangle + outline.
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x, y, x + w, y + h], fill=fill_bg)
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(annotated)
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=box_width)

        # Label the box.
        label = "NUT" if is_nut else ""
        if label:
            tw = draw.textlength(label, font=font)
            draw.rectangle([x, y - 14, x + tw + 6, y], fill=colour)
            draw.text((x + 3, y - 13), label, fill="white", font=font)

    return annotated


# ── Engine registry (updated) ────────────────────────────────────────────────

_ENGINES = {
    "ocrspace": _ocr_ocrspace,
    "paddleocr": _ocr_paddle,
    "easyocr": _ocr_easy,
    "tesseract": _ocr_tesseract,
}
_AUTO_ORDER = ["ocrspace", "paddleocr", "easyocr", "tesseract"]

# Combined mode: run OCR.space + Tesseract in parallel and merge.
_COMBINED_ENGINE = "ocrspace+tesseract"


def _run_combined_ocr(
    image: Image.Image, languages: list[str], preprocess: bool,
) -> OCRResult:
    """
    Run OCR.space (cloud, 3 passes) and Tesseract (local) concurrently,
    then merge all lines with the same fuzzy-dedup logic used elsewhere.

    OCR.space is strong on photographed/curved text.
    Tesseract (binarised) is strong on clean printed text and table rows
    with high contrast.  Together they cover each other's blind spots.
    """
    import concurrent.futures

    ocrspace_result: OCRResult = OCRResult()
    tess_lines: list[OCRLine] = []

    def run_ocrspace():
        return _ocr_ocrspace_with_overlay(image, languages, preprocess)

    def run_tesseract():
        return _ocr_tesseract_lines(image, languages, preprocess)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_cloud = pool.submit(run_ocrspace)
        f_tess = pool.submit(run_tesseract)

        try:
            ocrspace_result = f_cloud.result(timeout=90)
        except Exception:
            pass
        try:
            tess_lines = f_tess.result(timeout=30)
        except Exception:
            pass

    # Merge: OCR.space as base (has bboxes), Tesseract fills gaps.
    merged = _merge_lines(ocrspace_result.lines, tess_lines)

    combined = OCRResult()
    combined.lines = merged
    combined._scale = ocrspace_result._scale
    return combined


def available_engines() -> list[str]:
    """Return OCR engines that can actually run in this environment."""
    found = []
    has_ocrspace = False
    has_tesseract = False
    try:
        import requests  # noqa: F401
        found.append("ocrspace")
        has_ocrspace = True
    except Exception:
        pass
    for name, mod in [("paddleocr", "paddleocr"), ("easyocr", "easyocr"),
                      ("tesseract", "pytesseract")]:
        try:
            __import__(mod)
            found.append(name)
            if name == "tesseract":
                has_tesseract = True
        except Exception:
            pass
    # Combined mode is available when both engines are present.
    if has_ocrspace and has_tesseract:
        found.insert(1, _COMBINED_ENGINE)  # second option, right after ocrspace
    return found


def ocrspace_key_configured() -> bool:
    """True if a non-demo OCR.space key is set (for UI hints)."""
    return bool(_ocrspace_key())


def extract_text_with_overlay(
    image: Image.Image,
    languages: Optional[list[str]] = None,
    engine: str = "auto",
    preprocess: bool = True,
) -> OCRResult:
    """
    Full OCR pipeline returning text + bounding boxes + annotated image.

    engine options:
      "auto"               — try OCR.space → PaddleOCR → EasyOCR → Tesseract
      "ocrspace+tesseract" — run BOTH concurrently and merge (recommended)
      "ocrspace"           — OCR.space only (3 passes, best on photos)
      "tesseract"          — Tesseract only (fast, best on clean prints)
      "paddleocr"          — PaddleOCR (local only, heavy)
      "easyocr"            — EasyOCR (local only, heavy)
    """
    if languages is None:
        languages = ["en"]

    # ── Combined mode: OCR.space + Tesseract in parallel ─────────────────
    if engine == _COMBINED_ENGINE:
        result = _run_combined_ocr(image, languages, preprocess)
        if result.lines:
            result.annotated_image = draw_ocr_overlay(image, result)
        return result

    order = _AUTO_ORDER if engine == "auto" else [engine]
    last_err: Optional[Exception] = None

    for name in order:
        try:
            if name == "ocrspace":
                result = _ocr_ocrspace_with_overlay(image, languages, preprocess)
                if result.lines:
                    result.annotated_image = draw_ocr_overlay(image, result)
                    return result
            elif name == "tesseract":
                tess_lines = _ocr_tesseract_lines(image, languages, preprocess)
                if tess_lines:
                    result = OCRResult()
                    result.lines = tess_lines
                    # Tesseract lines have bboxes — draw overlay too.
                    result.annotated_image = draw_ocr_overlay(image, result)
                    return result
            else:
                fn = _ENGINES.get(name)
                if fn is None:
                    continue
                text = fn(image, languages, preprocess)
                if text and text.strip():
                    result = OCRResult()
                    for line in text.split("\n"):
                        if line.strip():
                            result.lines.append(OCRLine(line.strip(), engine=name))
                    return result
        except Exception as e:
            last_err = e
            continue

    if last_err and engine != "auto":
        raise last_err
    return OCRResult()


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
    return extract_text_with_overlay(image, languages, engine, preprocess).text


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text cleaning  +  Zone splitting
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


# ── Zone classification ──────────────────────────────────────────────────────

# Headers that signal the start of an ingredient list.
_INGREDIENT_HEADERS = re.compile(
    r"(?:ingredients|composition|made\s+(?:from|with))\s*[:\-]?",
    re.IGNORECASE,
)

# Headers / keywords that signal a nutrition table.
_NUTRITION_HEADERS = re.compile(
    r"(?:nutrition\s*(?:facts|information|value|per)|amount\s+per|daily\s+value|"
    r"per\s+(?:serving|\d+\s*[gm]l?)|valeur\s+nutritive)",
    re.IGNORECASE,
)

# A line is "nutritional" if it contains a nutrient keyword followed by a
# number, typical of table rows like "Total Fat 24g" or "Sodium 480 mg".
_NUTRITION_ROW = re.compile(
    r"(?:energy|calories?|kcal|total\s*fat|trans\s*fat|saturated|cholesterol|"
    r"sodium|salt|carbohydrate|sugar|fibre|fiber|protein|vitamin|calcium|"
    r"iron|potassium|daily\s*value|serving|amount)[^\n]{0,30}\d",
    re.IGNORECASE,
)


class TextZones:
    """OCR text split into functional zones of a product label."""
    __slots__ = ("nutrition", "ingredients", "other", "full")

    def __init__(self, nutrition: str, ingredients: str, other: str, full: str):
        self.nutrition = nutrition      # text from nutrition facts table
        self.ingredients = ingredients  # text from ingredient list
        self.other = other              # everything else (brand, instructions …)
        self.full = full

    @property
    def for_ingredient_matching(self) -> str:
        """
        Text used for ingredient detection.

        Uses ingredient zone + other zone (i.e. everything EXCEPT the
        nutrition table). This handles labels that don't have an
        "Ingredients:" header — the comma-separated list just sits in
        'other' — while still excluding nutrition-table rows.
        """
        parts = []
        if self.ingredients.strip():
            parts.append(self.ingredients)
        if self.other.strip():
            parts.append(self.other)
        return "\n".join(parts) if parts else self.full

    @property
    def for_nutrition_parsing(self) -> str:
        """Text that should be used for nutrition value parsing."""
        if self.nutrition.strip():
            return self.nutrition
        return self.full


def _preprocess_ocr_text(text: str) -> str:
    """
    Normalise common OCR artifacts before zone splitting.

    - Pipe characters that OCR reads from table column separators → space.
    - Multiple spaces/tabs → single space.
    - Lines that are pure percentage/number (e.g. "24%", "0 g") are kept
      so nutrition rows aren't split across lines.
    """
    lines_out: list[str] = []
    for line in text.splitlines():
        # Replace pipe table separators with spaces.
        line = line.replace("|", " ")
        # Collapse repeated spaces/tabs.
        line = re.sub(r"[ \t]{2,}", " ", line)
        lines_out.append(line)
    return "\n".join(lines_out)


# How many consecutive non-nutrition lines to tolerate before exiting the
# nutrition zone.  Nutrition tables often have lines like "% Daily Value"
# or unit lines ("g", "%") that don't match _NUTRITION_ROW; we allow a
# small run of them before deciding the table is over.
_NUTRITION_ZONE_TOLERANCE = 3


def split_text_zones(text: str) -> TextZones:
    """
    Split OCR text into nutrition-table, ingredient-list and other zones.

    This is critical to prevent words like "sugar", "sodium", "trans fat",
    "cholesterol" from the nutrition table being falsely flagged as
    harmful *ingredients*.

    Heuristic (line by line):
    1. Lines after an "Ingredients:" header → ingredient zone
       (until the next recognisable header or a blank line).
    2. Lines after a Nutrition Facts header OR matching nutrition-row patterns
       → nutrition zone.  A tolerance window (_NUTRITION_ZONE_TOLERANCE) lets
       short non-matching lines (unit rows, "%" lines) stay in the nutrition
       zone instead of prematurely ending it.
    3. Everything else → other zone.
    """
    text = _preprocess_ocr_text(text)
    lines = text.split("\n")
    nut_lines: list[str] = []
    ing_lines: list[str] = []
    other_lines: list[str] = []

    zone = "other"
    nut_miss_streak = 0  # consecutive non-nutrition lines while in nutrition zone

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if zone == "ingredients":
                zone = "other"  # blank line ends ingredient list
            # A blank line in nutrition zone counts against tolerance.
            elif zone == "nutrition":
                nut_miss_streak += 1
                if nut_miss_streak > _NUTRITION_ZONE_TOLERANCE:
                    zone = "other"
                    nut_miss_streak = 0
            continue

        # ── Zone-starting headers ────────────────────────────────────────
        if _INGREDIENT_HEADERS.search(stripped):
            zone = "ingredients"
            nut_miss_streak = 0
            after_colon = re.split(r"[:;\-]\s*", stripped, maxsplit=1)
            if len(after_colon) > 1 and after_colon[1].strip():
                ing_lines.append(after_colon[1].strip())
            continue

        if _NUTRITION_HEADERS.search(stripped):
            zone = "nutrition"
            nut_miss_streak = 0
            continue

        # ── Ingredient zone accumulation ─────────────────────────────────
        if zone == "ingredients":
            ing_lines.append(stripped)
            continue

        # ── Nutrition zone ───────────────────────────────────────────────
        if _NUTRITION_ROW.search(stripped):
            nut_lines.append(stripped)
            zone = "nutrition"
            nut_miss_streak = 0
            continue

        if zone == "nutrition":
            # Tolerate short numeric/unit-only lines inside the table.
            looks_like_table_fragment = bool(
                re.match(r"^[\d\s%gmlkj./<>()]+$", stripped, re.IGNORECASE)
                or len(stripped) <= 6
            )
            if looks_like_table_fragment:
                nut_lines.append(stripped)
                # Don't increment miss streak for these fragments.
            else:
                nut_miss_streak += 1
                if nut_miss_streak > _NUTRITION_ZONE_TOLERANCE:
                    zone = "other"
                    nut_miss_streak = 0
                    other_lines.append(stripped)
                else:
                    # Keep in nutrition zone tentatively.
                    nut_lines.append(stripped)
            continue

        # ── Default: other ───────────────────────────────────────────────
        other_lines.append(stripped)

    return TextZones(
        nutrition="\n".join(nut_lines),
        ingredients="\n".join(ing_lines),
        other="\n".join(other_lines),
        full=text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ingredient detection (dataset-driven)
# ─────────────────────────────────────────────────────────────────────────────

def detect_ingredients(text: str) -> list[dict]:
    """
    Match dataset ingredients against text; returns metadata dicts.

    Matchers are processed longest-first and claim character spans, so a
    shorter fragment fully inside an already-matched longer term (e.g.
    "flavor" inside "artificial flavor") is suppressed.

    IMPORTANT: pass only the *ingredient-zone* text (not the nutrition
    table) to avoid false positives from nutrient names.
    """
    matchers, meta = _build_index()
    haystack = " " + text.lower() + " "
    found: dict[str, dict] = {}
    claimed: list[tuple[int, int]] = []

    # Terms that are nutrient / table labels, not meaningful ingredients.
    # Even if they appear in the ingredient zone, flagging "sugar" as a
    # harmful ingredient is misleading — it's too generic.
    _NUTRIENT_NOISE = {
        "sugar", "sugars", "sodium", "cholesterol", "fat", "trans fat",
        "saturated fat", "total fat", "calories", "protein", "carbohydrate",
        "carbohydrates", "fibre", "fiber", "energy",
        # Generic words that appear on labels but aren't specific ingredients:
        "powder", "soup", "flavors", "flavor", "colour", "color",
        "extract", "concentrate", "blend",
    }

    for name, pattern in matchers:
        if name in found:
            continue
        if name in _NUTRIENT_NOISE:
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
    # Calories / Energy: handle "240 kcal", "1004 kJ", "Cal 240", kJ→kcal later
    "calories": [
        r"(?:energy|calories?)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*(kcal|cal|kj)",
        r"(?:energy|calories?)[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(kcal|cal|kj)?",
        r"(\d{2,4})\s*(kcal|cal)\b",
    ],
    # Sugar: "Sugars 12g", "Total Sugars 12 g", "Of which sugars 12g"
    "sugar": [
        r"(?:total\s+)?sugars?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"of\s+which\s+sugars?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"sugars?\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)\s*(mg|g)?\b",
    ],
    # Fat: "Total Fat 24g", "Fat 24 g", handle pipe separators
    "fat": [
        r"total\s+fat\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"(?<!\w)fat\b[^\d\n|]{0,15}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"(?<!\w)fat\b[^\d\n|]{0,8}(\d+(?:\.\d+)?)()\b",
    ],
    # Sodium: "Sodium 480mg", "Salt 1.2g" — keep mg separate (don't auto-convert until parse)
    "sodium": [
        r"sodium\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"salt\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"sodium\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)(mg)?",
    ],
    # Protein
    "protein": [
        r"protein\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"protein\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)(g)?",
    ],
    # Carbohydrates
    "carbs": [
        r"(?:total\s+)?carbohydrates?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"carbs?\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)\s*(mg|g)?\b",
    ],
    # Dietary fibre/fiber
    "fibre": [
        r"dietary\s+fi[be]re?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"fi[be]re?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"fi[be]re?\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)(g)?",
    ],
    # Saturated fat
    "saturated": [
        r"saturated\s+fat\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"saturates?\b[^\d\n]{0,20}(\d+(?:\.\d+)?)\s*(mg|g)\b",
        r"saturated\b[^\d\n|]{0,10}(\d+(?:\.\d+)?)(g)?",
    ],
}


def parse_nutritional_values(raw_text: str) -> dict[str, float]:
    """
    Extract numeric nutritional values (per 100 g/ml).

    Improvements over original:
    - Tries matching on the raw text AND on a version where newlines between
      short lines are collapsed (handles OCR that splits "Fat" / "24g" across
      two lines).
    - Converts mg → g for non-calorie nutrients.
    - Converts kJ → kcal for energy (÷ 4.184).
    - Skips values that are clearly %DV (followed by "%") rather than absolute.
    """
    result = {k: 0.0 for k in _NUTRIENT_PATTERNS}

    # Build two search targets: raw lower, and a single-line collapse
    # (join lines that are short — pure numbers/units — to their predecessor).
    raw_lower = raw_text.lower()
    collapsed_lines: list[str] = []
    for line in raw_lower.splitlines():
        if collapsed_lines and re.match(r"^\s*[\d.]+\s*(?:g|mg|kcal|kj|cal)?\s*$", line):
            collapsed_lines[-1] = collapsed_lines[-1].rstrip() + " " + line.strip()
        else:
            collapsed_lines.append(line)
    collapsed_lower = "\n".join(collapsed_lines)

    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        for target in (raw_lower, collapsed_lower):
            for pattern in patterns:
                m = re.search(pattern, target)
                if not m:
                    continue
                try:
                    value = float(m.group(1))
                except (ValueError, IndexError):
                    continue

                # Skip if the match is immediately followed by "%" (it's a %DV).
                end_ctx = target[m.end():m.end() + 3].strip()
                if end_ctx.startswith("%"):
                    continue

                unit = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
                unit = unit.strip().lower()

                if nutrient == "calories":
                    if unit == "kj":
                        value = value / 4.184  # kJ → kcal
                elif unit == "mg":
                    value /= 1000.0  # mg → g

                if value > 0:
                    result[nutrient] = value
                    break  # found for this nutrient; stop trying patterns
            if result[nutrient] > 0:
                break  # found on first target; don't re-search collapsed

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


# ─────────────────────────────────────────────────────────────────────────────
# 7. High-level pipeline (zone-aware)
# ─────────────────────────────────────────────────────────────────────────────

class LabelAnalysis:
    """All results from analysing a single label image."""
    __slots__ = (
        "zones", "detected", "harmful", "nutrition",
        "score", "grade", "risk", "h_cnt", "c_cnt", "s_cnt",
    )

    def __init__(self):
        self.zones: Optional[TextZones] = None
        self.detected: list[dict] = []
        self.harmful: list[str] = []
        self.nutrition: dict[str, float] = {}
        self.score: int = 50
        self.grade: str = "C"
        self.risk: str = "⚠️ Moderate"
        self.h_cnt = self.c_cnt = self.s_cnt = 0


def analyze_label(raw_text: str) -> LabelAnalysis:
    """
    Full zone-aware analysis pipeline.

    1. Split OCR text into nutrition / ingredient / other zones.
    2. Parse nutrition only from the nutrition zone.
    3. Match ingredients only from the ingredient zone.
    4. Compute score from both signals.
    """
    a = LabelAnalysis()
    a.zones = split_text_zones(raw_text)

    # Parse nutrition from nutrition zone (not full text).
    a.nutrition = parse_nutritional_values(a.zones.for_nutrition_parsing)

    # Match ingredients from ingredient zone (not nutrition table).
    a.detected = detect_ingredients(a.zones.for_ingredient_matching)
    a.harmful = [d["ingredient"] for d in a.detected
                 if d["classification"] in ("harmful", "caution")]
    a.h_cnt, a.c_cnt, a.s_cnt = count_concerns(a.detected)

    a.score = compute_score(a.detected, a.nutrition)
    a.grade = get_grade(a.score)
    a.risk = classify_health(a.score)
    return a
