"""
app.py – e-K Cha Label?

Design direction: "The Kitchen Counter"
You picked up a product, your knowledgeable friend tells you what's in it.
Warm, direct, no dashboard theatrics. Colours from real food: turmeric,
chili, leafy greens.
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    LabelAnalysis, analyze_label, load_ingredient_db,
    load_surya_models, _is_nutrition_line, generate_explanation,
)

st.set_page_config(page_title="e-K Cha Label?", page_icon="🍽️", layout="centered")

# ── Palette (kitchen colours: turmeric, chili, leafy green, slate) ───────────
GREEN  = "#2d6a4f"
AMBER  = "#d4940a"
RED    = "#b8332e"
SLATE  = "#546e8a"
DARK   = "#3a3535"
LIGHT  = "#f5f5f3"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&display=swap');
  html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: {DARK}; }}
  .block-container {{ max-width: 760px; padding-top: 2.2rem; }}
  h1, h2, h3 {{ font-family: 'DM Sans', sans-serif; font-weight: 700; letter-spacing: -0.02em; }}

  .site-header {{ margin-bottom: 2rem; }}
  .site-header h1 {{ font-size: 1.6rem; margin: 0; }}
  .site-header p {{ color: #7a7574; margin: 0.15rem 0 0; font-size: 0.92rem; }}

  .score-circle {{
    width: 96px; height: 96px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; font-weight: 700; color: #fff;
    margin: 0 auto;
  }}
  .score-label {{ text-align: center; margin-top: 6px; font-size: 0.82rem; color: #7a7574; }}
  .grade-letter {{ text-align: center; font-size: 3.2rem; font-weight: 700; line-height: 1; }}
  .grade-sub {{ text-align: center; font-size: 0.8rem; color: #7a7574; }}

  /* Real-world nutrition label style */
  .nut-label-box {{
    border: 3px solid {DARK}; padding: 0; margin: 1rem 0;
    font-size: 0.88rem; line-height: 1.4;
  }}
  .nut-label-box .nut-title {{
    font-size: 1.5rem; font-weight: 700; padding: 4px 8px;
    border-bottom: 1px solid {DARK};
  }}
  .nut-label-box .nut-row {{
    display: flex; justify-content: space-between;
    padding: 3px 8px; border-bottom: 1px solid #ddd;
  }}
  .nut-label-box .nut-row.thick {{ border-bottom: 5px solid {DARK}; }}
  .nut-label-box .nut-row.med {{ border-bottom: 3px solid {DARK}; }}
  .nut-label-box .nut-name {{ font-weight: 700; }}
  .nut-label-box .nut-val {{ text-align: right; }}
  .nut-label-box .nut-indent {{ padding-left: 24px; }}

  .ing-tag {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 0.84rem; margin: 2px; font-weight: 500;
  }}
  .ing-avoid {{ background: #f8d7d5; color: {RED}; }}
  .ing-limit {{ background: #fef3cd; color: {AMBER}; }}
  .ing-safe  {{ background: #d4edda; color: {GREEN}; }}

  .concern-item {{
    padding: 12px 16px; margin: 6px 0; border-radius: 6px;
    border-left: 4px solid; font-size: 0.9rem;
  }}
  .concern-item b {{ font-size: 0.95rem; }}
  .concern-item small {{ color: #7a7574; }}

  .explanation {{ background: {LIGHT}; border-radius: 8px; padding: 20px 24px;
    line-height: 1.7; font-size: 0.92rem; }}

  .section-label {{ font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: {SLATE}; font-weight: 500;
    margin: 2rem 0 0.5rem; }}

  .footer {{ color: #aaa; font-size: 0.78rem; margin-top: 3rem;
    padding-top: 1rem; border-top: 1px solid #eee; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
  <h1>e-K Cha Label?</h1>
  <p>Upload a product label. We'll read it, check the ingredients, and tell you what's worth knowing.</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OCR models (first run only)…")
def _load():
    return load_surya_models()

# ── Sidebar: just the database browser, nothing else ──────────────────────────
with st.sidebar:
    st.markdown("**Ingredient database**")
    try:
        _db = load_ingredient_db()
        vc = _db["additive_risk"].value_counts()
        st.caption(f"{len(_db):,} ingredients · "
                   f"{int(vc.get('avoid',0)):,} avoid · "
                   f"{int(vc.get('limit',0)):,} limit · "
                   f"{int(vc.get('safe',0)):,} safe")
        q = st.text_input("Search", placeholder="e.g. tartrazine")
        if q:
            matches = _db[_db["ingredient"].str.contains(q.lower(), na=False)]
            st.dataframe(matches[["ingredient","additive_risk","gemini_rating","reason"]],
                         use_container_width=True, hide_index=True, height=300)
        else:
            with st.expander("Browse all"):
                st.dataframe(_db[["ingredient","additive_risk","gemini_rating"]],
                             use_container_width=True, hide_index=True, height=280)
    except Exception as e:
        st.error(str(e))

# ── Input ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drop a label photo here", type=["jpg","jpeg","png","webp"])
cam = st.camera_input("or take a photo")

image = None
if uploaded:
    image = Image.open(uploaded).convert("RGB")
elif cam:
    image = Image.open(cam).convert("RGB")

if image is None:
    st.caption("Snap a photo of any packaged food label — we'll do the rest.")
    st.stop()

# ── Load models + analyse ─────────────────────────────────────────────────────
_load()

col_img, col_score = st.columns([1.4, 1], gap="medium")
with col_img:
    st.image(image, use_container_width=True)

try:
    with st.spinner("Reading the label…"):
        a: LabelAnalysis = analyze_label(image)
except Exception as e:
    st.error(f"Something went wrong: {e}")
    if st.button("Try again"):
        st.rerun()
    st.stop()

raw_text = a.ocr_result.text
if not raw_text.strip():
    st.warning("Couldn't read any text. Try a clearer photo with good lighting.")
    st.stop()

score, grade, risk = a.score, a.grade, a.risk
detected, nutrition = a.detected, a.nutrition
sc = GREEN if score >= 70 else AMBER if score >= 40 else RED

# ── Score display (simple circle + grade) ─────────────────────────────────────
with col_score:
    st.markdown(f"""
    <div style="padding-top:12px">
      <div class="score-circle" style="background:{sc}">{score}</div>
      <div class="score-label">out of 100</div>
      <div class="grade-letter" style="color:{sc};margin-top:16px">{grade}</div>
      <div class="grade-sub">nutri-grade</div>
    </div>
    """, unsafe_allow_html=True)

    # Counts as plain text
    st.markdown(f"""
    <div style="text-align:center;margin-top:18px;font-size:0.88rem">
      <span style="color:{RED}">{'●' * a.h_cnt}{'○' * max(0, 3 - a.h_cnt)}</span> {a.h_cnt} avoid &nbsp;
      <span style="color:{AMBER}">{'●' * a.c_cnt}{'○' * max(0, 3 - a.c_cnt)}</span> {a.c_cnt} limit &nbsp;
      <span style="color:{GREEN}">{'●' * min(a.s_cnt,5)}</span> {a.s_cnt} safe
    </div>
    """, unsafe_allow_html=True)

if a.zones and a.zones.ingredients.strip():
    product_context = f"from the ingredients list"
else:
    product_context = f"from the label text"

st.markdown(f"""
<p class="section-label">what we found</p>
<p style="font-size:0.92rem">
We read the label, matched <b>{len(detected)}</b> ingredients against our database of
{len(load_ingredient_db()):,} items, and parsed the nutrition table. Here's the breakdown.
</p>
""", unsafe_allow_html=True)

# ── Nutrition: styled like a real FDA/FSSAI label ─────────────────────────────
st.markdown('<p class="section-label">nutrition facts</p>', unsafe_allow_html=True)

if a.nutrition_display:
    rows_html = ""
    for i, nf in enumerate(a.nutrition_display):
        vs = f"{nf['value']:g}" if nf['value'] == int(nf['value']) else f"{nf['value']:.1f}"
        indent = "nut-indent" if nf["nutrient"].lower() in ("saturated","trans fat") else ""
        bold = "nut-name" if not indent else ""
        cls = "med" if i == 0 else ""
        rows_html += (f'<div class="nut-row {cls}">'
                      f'<span class="{bold} {indent}">{nf["nutrient"]}</span>'
                      f'<span class="nut-val">{vs} {nf["unit"]}</span></div>')

    st.markdown(f"""
    <div class="nut-label-box">
      <div class="nut-title">Nutrition Facts</div>
      <div class="nut-row thick">
        <span>Per serving / 100g</span><span></span>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("*No nutrition table found on this label.*")

# ── Ingredients: inline tags ──────────────────────────────────────────────────
st.markdown('<p class="section-label">ingredients</p>', unsafe_allow_html=True)

if detected:
    tags = ""
    for d in detected:
        cls_map = {"harmful": "ing-avoid", "caution": "ing-limit", "safe": "ing-safe"}
        css = cls_map.get(d["classification"], "ing-safe")
        tags += f'<span class="ing-tag {css}">{d["ingredient"].title()}</span> '
    st.markdown(tags, unsafe_allow_html=True)
else:
    st.markdown("*No ingredients matched our database.*")

# ── Concerns ──────────────────────────────────────────────────────────────────
flagged = [d for d in detected if d["classification"] in ("harmful", "caution")]
if flagged:
    st.markdown('<p class="section-label">watch out for these</p>', unsafe_allow_html=True)
    for d in flagged:
        col = RED if d["classification"] == "harmful" else AMBER
        label = "avoid" if d["classification"] == "harmful" else "limit"
        st.markdown(f"""
        <div class="concern-item" style="border-color:{col};background:{col}08">
          <b>{d['ingredient'].title()}</b>
          <span style="color:{col};font-size:0.78rem;margin-left:6px">{label} · {d['gemini_rating']:.1f}/5</span>
          <br><small>{d['description']}</small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("Nothing flagged — the ingredients look alright from what we could read.")

# ── Explanation ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">the full picture</p>', unsafe_allow_html=True)
explanation = generate_explanation(score, risk, detected, nutrition)
# Strip the markdown headers/emojis for a cleaner look
explanation = explanation.replace("### ", "**").replace("---", "")
explanation = explanation.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
explanation = explanation.replace("⛔ ", "").replace("> *", "*").replace("*\n", "*\n")
st.markdown(f'<div class="explanation">{explanation}</div>', unsafe_allow_html=True)

# ── What the OCR actually read (collapsed) ────────────────────────────────────
with st.expander("See what the OCR read"):
    if a.ocr_result.annotated_image:
        st.image(a.ocr_result.annotated_image, caption="Text regions detected by Surya",
                 use_container_width=True)
    st.text(raw_text)

    if a.zones:
        z = a.zones
        st.markdown("**Zones**")
        st.caption(f"Nutrition ({len(z.nutrition)} chars) · "
                   f"Ingredients ({len(z.ingredients)} chars) · "
                   f"Other ({len(z.other)} chars)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  This is an educational tool, not medical advice. It reads labels with OCR
  and checks ingredients against a curated database — it can miss things
  or misread text. For real dietary guidance, talk to a nutritionist.
</div>
""", unsafe_allow_html=True)
