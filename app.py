import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import (
    LabelAnalysis, analyze_label, classify_health, compute_score,
    count_concerns, detect_ingredients, draw_ocr_overlay,
    generate_explanation, get_grade, load_ingredient_db, load_surya_models,
    nova_label, parse_nutritional_values, _is_nutrition_line,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="e-K Cha Label?", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

PRIMARY, GOOD, WARN, BAD, INK = "#4361ee", "#2a9d8f", "#f4a261", "#e63946", "#1a1a2e"
CLASS_BADGE = {"harmful": ("Avoid", BAD), "caution": ("Limit", WARN), "safe": ("Safe", GOOD)}

st.markdown(f"""
<style>
.block-container {{ padding-top:1.4rem; max-width:1250px }}
.hero {{ background:linear-gradient(120deg,{INK} 0%,{PRIMARY} 100%);
    border-radius:18px; padding:1.8rem 2rem; color:#fff;
    box-shadow:0 10px 30px rgba(67,97,238,.25); margin-bottom:1.4rem }}
.hero h1 {{ font-size:2.3rem; font-weight:800; margin:0; color:#fff }}
.hero p {{ font-size:1.02rem; opacity:.9; margin:.4rem 0 0 }}
.chip {{ display:inline-block; background:rgba(255,255,255,.16); color:#fff;
    border-radius:999px; padding:.25rem .8rem; font-size:.78rem; margin:.35rem .35rem 0 0 }}
.mc {{ background:#fff; border:1px solid #eef0f5; border-radius:16px;
    padding:1.1rem 1.2rem; box-shadow:0 4px 14px rgba(20,20,50,.05) }}
.mc .l {{ font-size:.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; margin:0 }}
.mc .v {{ font-size:2.0rem; font-weight:800; margin:.1rem 0 0 }}
.ic {{ background:#fff; border:1px solid #eef0f5; border-left:5px solid {PRIMARY};
    border-radius:12px; padding:.8rem 1.05rem; margin-bottom:.65rem;
    box-shadow:0 2px 8px rgba(20,20,50,.04) }}
.badge {{ color:#fff; border-radius:999px; padding:2px 10px; font-size:.7rem; font-weight:600 }}
.nr {{ display:flex; justify-content:space-between; align-items:center;
    padding:6px 12px; border-bottom:1px solid #f0f0f5; font-size:.92rem }}
.nr:nth-child(odd) {{ background:#f8f9fb }}
.nl {{ font-weight:600; color:{INK} }}
.nv {{ font-weight:700; color:{PRIMARY}; font-size:1rem }}
.nu {{ color:#8d99ae; font-size:.82rem; margin-left:4px }}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔬 e-K Cha Label?</h1>
  <p>AI-based FMCG label analyzer — Surya OCR · 8.7k-ingredient EatSafe database</p>
  <span class="chip">📷 Scan any label</span>
  <span class="chip">🧪 Ingredient safety</span>
  <span class="chip">📊 Nutrition extraction</span>
  <span class="chip">💬 Health verdict</span>
</div>""", unsafe_allow_html=True)

# ── Model loading (cached, run once) ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading Surya OCR models (first run only) …")
def _load_models():
    return load_surya_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    lang_choice = st.multiselect("Label language(s)", ["en", "ne", "hi"], default=["en"])
    preprocess = st.toggle("Enhance image", value=True)
    show_raw = st.toggle("Show raw OCR text", value=False)
    st.markdown("---")
    st.header("📚 EatSafe database")
    try:
        _db = load_ingredient_db()
        vc = _db["additive_risk"].value_counts()
        st.caption(f"**{len(_db):,}** ingredients  \n"
                   f"🔴 {int(vc.get('avoid',0)):,} avoid · "
                   f"🟠 {int(vc.get('limit',0)):,} limit · "
                   f"🟢 {int(vc.get('safe',0)):,} safe")
        with st.expander("Browse"):
            st.dataframe(_db[["ingredient","nova_group","additive_risk","gemini_rating"]],
                         use_container_width=True, hide_index=True, height=260)
    except Exception as e:
        st.error(str(e))
    st.markdown("---")
    st.caption("Engine: **Surya OCR v0.16** (local, deep learning)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def mc(label, value, colour=INK):
    return (f'<div class="mc"><p class="l">{label}</p>'
            f'<p class="v" style="color:{colour}">{value}</p></div>')

def gauge(score, colour):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix":"/100","font":{"size":32}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":colour,"thickness":.3},
               "steps":[{"range":[0,40],"color":"#fdecea"},
                        {"range":[40,70],"color":"#fdf3e7"},
                        {"range":[70,100],"color":"#e8f5f2"}],
               "threshold":{"line":{"color":INK,"width":3},"thickness":.75,"value":score}}))
    fig.update_layout(height=230,margin=dict(l=20,r=20,t=8,b=8),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def ing_card(d):
    lbl, col = CLASS_BADGE.get(d["classification"], ("?","#8d99ae"))
    art = ('&nbsp;<span class="badge" style="background:#6d597a">Artificial</span>'
           if d.get("artificial_flag") else "")
    return (f'<div class="ic" style="border-left-color:{col}">'
            f'<b>{d["ingredient"].title()}</b>'
            f'&nbsp;<span class="badge" style="background:{col}">{lbl}</span>'
            f'&nbsp;<span class="badge" style="background:{PRIMARY}">★ {d["gemini_rating"]:.1f}/5</span>{art}'
            f'<br><small style="color:#6b7280"><b>{d["category"]}</b></small>'
            f'<br><small>{d["description"]}</small></div>')

def nut_table(items):
    h = (f'<div style="border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;margin:8px 0">'
         f'<div style="background:{PRIMARY};color:#fff;padding:10px 14px;font-weight:700">'
         f'Nutrition Facts (extracted by Surya OCR)</div>')
    for nf in items:
        vs = f"{nf['value']:g}" if nf['value'] == int(nf['value']) else f"{nf['value']:.1f}"
        h += (f'<div class="nr"><span class="nl">{nf["nutrient"]}</span>'
              f'<span><span class="nv">{vs}</span><span class="nu">{nf["unit"]}</span></span></div>')
    return h + '</div>'

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📥 Provide a label image")
tab_up, tab_cam = st.tabs(["📂 Upload image", "📷 Use camera"])
image = None
with tab_up:
    up = st.file_uploader("Upload FMCG label", type=["jpg","jpeg","png","webp"])
    if up: image = Image.open(up).convert("RGB")
with tab_cam:
    cam = st.camera_input("Point camera at label")
    if cam: image = Image.open(cam).convert("RGB")

if image is None:
    st.info("👆 Upload or capture a label to begin.")
    st.stop()

# ── Load models on first run ─────────────────────────────────────────────────
_load_models()

# ── Analysis ──────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.7], gap="large")
with left:
    st.image(image, caption="Input image", use_container_width=True)

with right:
    try:
        with st.spinner("🔍 Surya OCR + analysis …"):
            a: LabelAnalysis = analyze_label(image, lang_choice, preprocess)
    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("🔄 Retry"): st.rerun()
        st.stop()

    raw_text = a.ocr_result.text
    if not raw_text.strip():
        st.warning("⚠️ No text extracted. Try a sharper image.")
        st.stop()

    score, grade, risk = a.score, a.grade, a.risk
    detected, nutrition = a.detected, a.nutrition
    colour = GOOD if score >= 70 else WARN if score >= 40 else BAD
    avg_r = float(np.mean([d["gemini_rating"] for d in detected])) if detected else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(mc("Health score", str(score), colour), unsafe_allow_html=True)
    m2.markdown(mc("Nutri-grade", grade, colour), unsafe_allow_html=True)
    m3.markdown(mc("Risk", risk.split(" ")[-1], colour), unsafe_allow_html=True)
    m4.markdown(mc("Avg rating", f"{avg_r:.1f}/5", colour), unsafe_allow_html=True)

    st.plotly_chart(gauge(score, colour), use_container_width=True, config={"displayModeBar":False})
    c1,c2,c3 = st.columns(3)
    c1.metric("🔴 Avoid", a.h_cnt); c2.metric("🟠 Limit", a.c_cnt); c3.metric("🟢 Safe", a.s_cnt)

if show_raw:
    with st.expander("📄 Raw OCR text"): st.text(raw_text)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_nut, t_flag, t_all, t_ocr, t_verdict = st.tabs(
    ["📈 Nutrition", "⚠️ Flagged", "🧪 All ingredients", "🔍 OCR Debug", "💬 Verdict"])

with t_nut:
    st.subheader("Nutrition Facts — extracted from label")
    if a.nutrition_display:
        nc1, nc2 = st.columns([1.1, 1.3], gap="large")
        with nc1:
            st.markdown(nut_table(a.nutrition_display), unsafe_allow_html=True)
        with nc2:
            items = [n for n in a.nutrition_display if n["value"] > 0]
            if items:
                concern = {"sugar","fat","sodium","saturated","trans fat","cholesterol"}
                fig = go.Figure(go.Bar(
                    x=[n["value"] for n in items],
                    y=[f"{n['nutrient']} ({n['unit']})" for n in items],
                    orientation="h",
                    marker_color=[BAD if n["nutrient"].lower() in concern else PRIMARY for n in items],
                    text=[f"{n['value']:g}" for n in items], textposition="outside"))
                fig.update_layout(height=max(260, len(items)*32),
                    margin=dict(l=10,r=30,t=10,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("No nutrition values parsed from this label.")

with t_flag:
    st.subheader("Ingredients to avoid or limit")
    flagged = [d for d in detected if d["classification"] in ("harmful","caution")]
    if flagged:
        for d in flagged: st.markdown(ing_card(d), unsafe_allow_html=True)
    else:
        st.success("✅ No flagged ingredients detected.")

with t_all:
    st.subheader(f"All matched ingredients ({len(detected)})")
    for d in detected: st.markdown(ing_card(d), unsafe_allow_html=True)
    if not detected: st.info("No database matches.")

with t_ocr:
    st.subheader("OCR text detection — what Surya reads")
    st.caption("🟩 Green = nutrition lines · 🟦 Blue = other text")
    ol, or_ = st.columns([1.2, 1], gap="large")
    with ol:
        if a.ocr_result.annotated_image:
            st.image(a.ocr_result.annotated_image, caption="Annotated", use_container_width=True)
        else:
            st.image(image, caption="Original", use_container_width=True)
    with or_:
        for ln in a.ocr_result.lines:
            if not ln.text.strip(): continue
            is_nut = _is_nutrition_line(ln.text)
            bg = "#ecfdf5" if is_nut else "#eff6ff"
            border = "#22c55e" if is_nut else "#3b82f6"
            conf = f" · {ln.confidence:.0%}" if ln.confidence else ""
            st.markdown(
                f'<div style="background:{bg};border-left:4px solid {border};'
                f'padding:4px 10px;margin:3px 0;border-radius:6px;font-size:.88rem">'
                f'{ln.text}<span style="color:#9ca3af;font-size:.72rem">{conf}</span></div>',
                unsafe_allow_html=True)
    # Zone debug
    if a.zones:
        z = a.zones
        st.markdown("---")
        st.markdown("**Zone splitting**")
        z1,z2,z3 = st.columns(3)
        z1.text_area("🟩 Nutrition zone", z.nutrition or "(empty)", height=140, disabled=True)
        z2.text_area("🟦 Ingredient zone", z.ingredients or "(empty)", height=140, disabled=True)
        z3.text_area("⬜ Other", z.other or "(empty)", height=140, disabled=True)

with t_verdict:
    st.subheader("AI-generated health verdict")
    st.markdown(generate_explanation(score, risk, detected, nutrition))
    if detected:
        worst = sorted(detected, key=lambda d: d["gemini_rating"])[:12]
        cl = {"harmful":BAD,"caution":WARN,"safe":GOOD}
        fig = go.Figure(go.Bar(
            x=[d["gemini_rating"] for d in worst][::-1],
            y=[d["ingredient"].title() for d in worst][::-1],
            orientation="h",
            marker_color=[cl.get(d["classification"],PRIMARY) for d in worst][::-1],
            text=[f'{d["gemini_rating"]:.1f}' for d in worst][::-1], textposition="outside"))
        fig.update_layout(height=max(280,len(worst)*32),
            margin=dict(l=10,r=20,t=10,b=10),
            xaxis_title="Rating (0–5)", xaxis_range=[0,5.4],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

st.markdown("---")
st.caption("⚠️ Educational tool only. Consult a qualified nutritionist for dietary advice.")
