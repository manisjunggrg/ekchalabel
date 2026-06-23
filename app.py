import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import (
    LabelAnalysis,
    analyze_label,
    classify_health,
    compute_score,
    count_concerns,
    gemini_key_configured,
    generate_explanation,
    get_grade,
    load_ingredient_db,
    nova_label,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="e-K Cha Label?", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ── Palette ──────────────────────────────────────────────────────────────────
PRIMARY, GOOD, WARN, BAD, INK = "#4361ee", "#2a9d8f", "#f4a261", "#e63946", "#1a1a2e"
RISK_COLOURS = {"High": BAD, "Moderate": WARN, "Low": "#8d99ae", "Safe": GOOD}
CLASS_BADGE = {"harmful": ("Avoid", BAD), "caution": ("Limit", WARN), "safe": ("Safe", GOOD)}

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
    .hero {{ background: linear-gradient(120deg,{INK} 0%,{PRIMARY} 100%);
        border-radius:18px; padding:1.8rem 2rem; color:#fff;
        box-shadow:0 10px 30px rgba(67,97,238,.25); margin-bottom:1.4rem; }}
    .hero h1 {{ font-size:2.3rem; font-weight:800; margin:0; color:#fff; }}
    .hero p {{ font-size:1.02rem; opacity:.9; margin:.4rem 0 0; }}
    .chip {{ display:inline-block; background:rgba(255,255,255,.16); color:#fff;
        border-radius:999px; padding:.25rem .8rem; font-size:.78rem; margin:.35rem .35rem 0 0; }}
    .metric-card {{ background:#fff; border:1px solid #eef0f5; border-radius:16px;
        padding:1.1rem 1.2rem; box-shadow:0 4px 14px rgba(20,20,50,.05); height:100%; }}
    .metric-card .label {{ font-size:.78rem; color:#6b7280; text-transform:uppercase;
        letter-spacing:.05em; margin:0; }}
    .metric-card .value {{ font-size:2.0rem; font-weight:800; margin:.1rem 0 0; }}
    .ing-card {{ background:#fff; border:1px solid #eef0f5; border-left:5px solid {PRIMARY};
        border-radius:12px; padding:.8rem 1.05rem; margin-bottom:.65rem;
        box-shadow:0 2px 8px rgba(20,20,50,.04); }}
    .badge {{ color:#fff; border-radius:999px; padding:2px 10px; font-size:.7rem; font-weight:600; }}
    .nut-row {{ display:flex; justify-content:space-between; align-items:center;
        padding:6px 12px; border-bottom:1px solid #f0f0f5; font-size:.92rem; }}
    .nut-row:nth-child(odd) {{ background:#f8f9fb; }}
    .nut-label {{ font-weight:600; color:{INK}; }}
    .nut-val {{ font-weight:700; color:{PRIMARY}; font-size:1.0rem; }}
    .nut-unit {{ color:#8d99ae; font-size:.82rem; margin-left:4px; }}
    </style>
    """, unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔬 e-K Cha Label?</h1>
      <p>AI-based FMCG label analyzer — Gemini Vision · 8.7k-ingredient EatSafe database</p>
      <span class="chip">📷 Scan any label</span>
      <span class="chip">🧪 Ingredient safety</span>
      <span class="chip">📊 Nutrition extraction</span>
      <span class="chip">💬 Health verdict</span>
    </div>
    """, unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    if gemini_key_configured():
        st.success("✅ Gemini API key configured")
    else:
        st.error(
            "🔑 **GOOGLE_API_KEY** not set.  \n"
            "Add it in **Settings → Secrets**:  \n"
            '```\nGOOGLE_API_KEY = "your_key"\n```  \n'
            "Get a free key → [Google AI Studio](https://aistudio.google.com/apikey)"
        )

    st.markdown("---")
    st.header("📚 EatSafe database")
    try:
        _db = load_ingredient_db()
        vc = _db["additive_risk"].value_counts()
        st.caption(
            f"**{len(_db):,}** ingredients  \n"
            f"🔴 {int(vc.get('avoid',0)):,} avoid · 🟠 {int(vc.get('limit',0)):,} limit · "
            f"🟢 {int(vc.get('safe',0)):,} safe"
        )
        with st.expander("Browse database"):
            st.dataframe(
                _db[["ingredient", "nova_group", "additive_risk", "gemini_rating"]],
                use_container_width=True, hide_index=True, height=260,
            )
    except Exception as e:
        st.error(f"Could not load database: {e}")
    st.markdown("---")
    st.caption("Engine: **Gemini 2.0 Flash** (vision)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label, value, colour=INK):
    return (f'<div class="metric-card"><p class="label">{label}</p>'
            f'<p class="value" style="color:{colour}">{value}</p></div>')


def gauge_figure(score, colour):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "/100", "font": {"size": 32}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour, "thickness": 0.3},
               "steps": [{"range": [0, 40], "color": "#fdecea"},
                         {"range": [40, 70], "color": "#fdf3e7"},
                         {"range": [70, 100], "color": "#e8f5f2"}],
               "threshold": {"line": {"color": INK, "width": 3}, "thickness": .75, "value": score}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def ingredient_card(d):
    label, colour = CLASS_BADGE.get(d["classification"], ("?", "#8d99ae"))
    art = ('&nbsp;<span class="badge" style="background:#6d597a">Artificial</span>'
           if d.get("artificial_flag") else "")
    return f"""
    <div class="ing-card" style="border-left-color:{colour}">
      <b style="font-size:1.0rem">{d['ingredient'].title()}</b>
      &nbsp;<span class="badge" style="background:{colour}">{label}</span>
      &nbsp;<span class="badge" style="background:{PRIMARY}">★ {d['gemini_rating']:.1f}/5</span>{art}
      <br><small style="color:#6b7280"><b>{d['category']}</b></small>
      <br><small>{d['description']}</small>
    </div>"""


def render_nutrition_table(nf: list[dict]):
    """Render nutrition facts as a styled label → value + unit table."""
    html = '<div style="border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;margin:8px 0">'
    html += (f'<div style="background:{PRIMARY};color:#fff;padding:10px 14px;'
             f'font-weight:700;font-size:1.0rem">Nutrition Facts (extracted by Gemini)</div>')
    for item in nf:
        nutrient = item["nutrient"]
        value = item["value"]
        unit = item["unit"]
        # Format value: int if whole, else 1 decimal
        val_str = f"{value:g}" if value == int(value) else f"{value:.1f}"
        html += (f'<div class="nut-row">'
                 f'<span class="nut-label">{nutrient}</span>'
                 f'<span><span class="nut-val">{val_str}</span>'
                 f'<span class="nut-unit">{unit}</span></span>'
                 f'</div>')
    html += '</div>'
    return html


# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📥 Provide a label image")
tab_upload, tab_camera = st.tabs(["📂 Upload image", "📷 Use camera"])
image = None
with tab_upload:
    up = st.file_uploader("Upload an FMCG product label",
                          type=["jpg", "jpeg", "png", "webp"],
                          help="Clear, well-lit photos read best.")
    if up:
        image = Image.open(up).convert("RGB")
with tab_camera:
    cam = st.camera_input("Point your camera at the product label")
    if cam:
        image = Image.open(cam).convert("RGB")

if image is None:
    st.info("👆 Upload an image or capture one with your camera to begin.")
    with st.expander("ℹ️ How it works", expanded=True):
        steps = [("📷", "Image", "Upload or capture a label."),
                 ("🤖", "Gemini Vision", "AI reads the label, extracts text, nutrition table & ingredients."),
                 ("🧪", "Database match", "Ingredients matched against 8.7k EatSafe entries."),
                 ("📊", "Score", "0–100 health score from ingredients + nutrition."),
                 ("💬", "Verdict", "Plain-language health summary.")]
        cols = st.columns(len(steps))
        for col, (i, t, d) in zip(cols, steps):
            col.markdown(f"### {i}\n**{t}**\n\n<small>{d}</small>", unsafe_allow_html=True)
    st.stop()

# ── Analysis ──────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.7], gap="large")
with left:
    st.image(image, caption="Input image", use_container_width=True)

with right:
    try:
        with st.spinner("🤖 Gemini is reading the label …"):
            analysis: LabelAnalysis = analyze_label(image)
    except RuntimeError as e:
        st.error(str(e))
        if "rate limit" in str(e).lower():
            st.info("The Gemini free tier allows **15 requests per minute**. "
                    "Wait a moment and click below to retry.")
            if st.button("🔄 Retry"):
                st.rerun()
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()

    score = analysis.score
    grade = analysis.grade
    risk = analysis.risk
    detected = analysis.detected
    nutrition = analysis.nutrition
    colour = GOOD if score >= 70 else WARN if score >= 40 else BAD
    avg_rating = float(np.mean([d["gemini_rating"] for d in detected])) if detected else 0.0

    if analysis.product_name:
        st.markdown(f"**Product:** {analysis.product_name}")

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Health score", f"{score}", colour), unsafe_allow_html=True)
    m2.markdown(metric_card("Nutri-grade", grade, colour), unsafe_allow_html=True)
    m3.markdown(metric_card("Risk", risk.split(" ")[-1], colour), unsafe_allow_html=True)
    m4.markdown(metric_card("Avg rating", f"{avg_rating:.1f}/5", colour), unsafe_allow_html=True)

    st.plotly_chart(gauge_figure(score, colour), use_container_width=True,
                    config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Avoid", analysis.h_cnt)
    c2.metric("🟠 Limit", analysis.c_cnt)
    c3.metric("🟢 Safe", analysis.s_cnt)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_nut, tab_flag, tab_all, tab_verdict, tab_raw = st.tabs(
    ["📈 Nutrition", "⚠️ Flagged", "🧪 All ingredients", "💬 Verdict", "📄 Raw extraction"])

# Tab: Nutrition — THE KEY TABLE
with tab_nut:
    st.subheader("Nutrition Facts — extracted from label")
    if analysis.nutrition_facts:
        ncol1, ncol2 = st.columns([1.1, 1.3], gap="large")
        with ncol1:
            st.markdown(render_nutrition_table(analysis.nutrition_facts), unsafe_allow_html=True)

        with ncol2:
            # Bar chart of key nutrients
            plot_items = [nf for nf in analysis.nutrition_facts if nf["value"] > 0]
            if plot_items:
                names = [f"{nf['nutrient']} ({nf['unit']})" for nf in plot_items]
                vals = [nf["value"] for nf in plot_items]
                # Colour by concern level
                concern_keys = {"sugar", "sugars", "total fat", "fat", "sodium", "salt",
                                "saturated fat", "trans fat", "cholesterol"}
                bar_colours = [
                    BAD if nf["nutrient"].lower() in concern_keys else PRIMARY
                    for nf in plot_items
                ]
                fig = go.Figure(go.Bar(
                    x=vals, y=names, orientation="h",
                    marker_color=bar_colours,
                    text=[f"{v:g}" for v in vals], textposition="outside",
                ))
                fig.update_layout(
                    height=max(280, len(plot_items) * 32),
                    margin=dict(l=10, r=30, t=10, b=10),
                    xaxis_title="Amount",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Scoring breakdown
        if nutrition:
            st.markdown("##### Values used for scoring (normalised)")
            score_df = pd.DataFrame([
                {"Nutrient": k.title(), "Value": f"{v:g}", "Unit": "kcal" if k == "calories" else "g"}
                for k, v in nutrition.items() if v > 0
            ])
            if not score_df.empty:
                st.dataframe(score_df, use_container_width=True, hide_index=True)
    else:
        st.info("No nutrition table found on this label.")

# Tab: Flagged
with tab_flag:
    st.subheader("Ingredients to avoid or limit")
    flagged = [d for d in detected if d["classification"] in ("harmful", "caution")]
    if flagged:
        for d in flagged:
            st.markdown(ingredient_card(d), unsafe_allow_html=True)
    else:
        st.success("✅ No 'avoid' or 'limit' ingredients detected.")

# Tab: All ingredients
with tab_all:
    st.subheader(f"Gemini extracted {len(analysis.extracted_ingredients)} ingredients")

    # Show what Gemini read vs what matched the database
    ext_col, match_col = st.columns(2, gap="large")
    with ext_col:
        st.markdown("**📝 Extracted from label**")
        for ing in analysis.extracted_ingredients:
            matched_names = {d["ingredient"] for d in detected}
            is_matched = ing.strip().lower() in matched_names or any(
                ing.strip().lower() in d["ingredient"] or d["ingredient"] in ing.strip().lower()
                for d in detected
            )
            icon = "🟢" if is_matched else "⚪"
            st.markdown(f"{icon} {ing}")
        if not analysis.extracted_ingredients:
            st.info("No ingredients extracted.")

    with match_col:
        st.markdown(f"**🗄️ Matched in database ({len(detected)})**")
        safe = [d for d in detected if d["classification"] == "safe"]
        flagged_all = [d for d in detected if d["classification"] != "safe"]
        if flagged_all:
            for d in flagged_all:
                st.markdown(ingredient_card(d), unsafe_allow_html=True)
        if safe:
            st.markdown("**Safe:**")
            chips = "".join(
                f'<span style="display:inline-block;background:#e8f5f2;color:#1d6e63;'
                f'border-radius:8px;padding:3px 10px;font-size:.82rem;margin:3px;font-weight:500">'
                f'{d["ingredient"].title()} · ★{d["gemini_rating"]:.1f}</span>' for d in safe)
            st.markdown(chips, unsafe_allow_html=True)
        if not detected:
            st.info("No database matches found.")

# Tab: Verdict
with tab_verdict:
    st.subheader("AI-generated health verdict")
    st.markdown(generate_explanation(score, risk, detected, nutrition))

    if detected:
        st.markdown("##### Ingredient ratings (lower = worse)")
        worst = sorted(detected, key=lambda d: d["gemini_rating"])[:12]
        cl = {"harmful": BAD, "caution": WARN, "safe": GOOD}
        fig = go.Figure(go.Bar(
            x=[d["gemini_rating"] for d in worst][::-1],
            y=[d["ingredient"].title() for d in worst][::-1],
            orientation="h",
            marker_color=[cl.get(d["classification"], PRIMARY) for d in worst][::-1],
            text=[f'{d["gemini_rating"]:.1f}' for d in worst][::-1], textposition="outside"))
        fig.update_layout(height=max(280, len(worst) * 32),
                          margin=dict(l=10, r=20, t=10, b=10),
                          xaxis_title="Rating (0–5)", xaxis_range=[0, 5.4],
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Tab: Raw extraction
with tab_raw:
    st.subheader("Raw Gemini extraction")
    st.caption("Everything Gemini read from the label, shown for debugging.")
    if analysis.raw_text:
        st.text_area("Full text", analysis.raw_text, height=250, disabled=True)
    else:
        st.info("No raw text returned.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** Educational tool only. Consult a qualified nutritionist for dietary advice.")
