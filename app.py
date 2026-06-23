import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import (
    classify_health,
    clean_text,
    compute_health_score,
    count_concerns,
    detect_harmful_ingredients,
    detect_ingredients,
    extract_text,
    generate_explanation,
    get_grade,
    load_ingredient_db,
    parse_nutritional_values,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="e-K Cha Label?",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / palette ───────────────────────────────────────────────────────────
PRIMARY = "#4361ee"
GOOD = "#2a9d8f"
WARN = "#f4a261"
BAD = "#e63946"
INK = "#1a1a2e"

RISK_COLOURS = {"High": BAD, "Moderate": WARN, "Low": "#8d99ae", "Safe": GOOD}
CLASS_BADGE = {
    "harmful": ("Harmful", BAD),
    "caution": ("Caution", WARN),
    "safe": ("Safe", GOOD),
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
    .hero {{
        background: linear-gradient(120deg, {INK} 0%, {PRIMARY} 100%);
        border-radius: 18px; padding: 1.8rem 2rem; color: #fff;
        box-shadow: 0 10px 30px rgba(67,97,238,.25); margin-bottom: 1.4rem;
    }}
    .hero h1 {{ font-size: 2.3rem; font-weight: 800; margin: 0; color:#fff; }}
    .hero p  {{ font-size: 1.02rem; opacity: .9; margin: .4rem 0 0; }}
    .chip {{
        display:inline-block; background:rgba(255,255,255,.16); color:#fff;
        border-radius:999px; padding:.25rem .8rem; font-size:.78rem;
        margin:.35rem .35rem 0 0; backdrop-filter: blur(4px);
    }}
    .metric-card {{
        background:#fff; border:1px solid #eef0f5; border-radius:16px;
        padding:1.1rem 1.2rem; box-shadow:0 4px 14px rgba(20,20,50,.05);
        height:100%;
    }}
    .metric-card .label {{ font-size:.8rem; color:#6b7280; text-transform:uppercase;
        letter-spacing:.05em; margin:0; }}
    .metric-card .value {{ font-size:2.1rem; font-weight:800; margin:.1rem 0 0; }}
    .ing-card {{
        background:#fff; border:1px solid #eef0f5; border-left:5px solid {PRIMARY};
        border-radius:12px; padding:.85rem 1.1rem; margin-bottom:.7rem;
        box-shadow:0 2px 8px rgba(20,20,50,.04);
    }}
    .badge {{ color:#fff; border-radius:999px; padding:2px 10px;
        font-size:.72rem; font-weight:600; }}
    .tok {{ display:inline-block; border-radius:8px; padding:3px 10px;
        font-size:.8rem; margin:3px; font-weight:500; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>🔬 e-K Cha Label?</h1>
      <p>AI-based FMCG label analyzer — Tesseract OCR · NLP · dataset-driven ingredient intelligence</p>
      <span class="chip">📷 Scan any product label</span>
      <span class="chip">🧪 Ingredient safety</span>
      <span class="chip">📊 Health score 0–100</span>
      <span class="chip">💬 Plain-language verdict</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    lang_choice = st.multiselect(
        "OCR language(s)", ["en", "ne", "hi"], default=["en"],
        help="Select the languages printed on the label (requires the matching "
             "Tesseract language packs: eng / nep / hin).",
    )
    preprocess = st.toggle("Enhance image before OCR", value=True,
                           help="Grayscale, denoise and binarise for sharper text.")
    show_raw = st.toggle("Show raw OCR text", value=False)

    st.markdown("---")
    st.header("📚 Knowledge base")
    try:
        _db = load_ingredient_db()
        n_total = len(_db)
        n_harm = int((_db["classification"] == "harmful").sum())
        n_caut = int((_db["classification"] == "caution").sum())
        n_safe = int((_db["classification"] == "safe").sum())
        st.caption(
            f"**{n_total}** ingredients loaded from dataset  \n"
            f"🔴 {n_harm} harmful · 🟠 {n_caut} caution · 🟢 {n_safe} safe"
        )
        with st.expander("Browse dataset"):
            st.dataframe(
                _db[["ingredient", "category", "classification", "risk_level"]],
                use_container_width=True, hide_index=True, height=260,
            )
    except Exception as e:
        st.error(f"Could not load ingredient dataset: {e}")

    st.markdown("---")
    st.caption("Edit `ingredients_dataset.csv` to add or change ingredients — no code changes needed.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, colour: str = INK) -> str:
    return (
        f'<div class="metric-card"><p class="label">{label}</p>'
        f'<p class="value" style="color:{colour}">{value}</p></div>'
    )


def gauge_figure(score: int, colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0, 40], "color": "#fdecea"},
                {"range": [40, 70], "color": "#fdf3e7"},
                {"range": [70, 100], "color": "#e8f5f2"},
            ],
            "threshold": {"line": {"color": INK, "width": 3},
                          "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("📥 Provide a label image")
tab_upload, tab_camera = st.tabs(["📂 Upload image", "📷 Use camera"])
image = None

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an FMCG product label",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear, well-lit, straight-on photos give the best OCR results.",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

with tab_camera:
    cam_img = st.camera_input("Point your camera at the product label")
    if cam_img:
        image = Image.open(cam_img).convert("RGB")

# ── Analysis ──────────────────────────────────────────────────────────────────
if image is None:
    st.info("👆 Upload an image or capture one with your camera to begin.")
    with st.expander("ℹ️ How it works", expanded=True):
        steps = [
            ("📷", "Image input", "Upload or capture a product label."),
            ("🔍", "Tesseract OCR", "Image is enhanced, then text is extracted."),
            ("🧪", "Ingredient match", "Tokens are checked against the ingredient dataset."),
            ("📊", "Health score", "Weighted score (0–100) plus a Nutri-grade."),
            ("💬", "Verdict", "A plain-language explanation of the risks."),
        ]
        cols = st.columns(len(steps))
        for col, (icon, title, desc) in zip(cols, steps):
            col.markdown(f"### {icon}\n**{title}**\n\n<small>{desc}</small>",
                         unsafe_allow_html=True)
    st.stop()

# Layout: image on the left, headline results on the right.
left, right = st.columns([1, 1.7], gap="large")

with left:
    st.image(image, caption="Input image", use_container_width=True)

with right:
    with st.spinner("🔍 Running Tesseract OCR …"):
        raw_text = extract_text(image, languages=lang_choice, preprocess=preprocess)

    if not raw_text.strip():
        st.warning("⚠️ No text could be extracted. Try a clearer, higher-resolution "
                   "image, or enable image enhancement in the sidebar.")
        st.stop()

    with st.spinner("🧠 Analysing ingredients & nutrition …"):
        cleaned = clean_text(raw_text)
        detected = detect_ingredients(raw_text)
        harmful = detect_harmful_ingredients(raw_text)
        nutrition = parse_nutritional_values(raw_text)
        h_cnt, c_cnt, s_cnt = count_concerns(detected)

    with st.spinner("🤖 Scoring …"):
        score = compute_health_score(nutrition, len(harmful))
        grade = get_grade(score)
        risk = classify_health(score)

    score_colour = GOOD if score >= 70 else WARN if score >= 40 else BAD

    m1, m2, m3 = st.columns(3)
    m1.markdown(metric_card("Health score", f"{score}", score_colour), unsafe_allow_html=True)
    m2.markdown(metric_card("Nutri-grade", grade, score_colour), unsafe_allow_html=True)
    m3.markdown(metric_card("Risk level", risk.split(" ")[-1], score_colour),
                unsafe_allow_html=True)

    st.plotly_chart(gauge_figure(score, score_colour), use_container_width=True,
                    config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Harmful", h_cnt)
    c2.metric("🟠 Caution", c_cnt)
    c3.metric("🟢 Safe", s_cnt)

if show_raw:
    with st.expander("📄 Raw OCR text"):
        st.text(raw_text)

st.markdown("---")

# ── Detailed tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🧪 Ingredients", "📈 Nutrition", "⚠️ Flagged additives", "💬 Verdict"]
)

# Tab 1 — detected ingredients (from dataset)
with tab1:
    st.subheader("Ingredients matched against the dataset")
    if detected:
        for d in detected:
            label, colour = CLASS_BADGE.get(d["classification"], ("Unknown", "#8d99ae"))
            risk_colour = RISK_COLOURS.get(d["risk_level"], "#8d99ae")
            st.markdown(
                f"""
                <div class="ing-card" style="border-left-color:{colour}">
                  <b style="font-size:1.02rem">{d['ingredient'].title()}</b>
                  &nbsp;<span class="badge" style="background:{colour}">{label}</span>
                  &nbsp;<span class="badge" style="background:{risk_colour}">{d['risk_level']} risk</span>
                  <br><small style="color:#6b7280"><b>{d['category']}</b> — {d['description']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No dataset ingredients were recognised in the extracted text.")

    other = [t for t in cleaned.split() if len(t) > 3][:40]
    if other:
        st.markdown("##### Other extracted keywords")
        chips = "".join(
            f'<span class="tok" style="background:#eef0f5;color:#374151">{t}</span>'
            for t in other
        )
        st.markdown(chips, unsafe_allow_html=True)

# Tab 2 — nutrition
with tab2:
    st.subheader("Parsed nutritional values (per 100 g/ml)")
    present = {k: v for k, v in nutrition.items() if v > 0}
    if present:
        df_nut = pd.DataFrame(
            [(k.title(), v, "kcal" if k == "calories" else "g") for k, v in nutrition.items()],
            columns=["Nutrient", "Value", "Unit"],
        )
        cc1, cc2 = st.columns([1, 1.4], gap="large")
        with cc1:
            st.dataframe(df_nut, use_container_width=True, hide_index=True)
        with cc2:
            keys = list(present.keys())
            vals = [present[k] for k in keys]
            bar_colours = [BAD if k in ("sugar", "fat", "sodium", "saturated")
                           else PRIMARY for k in keys]
            fig = go.Figure(go.Bar(
                x=vals, y=[k.title() for k in keys], orientation="h",
                marker_color=bar_colours,
                text=[f"{v:g}" for v in vals], textposition="outside",
            ))
            fig.update_layout(
                height=300, margin=dict(l=10, r=20, t=10, b=10),
                xaxis_title="Amount (g / kcal)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No nutritional values could be parsed. The table may be in an "
                "unsupported format or language, or the numbers were not read clearly.")

# Tab 3 — flagged additives
with tab3:
    st.subheader("Additives of concern")
    flagged = [d for d in detected if d["classification"] in ("harmful", "caution")]
    if flagged:
        for d in flagged:
            risk_colour = RISK_COLOURS.get(d["risk_level"], "#8d99ae")
            st.markdown(
                f"""
                <div class="ing-card" style="border-left-color:{risk_colour}">
                  <b style="font-size:1.02rem">{d['ingredient'].title()}</b>
                  &nbsp;<span class="badge" style="background:{risk_colour}">{d['risk_level']} risk</span>
                  <br><small style="color:#6b7280"><b>{d['category']}</b></small>
                  <br><small>{d['description']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No additives of concern were detected in the extracted text.")

# Tab 4 — verdict + breakdown
with tab4:
    st.subheader("AI-generated health verdict")
    explanation = generate_explanation(score, risk, detected, nutrition)
    st.markdown(explanation)

    st.markdown("##### Risk contribution by component")
    labels = ["Sugar", "Fat", "Additives", "Calories"]
    weights = [0.30, 0.25, 0.25, 0.20]
    raw_vals = [
        nutrition.get("sugar", 0),
        nutrition.get("fat", 0),
        len(harmful) * 5,
        nutrition.get("calories", 0) / 20,
    ]
    max_vals = [50, 30, 25, 10]
    component_scores = [
        min(v / m, 1.0) * w * 100 for v, m, w in zip(raw_vals, max_vals, weights)
    ]
    fig_donut = go.Figure(go.Pie(
        labels=labels,
        values=[max(s, 0.1) for s in component_scores],
        hole=0.55,
        marker_colors=[BAD, WARN, PRIMARY, GOOD],
        textinfo="label+percent",
    ))
    fig_donut.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Educational tool only. Ingredient classifications come from a "
    "curated dataset and may be incomplete. Always consult a qualified nutritionist or "
    "healthcare professional for dietary advice."
)
