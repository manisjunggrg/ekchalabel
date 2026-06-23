import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

from utils import (
    extract_text,
    clean_text,
    detect_harmful_ingredients,
    compute_health_score,
    classify_health,
    generate_explanation,
    parse_nutritional_values,
    get_grade,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="e-K Cha Label?",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title  { font-size:2.6rem; font-weight:800; color:#1a1a2e; }
    .sub-title   { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
    .card        { background:#f8f9fa; border-radius:12px; padding:1.2rem 1.5rem;
                   margin-bottom:1rem; border-left:5px solid #4361ee; }
    .risk-high   { border-left-color:#e63946; }
    .risk-mid    { border-left-color:#f4a261; }
    .risk-low    { border-left-color:#2a9d8f; }
    .badge-red   { background:#e63946; color:#fff; border-radius:6px;
                   padding:2px 8px; font-size:.8rem; }
    .badge-grn   { background:#2a9d8f; color:#fff; border-radius:6px;
                   padding:2px 8px; font-size:.8rem; }
    .badge-org   { background:#f4a261; color:#fff; border-radius:6px;
                   padding:2px 8px; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 e-K Cha Label?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-Based FMCG Label Analyzer · OCR + NLP + Machine Learning</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "**e-K Cha Label?** scans FMCG product labels and tells you:\n"
        "- What ingredients are present\n"
        "- Which ones are potentially harmful\n"
        "- An overall health score (0–100)\n"
        "- An AI-generated explanation"
    )
    st.markdown("---")
    st.header("⚙️ Settings")
    lang_choice = st.multiselect(
        "OCR Language(s)", ["en", "ne", "hi"], default=["en"],
        help="Select languages present on the label"
    )
    show_raw = st.checkbox("Show raw OCR text", value=False)
    st.markdown("---")
    st.caption("Built for academic submission · Streamlit Cloud")

# ── Input section ─────────────────────────────────────────────────────────────
st.subheader("📥 Input")
input_mode = st.radio(
    "Choose input method:",
    ["📂 Upload Image", "📷 Capture from Camera"],
    horizontal=True,
)

image = None

if input_mode == "📂 Upload Image":
    uploaded = st.file_uploader(
        "Upload a FMCG product label image",
        type=["jpg", "jpeg", "png"],
        help="Clear, well-lit photos give the best OCR results.",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

else:
    cam_img = st.camera_input("Point your camera at the product label")
    if cam_img:
        image = Image.open(cam_img).convert("RGB")

# ── Analysis ──────────────────────────────────────────────────────────────────
if image is not None:
    col_img, col_res = st.columns([1, 2], gap="large")

    with col_img:
        st.image(image, caption="Input Image", use_column_width=True)

    with col_res:
        with st.spinner("🔍 Running OCR …"):
            raw_text = extract_text(image, languages=lang_choice)

        if not raw_text.strip():
            st.warning("⚠️ No text could be extracted. Try a clearer or higher-resolution image.")
            st.stop()

        if show_raw:
            st.subheader("📄 Raw OCR Text")
            st.text_area("", raw_text, height=140)

        with st.spinner("🧠 Processing with NLP …"):
            cleaned = clean_text(raw_text)
            harmful = detect_harmful_ingredients(cleaned)
            nutrition = parse_nutritional_values(raw_text)

        with st.spinner("🤖 Running ML Classifier …"):
            score = compute_health_score(nutrition, len(harmful))
            grade = get_grade(score)
            risk  = classify_health(score)

        # ── Score gauge ──────────────────────────────────────────────────────
        st.subheader("📊 Health Score")
        score_col, grade_col, risk_col = st.columns(3)
        score_col.metric("Score", f"{score}/100")
        grade_col.metric("Nutri-Grade", grade)
        risk_col.metric("Risk Level", risk)

        # Colour bar
        colour = "#2a9d8f" if score >= 70 else "#f4a261" if score >= 40 else "#e63946"
        st.markdown(
            f"""
            <div style="background:#e9ecef;border-radius:8px;height:18px;width:100%">
              <div style="background:{colour};width:{score}%;height:18px;
                          border-radius:8px;transition:width .5s"></div>
            </div>
            <p style="font-size:.8rem;color:#777;margin-top:4px">{score}% health score</p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Detailed results ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧪 Ingredients", "📈 Nutrition", "⚠️ Harmful Additives", "🧠 AI Explanation"]
    )

    # Tab 1 – Ingredients
    with tab1:
        st.subheader("Detected Keywords")
        ingredient_keywords = [
            tok for tok in cleaned.split()
            if len(tok) > 3
        ][:40]
        if ingredient_keywords:
            cols = st.columns(4)
            for i, kw in enumerate(ingredient_keywords):
                badge = "badge-red" if kw in harmful else "badge-grn"
                cols[i % 4].markdown(
                    f'<span class="{badge}">{kw}</span>', unsafe_allow_html=True
                )
        else:
            st.info("No clear ingredient tokens found.")

    # Tab 2 – Nutrition
    with tab2:
        st.subheader("Parsed Nutritional Values")
        if any(v > 0 for v in nutrition.values()):
            df_nut = pd.DataFrame(
                list(nutrition.items()), columns=["Nutrient", "Value (per 100g/ml)"]
            )
            st.dataframe(df_nut, use_container_width=True, hide_index=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(6, 3))
            keys = [k for k, v in nutrition.items() if v > 0]
            vals = [nutrition[k] for k in keys]
            bars = ax.barh(keys, vals, color=["#4361ee", "#e63946", "#f4a261", "#2a9d8f"][:len(keys)])
            ax.set_xlabel("Amount (g / kcal)")
            ax.set_title("Nutritional Breakdown")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info(
                "Nutritional values could not be parsed automatically from this label. "
                "Values may be in an unsupported format or language."
            )

    # Tab 3 – Harmful additives
    with tab3:
        st.subheader("Harmful / Concerning Ingredients")
        HARMFUL_INFO = {
            "aspartame":        ("Artificial sweetener", "Moderate", "May cause headaches in sensitive individuals"),
            "msg":              ("Flavour enhancer", "Moderate", "Can trigger reactions in sensitive people"),
            "sodium benzoate":  ("Preservative", "High",     "Linked to hyperactivity; potential carcinogen"),
            "trans fat":        ("Fat type",      "High",     "Raises LDL cholesterol; major cardiovascular risk"),
            "tartrazine":       ("Food colouring","Moderate", "Linked to allergic reactions and hyperactivity"),
            "sodium nitrite":   ("Preservative",  "High",     "Associated with colorectal cancer risk"),
            "high fructose corn syrup": ("Sweetener","High",  "Linked to obesity and metabolic syndrome"),
            "bha":              ("Antioxidant",   "Moderate", "Possible carcinogen (IARC Group 2B)"),
            "bht":              ("Antioxidant",   "Moderate", "Possible endocrine disruptor"),
            "potassium bromate":("Flour improver","High",     "Banned in many countries; possible carcinogen"),
            "saccharin":        ("Sweetener",     "Low",      "Once linked to cancer – now generally regarded as safe"),
            "acesulfame":       ("Sweetener",     "Low",      "Limited long-term safety data"),
            "carrageenan":      ("Thickener",     "Moderate", "May cause gut inflammation"),
        }

        if harmful:
            for h in harmful:
                info = HARMFUL_INFO.get(h, ("Unknown category", "Unknown", "No details available"))
                risk_class = (
                    "risk-high" if info[1] == "High"
                    else "risk-mid" if info[1] == "Moderate"
                    else "risk-low"
                )
                st.markdown(
                    f"""
                    <div class="card {risk_class}">
                      <b>{h.title()}</b> &nbsp;
                      <span class="badge-{'red' if info[1]=='High' else 'org' if info[1]=='Moderate' else 'grn'}">
                        {info[1]} Risk
                      </span><br/>
                      <small><b>Category:</b> {info[0]}</small><br/>
                      <small>{info[2]}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No commonly known harmful additives detected in the extracted text.")

    # Tab 4 – AI Explanation
    with tab4:
        st.subheader("AI-Generated Health Explanation")
        explanation = generate_explanation(score, risk, harmful, nutrition)
        st.markdown(explanation)

        # Pie chart of score components
        st.markdown("#### Score Breakdown")
        labels  = ["Sugar", "Fat", "Additives", "Calories"]
        weights = [0.30,    0.25,  0.25,         0.20]
        raw_vals = [
            nutrition.get("sugar", 0),
            nutrition.get("fat",   0),
            len(harmful) * 5,
            nutrition.get("calories", 0) / 20,
        ]
        # Normalise to 0-1 per component
        max_vals = [50, 30, 25, 10]
        component_scores = [
            min(v / m, 1.0) * w * 100
            for v, m, w in zip(raw_vals, max_vals, weights)
        ]

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        colours = ["#e63946", "#f4a261", "#4361ee", "#2a9d8f"]
        wedges, texts, autotexts = ax2.pie(
            [max(s, 0.1) for s in component_scores],
            labels=labels,
            autopct="%1.0f%%",
            colors=colours,
            startangle=140,
        )
        ax2.set_title("Risk Contribution by Component")
        st.pyplot(fig2)

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Always consult a qualified nutritionist or healthcare professional for dietary advice."
    )

else:
    # Landing state
    st.info("👆 Upload an image or use your camera to get started.")
    with st.expander("ℹ️ How it works"):
        steps = [
            ("📷", "Image Input",      "Upload or capture a FMCG product label"),
            ("🔍", "OCR Engine",       "EasyOCR extracts all text from the label"),
            ("🧠", "NLP Processing",   "spaCy cleans and analyses the text"),
            ("🤖", "ML Classifier",    "Rule-based + ML model rates the product"),
            ("📊", "Health Score",     "Weighted score (0–100) with Nutri-grade"),
            ("💬", "AI Explanation",   "Plain-language summary of health risks"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"**{icon} {title}** — {desc}")
