# 🔬 e-K Cha Label?
### AI-Based FMCG Label Analyzer · OCR + NLP + Machine Learning

> **"e-K Cha Label?"** (Nepali: "What's in this label?") is a Streamlit web application that analyses FMCG product labels using Optical Character Recognition, Natural Language Processing, and Machine Learning to provide instant health risk assessments.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📂 Image Upload | Upload JPEG/PNG label images from your device |
| 📷 Camera Scan | Capture label directly using device camera |
| 🔍 OCR Extraction | EasyOCR extracts all text from the label image |
| 🧠 NLP Processing | spaCy cleans, lemmatises, and analyses ingredient text |
| ⚠️ Harmful Detection | Flags 13+ known harmful additives (MSG, trans fat, aspartame…) |
| 📊 Health Score | Weighted 0–100 score with Nutri-grade (A–E) |
| 💬 AI Explanation | Rule-based explanation of health risks in plain language |
| 🌐 Cloud Deployment | Deployed on Streamlit Community Cloud |

---

## 🏗️ System Architecture

```
┌─────────────────────┐
│  Streamlit UI        │  ← Upload / Camera Input
└────────┬────────────┘
         │
┌────────▼────────────┐
│  Image Preprocessing │  ← PIL / OpenCV (resize, RGB)
└────────┬────────────┘
         │
┌────────▼────────────┐
│  OCR Engine          │  ← EasyOCR (multi-language)
│  (EasyOCR)           │
└────────┬────────────┘
         │
┌────────▼────────────┐
│  NLP Processing      │  ← spaCy en_core_web_sm
│  (spaCy)             │     stop-word removal, lemmatisation
└────────┬────────────┘
         │
┌────────▼────────────┐
│  ML Classifier +     │  ← Weighted health-score formula
│  Health Score Engine │     + rule-based classification
└────────┬────────────┘
         │
┌────────▼────────────┐
│  Result Dashboard    │  ← Score, grade, ingredient cards,
│  (Streamlit)         │     charts, AI explanation
└─────────────────────┘
```

---

## 📁 Project Structure

```
e-k-cha-label/
├── app.py              ← Main Streamlit application
├── utils.py            ← OCR, NLP, scoring, explanation logic
├── requirements.txt    ← Python dependencies
├── setup.sh            ← spaCy model download (Streamlit Cloud)
├── packages.txt        ← System packages (if needed)
├── .streamlit/
│   └── config.toml     ← Theme and server settings
└── README.md           ← This file
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/e-k-cha-label.git
cd e-k-cha-label

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy English model
python -m spacy download en_core_web_sm

# 5. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub (public or private)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New App"**
4. Select:
   - **Repo:** `e-k-cha-label`
   - **Branch:** `main`
   - **Main file:** `app.py`
5. Click **Deploy**

> The `setup.sh` file automatically downloads the spaCy model on first deployment.

---

## 🧪 How It Works

### 1. OCR (EasyOCR)
EasyOCR reads the product label image pixel-by-pixel using a CRAFT text detector + CRNN recogniser. It supports 80+ languages and works without an internet connection.

### 2. NLP (spaCy)
The extracted text is piped through spaCy's `en_core_web_sm` model:
- Lowercasing
- Stop-word removal
- Lemmatisation (e.g. "sugars" → "sugar")

### 3. Harmful Ingredient Detection
A curated list of 13+ additives is matched against the cleaned text using substring search. Each additive has an associated risk level (Low / Moderate / High) and health notes.

### 4. Nutritional Parsing
Regular expressions extract numeric values for: calories, sugar, fat, sodium, protein, carbohydrates, dietary fibre, and saturated fat.

### 5. Health Scoring
Weighted penalty formula:

| Component | Weight | Threshold (per 100g) |
|---|---|---|
| Sugar | 30% | Low <5g, High >22.5g |
| Fat | 25% | Low <3g, High >17.5g |
| Additives | 25% | 5+ additives = max penalty |
| Calories | 20% | Low <40 kcal, High >400 kcal |

**Score = (1 − total\_penalty) × 100**

### 6. Nutri-Grade

| Grade | Score |
|---|---|
| A | 80–100 |
| B | 65–79 |
| C | 50–64 |
| D | 35–49 |
| E | 0–34 |

### 7. AI Explanation
A rule-based engine generates a plain-language Markdown report highlighting specific concerns, recommendations, and a final verdict.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `easyocr` | OCR text extraction |
| `opencv-python-headless` | Image preprocessing |
| `spacy` | NLP processing |
| `torch` | EasyOCR backend |
| `Pillow` | Image handling |
| `numpy` | Array operations |
| `pandas` | Tabular data display |
| `matplotlib` | Charts and visualisations |
| `scikit-learn` | ML utilities |

---

## ⚠️ Disclaimer

This application is developed for **academic and educational purposes only**. The health assessments generated are based on simple rule-based and ML models and should **not** be used as medical or dietary advice. Always consult a qualified nutritionist or healthcare professional.

---

## 👤 Author

**[Your Name]**  
Department of [Your Department]  
[Your Institution]  
Academic Year: 2024–25

---

## 📄 License

MIT License — free to use for academic purposes with attribution.
