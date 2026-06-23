# e-K Cha Label? — FMCG Label Analyzer

Scan a product label → OCR → match against the ~8.7k-ingredient **EatSafe**
database → health score (0–100), Nutri-grade, NOVA processing level and a
plain-language verdict.

## Files
- `app.py` — Streamlit UI
- `utils.py` — OCR (dual-engine + overlay), ingredient matching, scoring, verdict
- `eatsafe_master_database.csv` — ingredient knowledge base (8,752 items)
- `requirements.txt` / `packages.txt` — dependencies

## OCR pipeline

### The problem with Tesseract
Tesseract is built for clean scans. It is unreliable on photographed labels
(curved text, shadows, uneven lighting). That is why earlier versions
struggled to extract text.

### Solution: OCR.space dual-engine with bounding boxes
The app calls the **OCR.space** cloud API (free tier, no credit card) which
runs deep-learning OCR on their server — accurate on real photos and uses
almost no app memory:

| Engine | Strength |
|--------|----------|
| **OCR.space Engine 2** | Best on photographed / curved text |
| **OCR.space Engine 1** | Best on structured tables (nutrition facts) |

Both engines run; their results are **merged and deduplicated**. The API also
returns **word-level bounding boxes**, which the app draws on the image:

- 🟩 **Green** boxes — nutrition-related lines (Energy, Fat, Sugar …)
- 🟦 **Blue** boxes — other text (ingredients, brand name, etc.)

The **🔍 OCR Debug** tab shows the annotated image beside a line-by-line
readout so you can immediately see what was read, what was missed, and which
lines are nutrition vs ingredients.

If OCR.space is unavailable the app falls back to Tesseract (installed via
`packages.txt`). PaddleOCR / EasyOCR can be enabled for local runs but are
too heavy for Streamlit Cloud's 1 GB free tier.

## Deploying on Streamlit Community Cloud

1. Get a free API key (no credit card) at https://ocr.space/ocrapi
2. In your app on Streamlit Cloud → **Settings → Secrets**, add:
   ```toml
   OCR_SPACE_API_KEY = "your_key_here"
   ```
3. Deploy. `requirements.txt` and `packages.txt` are already Cloud-safe.

`packages.txt` lists only `tesseract-ocr` + `tesseract-ocr-eng` (the fallback
engine). Do **not** add `libgl1` / `libglib2.0-0` — `opencv-python-headless`
needs no system GL libs, and pinning glib breaks the apt solver on Streamlit
Cloud's Debian image.

Without a key the app uses OCR.space's public demo key — works but is
heavily rate-limited (fine for a quick test, not for a demo day).

## Run locally
```bash
pip install -r requirements.txt
export OCR_SPACE_API_KEY=your_key   # recommended
streamlit run app.py
```

## Update the knowledge base
Replace `eatsafe_master_database.csv` (same columns) — no code changes needed.
