# e-K Cha Label? — FMCG Label Analyzer

Scan a product label → OCR → match against the ~8.7k-ingredient **EatSafe**
database → health score (0–100), Nutri-grade, NOVA processing level and a
plain-language verdict.

## Files
- `app.py` — Streamlit UI
- `utils.py` — OCR, ingredient matching, scoring, verdict
- `eatsafe_master_database.csv` — ingredient knowledge base
- `requirements.txt` / `packages.txt` — dependencies

## OCR engines
Tesseract is built for clean scans and is unreliable on photographed labels.
This app uses a multi-engine pipeline (`engine="auto"` picks the best one
available):

| Engine      | Accuracy on photos | Memory | Good for |
|-------------|--------------------|--------|----------|
| **OCR.space** (cloud) | High | Tiny (runs server-side) | **Streamlit Cloud** |
| PaddleOCR   | Highest | Heavy | Local / Docker only |
| EasyOCR     | High | Heavy (torch) | Local / Docker only |
| Tesseract   | Low | Light | Fallback / clean scans |

## Deploying on Streamlit Community Cloud (recommended path)
The free tier (~1 GB RAM) cannot reliably run PaddleOCR/EasyOCR — they bloat
the build or run out of memory. Use the **OCR.space cloud engine** instead:

1. Get a free API key (no credit card) at https://ocr.space/ocrapi
2. In your app on Streamlit Cloud: **Settings → Secrets**, add:
   ```toml
   OCR_SPACE_API_KEY = "your_key_here"
   ```
3. Deploy. `requirements.txt` and `packages.txt` are already Cloud-safe.

`packages.txt` lists only `tesseract-ocr` + `tesseract-ocr-eng` (the fallback
engine). Do **not** add `libgl1` / `libglib2.0-0` — `opencv-python-headless`
needs no system GL libs, and pinning glib breaks the apt solver on Streamlit
Cloud's Debian image.

Without a key the app uses OCR.space's public demo key, which works but is
heavily rate-limited — fine for a quick test, not for a demo day.

> Note: OCR.space's photo engine targets Latin scripts (English). Most FMCG
> labels in Nepal include English text, so this is usually sufficient. For
> Devanagari (Nepali/Hindi), run locally with EasyOCR, or use Google Cloud
> Vision (premium cloud OCR, needs a GCP key).

## Run locally
```bash
pip install -r requirements.txt
# To use the heavier local engines, also:
#   pip install easyocr            (or)   pip install paddleocr paddlepaddle
# For the Tesseract fallback on Ubuntu:
#   sudo apt install tesseract-ocr
export OCR_SPACE_API_KEY=your_key   # optional but recommended
streamlit run app.py
```

## Update the knowledge base
Replace `eatsafe_master_database.csv` (same columns) — no code changes needed.
