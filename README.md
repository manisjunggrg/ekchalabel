# e-K Cha Label? — FMCG Label Analyzer

Scan a product label → **Gemini Vision** reads it → ingredients matched
against 8.7k EatSafe database → health score + verdict.

## Why Gemini Vision instead of OCR?

Traditional OCR (Tesseract, EasyOCR, PaddleOCR, OCR.space) **cannot reliably
read photographed product labels** — curved text, glare, small fonts, and
complex table layouts break them. Layering regex on top of broken OCR
compounds the problem.

**Gemini Flash Vision** solves both problems at once:
- Reads text from photos accurately (it's a vision-language model, not OCR)
- Extracts **structured data directly** — nutrient names, values, units,
  ingredient lists — as JSON. No regex parsing needed.
- Lightweight API call, works perfectly on Streamlit Cloud's free tier.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `utils.py` | Gemini extraction, dataset matching, scoring |
| `eatsafe_master_database.csv` | 8,752-ingredient knowledge base |
| `requirements.txt` | Pip dependencies (6 packages, all lightweight) |

## Deploy on Streamlit Community Cloud

1. Get a free Gemini API key at https://aistudio.google.com/apikey
2. In your Streamlit Cloud app → **Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
3. Push and deploy. No `packages.txt` system deps needed.

## Run locally
```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key
streamlit run app.py
```
