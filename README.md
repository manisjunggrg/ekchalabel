# e-K Cha Label? — FMCG Label Analyzer

Surya OCR → zone splitting → nutrition parsing → 8.7k EatSafe database match → score.

## OCR engine: Surya

**Surya** (v0.16.x) is a deep-learning document OCR toolkit supporting 90+ languages,
far more accurate than Tesseract on photographed labels. It runs locally — no API key needed.

⚠️ **Surya requires ~2 GB RAM** (PyTorch + models). It will NOT fit on Streamlit
Community Cloud's free tier (1 GB). Options:
- Run locally: `pip install -r requirements.txt && streamlit run app.py`
- Upgrade Streamlit Cloud to a paid plan with more memory
- Use a Docker/VPS deployment with ≥4 GB RAM

Models download automatically on first run (~500 MB one-time download).

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `utils.py` | Surya OCR, zone splitting, parsing, matching, scoring |
| `eatsafe_master_database.csv` | 8,752-ingredient knowledge base |
| `requirements.txt` | Dependencies |

## Run on Google Colab (recommended — free, no setup)

1. Upload `run_in_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `app.py`, `utils.py`, `eatsafe_master_database.csv` to `/content/ekchalabel/`
   using the file browser (folder icon on the left)
3. Run Cell 1 (installs dependencies)
4. Run Cell 2 (launches Streamlit with a public URL)

Colab gives you ~12 GB RAM and optional GPU — more than enough for Surya.
For faster OCR, switch to a GPU runtime (`Runtime → Change runtime type → T4 GPU`)
and the notebook sets `TORCH_DEVICE=cuda` automatically.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
