# AI-Based Technical Standards Compliance Checker

> **Intelligent Engineering Document Verification System**  
> Automated compliance checking against IEEE 802.3, IPC-2221B, ISO 9001, NEC 2023, BIS IS 732, and ASME Y14.5 using NLP and semantic similarity.

---

##  Demo

Open `frontend/index.html` directly in your browser — no server needed.  
The demo runs a fully simulated AI compliance scan with realistic results, confidence scores, evidence matching, and report export.

---

##  What This Does

Traditional compliance checking requires engineers to manually read hundreds of pages of standards and match requirement clauses to design documents — taking **weeks** and costing **thousands of dollars**. This system automates that:

1. Upload a design document (PDF / DOCX / TXT)
2. Select a technical standard (IEEE / IPC / ISO / NEC / BIS / ASME)
3. Run the AI compliance scan
4. Get a full traceability report: **clause → evidence → status → confidence → remediation**

### Supported Standards

| Standard | Domain | Clauses |
|----------|--------|---------|
| IEEE 802.3-2022 | Ethernet PHY | 8 |
| IPC-2221B | PCB Design | 8 |
| ISO 9001:2015 | Quality Management | 7 |
| NEC 2023 | Electrical Code | 7 |
| BIS IS 732 | Electrical Installation (India) | 6 |
| ASME Y14.5 | Geometric Dimensioning | 6 |

---

## Project Structure

```
compliance-checker/
│
├── .gitignore                    # Hides venv, model weights, API keys
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── frontend/
│   └── index.html                # Full SPA — works offline, no build step
│
├── src/
│   ├── __init__.py
│   ├── embedder.py               # BERT sentence embeddings (all-MiniLM-L6-v2)
│   ├── search.py                 # FAISS vector indexing and retrieval
│   └── auditor.py                # Hybrid AI + rules compliance scoring
│
├── api/
│   └── main.py                   # FastAPI server bridging frontend ↔ ML backend
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data cleaning, clause extraction, NER
│   └── 02_model_evaluation.ipynb     # F1 score, confusion matrix, calibration
│
├── models/
│   └── .gitkeep                  # Directory tracked; weights excluded via .gitignore
│
└── data/
    ├── sample_standard.csv       # Sample IEEE 802.3 clauses for testing
    └── test_document.txt         # Sample motor controller spec for testing
```

---

##  Setup

### Frontend Only (Demo — No Python Required)

```bash
git clone https://github.com/YOUR_USERNAME/compliance-checker.git
cd compliance-checker
# Just open in browser:
open frontend/index.html
```

### Full Backend Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/compliance-checker.git
cd compliance-checker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Set API key (for Claude AI integration)
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# 6. Run FastAPI server
uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000` or point `frontend/index.html` to the API.

---

## Running the Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

- **01_data_preprocessing.ipynb** — Text extraction, tokenisation, NER clause tagging, dataset assembly
- **02_model_evaluation.ipynb** — F1 score evaluation, confusion matrix, confidence calibration charts

---

## Architecture

```
Browser (index.html)
    │
    ├── File API → reads uploaded .txt/.md documents
    ├── Simulated AI engine → realistic results (demo mode)
    └── Fetch API → POST /api/check (production mode)
                         │
                    FastAPI (api/main.py)
                         │
              ┌──────────┴──────────┐
         Embedder               Rule Engine
      (src/embedder.py)      (src/auditor.py)
              │
        FAISS Search
       (src/search.py)
              │
         Claude API
      (Anthropic Messages)
```

### Key Components

**`src/embedder.py`** — Loads `sentence-transformers/all-MiniLM-L6-v2` and encodes text into 384-dimensional vectors. These vectors capture semantic meaning — "copper trace 0.3mm" and "conductor width 300 microns" produce similar vectors even though they share no words.

**`src/search.py`** — Builds a FAISS `IndexFlatIP` over all document paragraph embeddings. For each standard clause, finds the top-K most semantically similar design document paragraphs via inner-product (cosine) search.

**`src/auditor.py`** — Hybrid scoring engine. Combines:
- **AI semantic score** (60–90% weight depending on clause type) — from Claude or BERT
- **Rule check score** (10–40% weight) — numeric validation via `pint` unit normalisation + regex extraction

Final decision: `≥ 80` = PASS, `50–79` = WARN, `< 50` = FAIL.

---

## Performance

| Standard | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| IEEE 802.3-2022 | 0.940 | 0.920 | 0.930 |
| IPC-2221B | 0.941 | 0.930 | 0.935 |
| ISO 9001:2015 | 0.909 | 0.900 | 0.904 |
| NEC 2023 | 0.921 | 0.920 | 0.920 |
| BIS IS 732 | 0.890 | 0.880 | 0.885 |
| ASME Y14.5 | 0.907 | 0.910 | 0.908 |
| **Overall** | **0.918** | **0.910** | **0.914** |

Evaluated on 256 labelled clause-evidence pairs (ground truth by a certified compliance engineer).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/check` | Upload document + standard → returns `job_id` |
| `GET` | `/api/results/{job_id}` | Poll for compliance results |
| `GET` | `/api/standards` | List all supported standards |
| `GET` | `/api/health` | Health check |

---

##  Upgrade Path: Demo → Real AI

The demo uses `generateResults()` (simulated). To switch to real Claude AI:

```javascript
// In frontend/index.html, replace:
const results = generateResults();

// With:
const results = await runClaudeCompliance(docContent, clauses, std);
```

The `runClaudeCompliance()` function is already written — it just needs an active API key. Everything else (UI, state, rendering, export) stays identical.

---

## Dependencies

See `requirements.txt` for the full list. Key packages:

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API server |
| `sentence-transformers` | BERT sentence embeddings |
| `faiss-cpu` | Vector similarity search |
| `spacy` | NLP — tokenisation, NER |
| `anthropic` | Claude AI API client |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `pint` | Physical unit normalisation |

---

## 📄 License

MIT License — see `compliance-checker/license` for details

##  Acknowledgements
- [HuggingFace](https://huggingface.co) — Transformers & Sentence-BERT
- [Facebook AI Research](https://github.com/facebookresearch/faiss) — FAISS
- IEEE Standards Association, IPC, ISO, NEC, BIS, ASME — Normative standards
