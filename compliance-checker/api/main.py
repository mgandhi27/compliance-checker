"""
api/main.py
-----------
FastAPI server — bridges the frontend HTML with the ML backend.

Endpoints
---------
POST /api/check          Upload document + select standard → returns job_id
GET  /api/results/{id}   Poll for completed compliance results
GET  /api/standards      List all supported standards
GET  /api/health         Health check
GET  /                   Serve frontend/index.html
"""

from __future__ import annotations

import os
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embedder import Embedder
from src.search   import DocumentIndex
from src.auditor  import Auditor, Clause

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AI Compliance Checker API",
    description = "Automated engineering document compliance verification",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── In-memory job store (use Redis in production) ───────────────────────────
jobs: Dict[str, dict] = {}

# ── Shared embedder (loaded once) ───────────────────────────────────────────
embedder = Embedder()

# ── Standards & Clauses registry ────────────────────────────────────────────
STANDARDS = {
    "IEEE": {"name": "IEEE 802.3-2022", "desc": "Ethernet Physical Layer"},
    "IPC":  {"name": "IPC-2221B",       "desc": "PCB Design Standard"},
    "ISO":  {"name": "ISO 9001:2015",   "desc": "Quality Management Systems"},
    "NEC":  {"name": "NEC 2023",        "desc": "National Electrical Code"},
    "BIS":  {"name": "BIS IS 732",      "desc": "Electrical Installations (India)"},
    "ASME": {"name": "ASME Y14.5",      "desc": "Geometric Dimensioning & Tolerancing"},
}

STD_CLAUSES: Dict[str, List[Clause]] = {
    "IEEE": [
        Clause("55.5.3.2", "Differential output voltage at MDI shall be between 0.67V and 1.33V peak-to-peak", "numeric", 0.67, 1.33, "V"),
        Clause("55.5.3.3", "Output impedance at MDI shall be 85 to 115 Ohm differential", "numeric", 85, 115, "Ohm"),
        Clause("55.5.3.4", "Return loss shall be at minimum 12 dB across 1 to 500 MHz", "numeric", 12, None, "dB"),
        Clause("55.5.3.6", "Transmit pair skew shall not exceed 2.5 ns between any two pairs", "numeric", None, 2.5, "ns"),
        Clause("28.2.1",   "Auto-negotiation shall be implemented and enabled by default", "boolean"),
        Clause("73.6.1",   "Clause 73 base page shall correctly encode all supported link modes", "semantic"),
        Clause("55.4.2",   "Link training shall complete within 10 seconds of link partner detection", "numeric", None, 10, "s"),
        Clause("78.3.1",   "EEE capability shall be advertised during auto-negotiation if implemented", "boolean"),
    ],
    "IPC": [
        Clause("4.1.2", "All PCB traces carrying more than 1A shall have minimum width of 0.3mm at 25 degrees C ambient", "numeric", 0.3, None, "mm"),
        Clause("4.3.1", "Solder mask clearance shall be minimum 0.05mm from copper features", "numeric", 0.05, None, "mm"),
        Clause("5.2.4", "Via drill diameter shall not be less than 0.2mm for any signal layer", "numeric", 0.2, None, "mm"),
        Clause("6.1.1", "All components shall be rated for the specified operating temperature range", "boolean"),
        Clause("6.4.2", "Conformal coating shall be applied to assemblies for harsh environments", "boolean"),
        Clause("7.2.1", "Impedance controlled traces shall maintain specified impedance within 10 percent", "numeric", None, 10, "%"),
        Clause("8.1.1", "Each PCB assembly shall carry a unique serial number visible on silkscreen", "boolean"),
        Clause("9.4.1", "BGA component land patterns shall comply with IPC-7351 recommendations", "semantic"),
    ],
    "ISO": [
        Clause("4.1",  "Organisation shall determine external and internal issues relevant to QMS", "process"),
        Clause("6.1",  "Organisation shall determine risks and opportunities for QMS effectiveness", "process"),
        Clause("7.2",  "Organisation shall determine necessary competence of persons affecting quality", "process"),
        Clause("8.1",  "Organisation shall plan, implement, control, and monitor product realisation", "process"),
        Clause("8.3",  "Organisation shall establish documented design and development process", "process"),
        Clause("9.1",  "Organisation shall monitor, measure, analyse and evaluate quality performance", "process"),
        Clause("10.2", "Organisation shall react to nonconformities and take corrective action", "process"),
    ],
    "NEC": [
        Clause("110.3",  "Listed or labelled equipment shall be installed per manufacturer instructions", "boolean"),
        Clause("210.8",  "GFCI protection shall be provided for personnel in specified locations", "boolean"),
        Clause("230.42", "Service entrance conductors shall have ampacity not less than load served", "semantic"),
        Clause("300.5",  "Underground wiring shall be installed at minimum cover depths per conductor type", "semantic"),
        Clause("310.15", "Conductor ampacity shall be per applicable table and installation conditions", "semantic"),
        Clause("408.3",  "Switchboards shall be protected and accessible only to qualified persons", "boolean"),
        Clause("501.10", "Class I Division 1 wiring shall use approved explosion-proof systems", "semantic"),
    ],
    "BIS": [
        Clause("3.1.1", "Installations shall be carried out by licensed contractors or qualified engineer", "boolean"),
        Clause("4.2.3", "Earthing conductors shall have cross-section per Table 3 of the standard", "semantic"),
        Clause("5.1.2", "All switchgear shall be suitable for the prospective short-circuit current", "boolean"),
        Clause("6.3.1", "Cable routes shall minimise risks from mechanical damage, heat and corrosion", "semantic"),
        Clause("7.1.4", "Every installation shall be protected against overcurrent before danger arises", "boolean"),
        Clause("8.2.1", "Test certificates and records shall be maintained for minimum 10 years", "numeric", 10, None, "years"),
    ],
    "ASME": [
        Clause("1.4",   "All drawings shall use inch or millimetre system consistently with units in title block", "boolean"),
        Clause("2.1.1", "Geometric tolerances shall be specified to control form, orientation, location", "boolean"),
        Clause("3.3.2", "Datum reference frames shall be established using functional features", "semantic"),
        Clause("4.5.1", "Position tolerances shall be specified for all features located to datums", "boolean"),
        Clause("5.2",   "Surface finish symbols shall specify Ra or Rz on all machined surfaces", "boolean"),
        Clause("6.1",   "All dimensions shall have associated tolerances — direct, block, or geometric", "boolean"),
    ],
}


# ── Pydantic models ─────────────────────────────────────────────────────────
class StandardInfo(BaseModel):
    id      : str
    name    : str
    desc    : str
    clauses : int


class JobStatus(BaseModel):
    job_id  : str
    status  : str   # queued | processing | complete | failed
    message : str


class ClauseResultOut(BaseModel):
    clause_id    : str
    clause_text  : str
    status       : str
    confidence   : int
    matched_text : str
    reasoning    : str
    remediation  : Optional[str]


class ComplianceReport(BaseModel):
    job_id       : str
    document_name: str
    standard_id  : str
    standard_name: str
    total_clauses: int
    pass_count   : int
    warn_count   : int
    fail_count   : int
    score_pct    : int
    results      : List[ClauseResultOut]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/standards", response_model=List[StandardInfo])
async def list_standards():
    return [
        StandardInfo(
            id      = sid,
            name    = info["name"],
            desc    = info["desc"],
            clauses = len(STD_CLAUSES.get(sid, [])),
        )
        for sid, info in STANDARDS.items()
    ]


@app.post("/api/check", response_model=JobStatus)
async def create_compliance_check(
    standard_id: str = Form(...),
    document   : UploadFile = File(...),
):
    if standard_id not in STANDARDS:
        raise HTTPException(status_code=400, detail=f"Unknown standard: {standard_id}")

    content = await document.read()
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode document as text.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "doc_name": document.filename}

    # Run analysis in background (use Celery in production)
    asyncio.create_task(_run_scan(job_id, text, document.filename, standard_id))

    return JobStatus(job_id=job_id, status="queued", message="Scan queued.")


async def _run_scan(job_id: str, text: str, doc_name: str, std_id: str):
    """Background task: run the full compliance analysis pipeline."""
    jobs[job_id]["status"] = "processing"
    try:
        # 1. Chunk document into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        if not paragraphs:
            paragraphs = [text[i:i+500] for i in range(0, len(text), 450)]

        # 2. Build FAISS index
        index = DocumentIndex(embedder)
        index.build(paragraphs)

        # 3. Run auditor
        auditor = Auditor(index)
        clauses = STD_CLAUSES.get(std_id, [])
        raw_results = auditor.audit_all(clauses)

        # 4. Build report
        pass_n = sum(1 for r in raw_results if r.status == "pass")
        warn_n = sum(1 for r in raw_results if r.status == "warn")
        fail_n = sum(1 for r in raw_results if r.status == "fail")
        score  = round(pass_n / len(raw_results) * 100) if raw_results else 0

        report = ComplianceReport(
            job_id        = job_id,
            document_name = doc_name,
            standard_id   = std_id,
            standard_name = STANDARDS[std_id]["name"],
            total_clauses = len(raw_results),
            pass_count    = pass_n,
            warn_count    = warn_n,
            fail_count    = fail_n,
            score_pct     = score,
            results       = [
                ClauseResultOut(
                    clause_id    = r.clause_id,
                    clause_text  = r.clause_text,
                    status       = r.status,
                    confidence   = r.confidence,
                    matched_text = r.matched_text,
                    reasoning    = r.reasoning,
                    remediation  = r.remediation,
                )
                for r in raw_results
            ],
        )
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["report"] = report.model_dump()

    except Exception as exc:
        logger.error(f"Scan {job_id} failed: {exc}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(exc)


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "processing":
        return {"status": "processing"}
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Unknown error"))
    return job.get("report", {"status": "queued"})


# ── Serve frontend ───────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found — open frontend/index.html directly.</h1>")
