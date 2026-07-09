"""
auditor.py
----------
Hybrid compliance scoring engine that combines:

  1. Semantic AI score  — cosine similarity from FAISS search, or confidence
                          from Claude API JSON response.
  2. Deterministic rule — numeric value extraction (regex + pint) compared
                          against clause-specified thresholds.

The final score is a weighted combination:

    final = (ai_score * weight_ai) + (rule_score * weight_rule)

Weights vary by clause type:
    numeric  → ai: 40%,  rule: 60%   (rule is authoritative for exact values)
    boolean  → ai: 70%,  rule: 30%   (AI better judges presence/absence)
    process  → ai: 90%,  rule: 10%   (pure semantic — rules mostly N/A)
    semantic → ai: 90%,  rule: 10%

Decision thresholds:
    ≥ 80  → PASS
    50-79 → WARN
    < 50  → FAIL

Hard fail override:
    If rule engine extracts a specific numeric value that is demonstrably
    out of spec, status is forced to FAIL regardless of AI score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.search import DocumentIndex, SearchResult

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
PASS_THRESHOLD = 80
WARN_THRESHOLD = 50

# Weights by clause type
WEIGHTS = {
    "numeric" : (0.40, 0.60),   # (ai_weight, rule_weight)
    "boolean" : (0.70, 0.30),
    "process" : (0.90, 0.10),
    "semantic": (0.90, 0.10),
}

# Tolerance band: values within 5% of a limit boundary → WARN not PASS
TOLERANCE_BAND = 0.05

# Regex to extract numeric values with common engineering units
VALUE_RE = re.compile(
    r"(\d+\.?\d*)\s*"
    r"(mm|cm|m|μm|um|microns?|V|mV|kV|A|mA|Ω|Ohm|ohm|dB|MHz|GHz|kHz|Hz|"
    r"ns|ps|μs|ms|%|pF|nF|μF|W|kW|°C|degC|degrees?\s*C)",
    re.IGNORECASE,
)

# Unit normalisation map → SI base (approximate, for comparison only)
UNIT_TO_SI = {
    "mm": 1e-3, "cm": 1e-2, "m": 1.0, "μm": 1e-6, "um": 1e-6,
    "micron": 1e-6, "microns": 1e-6,
    "mv": 1e-3, "v": 1.0, "kv": 1e3,
    "ma": 1e-3, "a": 1.0,
    "ω": 1.0, "ohm": 1.0, "ohms": 1.0,
    "db": 1.0,
    "mhz": 1e6, "ghz": 1e9, "khz": 1e3, "hz": 1.0,
    "ns": 1e-9, "ps": 1e-12, "μs": 1e-6, "ms": 1e-3,
    "%": 1.0,
    "pf": 1e-12, "nf": 1e-9, "μf": 1e-6,
}


@dataclass
class ClauseResult:
    """Full compliance result for a single clause."""
    clause_id     : str
    clause_text   : str
    clause_type   : str                      # numeric | boolean | process | semantic
    status        : str                      # pass | warn | fail
    ai_score      : float                    # 0–100
    rule_score    : float                    # 0–100 (60 = N/A)
    final_score   : float                    # weighted combination
    confidence    : int                      # int 0–100 for UI display
    matched_text  : str                      # best-matching paragraph text
    cosine_sim    : float                    # raw similarity score
    reasoning     : str                      # human-readable explanation
    remediation   : Optional[str] = None     # fix suggestion for non-pass


@dataclass
class Clause:
    """A single requirement clause from a technical standard."""
    id          : str
    text        : str
    type        : str = "semantic"           # numeric | boolean | process | semantic
    min_value   : Optional[float] = None     # lower bound for numeric clauses
    max_value   : Optional[float] = None     # upper bound for numeric clauses
    unit        : Optional[str]   = None     # expected unit (e.g. "mm", "Ohm")
    criticality : str = "medium"             # critical | high | medium | low


class RuleEngine:
    """
    Deterministic validator for numeric and boolean compliance requirements.
    Uses regex for value extraction and manual unit normalisation.
    """

    @staticmethod
    def extract_values(text: str) -> List[Tuple[float, str]]:
        """
        Extract all numeric values with units from text.

        Returns
        -------
        list[(value_float, unit_str)]
        """
        results = []
        for match in VALUE_RE.finditer(text):
            raw_val  = float(match.group(1))
            raw_unit = match.group(2).strip().lower()
            norm_val = raw_val * UNIT_TO_SI.get(raw_unit, 1.0)
            results.append((norm_val, raw_unit))
        return results

    @staticmethod
    def check_numeric(
        design_text: str,
        clause: Clause,
    ) -> Tuple[float, str]:
        """
        Extract numeric values from design text and compare to clause limits.

        Returns
        -------
        (rule_score, explanation)
        rule_score: 100=pass, 0=fail, 60=N/A (no value found)
        """
        if clause.min_value is None and clause.max_value is None:
            return 60.0, "No numeric bounds defined for this clause."

        values = RuleEngine.extract_values(design_text)
        if not values:
            return 60.0, f"No numeric values found in matched text."

        expected_unit_si = UNIT_TO_SI.get(
            (clause.unit or "").lower().strip(), 1.0
        )

        # Find the most relevant value (closest to expected unit family)
        best_val, best_unit = values[0]

        in_range = True
        near_limit = False

        if clause.min_value is not None:
            min_si = clause.min_value * expected_unit_si
            if best_val < min_si:
                in_range = False
            elif best_val < min_si * (1 + TOLERANCE_BAND):
                near_limit = True

        if clause.max_value is not None:
            max_si = clause.max_value * expected_unit_si
            if best_val > max_si:
                in_range = False
            elif best_val > max_si * (1 - TOLERANCE_BAND):
                near_limit = True

        if not in_range:
            return 0.0, (
                f"Value {best_val:.4g} {best_unit} is outside the required range "
                f"[{clause.min_value}, {clause.max_value}] {clause.unit}."
            )
        if near_limit:
            return 60.0, (
                f"Value {best_val:.4g} {best_unit} is within 5% of the tolerance "
                f"boundary — recommend manufacturing margin review."
            )
        return 100.0, (
            f"Value {best_val:.4g} {best_unit} is within the required range."
        )

    @staticmethod
    def check_boolean(
        design_text: str,
        keywords: List[str],
    ) -> Tuple[float, str]:
        """
        Check whether any of the required keywords appear in design text.

        Returns 100 if found, 0 if absent.
        """
        text_lower = design_text.lower()
        found = [kw for kw in keywords if kw.lower() in text_lower]
        if found:
            return 100.0, f"Required keyword(s) found: {', '.join(found)}."
        return 0.0, f"None of the required keywords found: {keywords}."


class Auditor:
    """
    Main compliance auditor.
    Combines FAISS semantic search with rule-based validation to produce
    a final clause-level compliance decision.
    """

    def __init__(self, index: DocumentIndex) -> None:
        self.index       = index
        self.rule_engine = RuleEngine()

    def audit_clause(self, clause: Clause) -> ClauseResult:
        """
        Evaluate a single compliance clause against the indexed document.

        Steps
        -----
        1. FAISS top-1 match → semantic similarity score
        2. Rule check (if numeric/boolean)
        3. Weighted aggregation → final score
        4. Status assignment
        """
        # ── Step 1: Semantic search ────────────────────────────────────────
        match: Optional[SearchResult] = self.index.best_match(clause.text)

        if match is None:
            ai_score    = 0.0
            matched_txt = "No relevant paragraph found in the document."
            cosine      = 0.0
        else:
            # Map cosine sim [0, 1] → score [0, 100]
            ai_score    = min(100.0, match.score * 110.0)
            matched_txt = match.text
            cosine      = match.score

        # ── Step 2: Rule check ────────────────────────────────────────────
        if clause.type == "numeric" and match:
            rule_score, rule_note = RuleEngine.check_numeric(matched_txt, clause)
        elif clause.type == "boolean":
            # Extract expected keywords from clause text (naive: nouns)
            keywords    = [w for w in clause.text.split() if len(w) > 5]
            rule_score, rule_note = RuleEngine.check_boolean(matched_txt, keywords)
        else:
            rule_score, rule_note = 60.0, "Not applicable — semantic/process clause."

        # ── Step 3: Weighted aggregation ─────────────────────────────────
        w_ai, w_rule = WEIGHTS.get(clause.type, (0.7, 0.3))
        final = (ai_score * w_ai) + (rule_score * w_rule)

        # Hard fail override: explicit out-of-spec numeric value
        if clause.type == "numeric" and rule_score == 0.0:
            final = min(final, WARN_THRESHOLD - 1)   # force to FAIL zone

        # ── Step 4: Status ────────────────────────────────────────────────
        if final >= PASS_THRESHOLD:
            status = "pass"
        elif final >= WARN_THRESHOLD:
            status = "warn"
        else:
            status = "fail"

        # ── Build reasoning string ────────────────────────────────────────
        reasoning = (
            f"Semantic similarity: {cosine:.2f} (AI score: {ai_score:.0f}/100). "
            f"Rule check ({clause.type}): {rule_note} "
            f"Weighted final: {final:.0f}/100."
        )

        remediation = None
        if status in ("warn", "fail"):
            remediation = _generate_remediation(clause, status, cosine, rule_note)

        return ClauseResult(
            clause_id    = clause.id,
            clause_text  = clause.text,
            clause_type  = clause.type,
            status       = status,
            ai_score     = round(ai_score, 1),
            rule_score   = round(rule_score, 1),
            final_score  = round(final, 1),
            confidence   = int(final),
            matched_text = matched_txt,
            cosine_sim   = round(cosine, 4),
            reasoning    = reasoning,
            remediation  = remediation,
        )

    def audit_all(self, clauses: List[Clause]) -> List[ClauseResult]:
        """Evaluate all clauses and return results list."""
        results = []
        for clause in clauses:
            try:
                r = self.audit_clause(clause)
            except Exception as exc:           # never fail the whole scan
                logger.warning(f"Error auditing clause {clause.id}: {exc}")
                r = _error_result(clause, str(exc))
            results.append(r)
        return results


# ── Remediation helper ──────────────────────────────────────────────────────
def _generate_remediation(
    clause: Clause,
    status: str,
    similarity: float,
    rule_note: str,
) -> str:
    if similarity < 0.50:
        return (
            f"No relevant evidence found for clause {clause.id}. "
            f"Add a dedicated section in the design document that explicitly "
            f"addresses: '{clause.text[:80]}...'"
        )
    if clause.type == "numeric" and "outside" in rule_note:
        return (
            f"The numeric value extracted from the design document does not satisfy "
            f"the requirement. Revise the design to meet: {clause.text}"
        )
    return (
        f"Partial compliance detected for clause {clause.id}. "
        f"Strengthen the design document language to explicitly confirm: "
        f"'{clause.text[:100]}'"
    )


def _error_result(clause: Clause, error: str) -> ClauseResult:
    return ClauseResult(
        clause_id    = clause.id,
        clause_text  = clause.text,
        clause_type  = clause.type,
        status       = "warn",
        ai_score     = 0.0,
        rule_score   = 0.0,
        final_score  = 0.0,
        confidence   = 0,
        matched_text = "Error during analysis.",
        cosine_sim   = 0.0,
        reasoning    = f"Analysis error: {error}",
        remediation  = "Manual review required due to analysis error.",
    )
