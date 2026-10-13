"""Project-owned MMStar scoring; independent of native ApET and model execution."""

import hashlib
from pathlib import Path
import re

from eval_mmstar import parse_answer as strict_parse_answer, summarize

VERSION = "mmstar-explicit-choice-v2"
PREFIX = r"(?:(?:THE\s+)?(?:CORRECT\s+)?ANSWER(?:\s+IS)?\s*[:=]?\s*|OPTION\s+)?"


def option_texts(question):
    """Read exactly four ordered labels, avoiding guesses from arbitrary prose."""
    matches = list(re.finditer(r"(?<!\w)(?:\(([A-D])\)|([A-D])[:.)])\s*", question))
    labels = [match.group(1) or match.group(2) for match in matches]
    if labels != list("ABCD"):
        return {}
    return {label: question[match.end():matches[i + 1].start() if i < 3 else len(question)].strip().rstrip(",;").strip()
            for i, (label, match) in enumerate(zip(labels, matches))}


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold().rstrip(".")


def score_answer(response, question, gold):
    """Accept a strict choice, or a choice followed by its matching option text.

    Ground truth is used only to calculate correctness after parsing. Ambiguous
    choices, conflicting option text, and free-form explanations stay unparsed.
    """
    strict = strict_parse_answer(response)
    prediction, method = strict, "strict_choice" if strict else "unparsed"
    if prediction is None:
        match = re.fullmatch(PREFIX + r"\(?([A-D])\)?\s*[:.)-]\s*(.+)", response.strip(), re.I | re.S)
        if match:
            label, text = match.group(1).upper(), match.group(2)
            options = option_texts(question)
            if label in options and options[label] and normalize_text(text) == normalize_text(options[label]):
                prediction, method = label, "matched_option_text"
    return {"prediction": prediction, "correct": prediction == gold,
            "strict_prediction": strict, "strict_correct": strict == gold,
            "format_compliant": re.fullmatch(r"[A-D]", response.strip()) is not None,
            "parse_method": method, "scoring_version": VERSION}


def summarize_scored(rows):
    result = summarize(rows)
    compliant = sum(row["format_compliant"] for row in rows)
    strict_correct = sum(row["strict_correct"] for row in rows)
    result.update({"format_compliant": compliant,
                   "format_compliance_rate": compliant / len(rows) if rows else None,
                   "format_noncompliant_ids": [row["index"] for row in rows if not row["format_compliant"]],
                   "strict_correct": strict_correct,
                   "strict_accuracy": strict_correct / len(rows) if rows else None,
                   "option_text_parsed_ids": [row["index"] for row in rows if row["parse_method"] == "matched_option_text"],
                   "unparsed_ids": [row["index"] for row in rows if row["prediction"] is None]})
    return result


def scoring_metadata():
    root = Path(__file__).resolve().parent
    return {"version": VERSION, "owner": "CSE590 evaluation code; not upstream ApET",
            "rule": "Strict explicit choice, or explicit choice plus matching labeled option text. "
                    "Unparsed responses count as incorrect; gold is never used for parsing.",
            "format_rule": "After trimming whitespace, exactly one uppercase A/B/C/D.",
            "text_normalization": "Collapse whitespace, ignore case and trailing periods.",
            "strict_reference": "Previous conservative parser retained as separate strict metrics.",
            "official_mmstar_scoring": False,
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ["mmstar_scoring.py", "eval_mmstar.py"]}}
