"""Project GQA short-answer scoring; no multiple-choice extraction or model judge."""

import hashlib
from pathlib import Path

VERSION = "gqa-short-answer-v1"
INSTRUCTION = "Answer the question using a single word or phrase."


def normalize_answer(text):
    # Upstream ApET's GQA converter lowercases and removes trailing periods.
    # Native generation already strips surrounding whitespace; handle saved text too.
    return text.strip().rstrip(".").lower().strip()


def score_answer(response, gold):
    prediction = normalize_answer(response) or None
    return {"prediction": prediction, "correct": prediction == normalize_answer(gold),
            "literal_correct": response.strip() == gold.strip(), "scoring_version": VERSION}


def metadata():
    return {"version": VERSION, "owner": "CSE590 evaluation code",
            "rule": "Exact short-answer match after stripping whitespace, lowercasing and removing trailing periods. "
                    "No synonyms, answer extraction from prose, or gold-dependent parsing.",
            "reference": "scripts/convert_gqa_for_eval.py upstream conversion policy",
            "official_full_benchmark": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
