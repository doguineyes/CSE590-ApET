"""Compare paired MMStar diagnostic reports after checking their provenance."""

import argparse
import json
from pathlib import Path


def read_predictions(report, report_path):
    path = Path(report.get("predictions_file", Path(report_path).with_suffix(".jsonl")))
    if not path.is_absolute():
        path = Path(report_path).parent / path
    if not path.is_file():
        # Saved outputs may be restored together under an attached Input path.
        path = Path(report_path).parent / path.name
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["index"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(report["dataset"]["sample_ids"]):
        raise ValueError("Prediction IDs do not match the report's sample IDs")
    if len(rows) != report["summary"]["evaluated"]:
        raise ValueError("Prediction count does not match the report summary")
    return by_id


def validate_compatible(baseline, current):
    if baseline.get("status") != "passed":
        raise ValueError("Baseline evaluation did not complete")
    if baseline.get("compression", {}).get("enabled", False):
        raise ValueError("Expected an uncompressed reference baseline")
    if not baseline["dataset"].get("sample_ids"):
        raise ValueError("Baseline sample IDs are missing")
    checks = {
        "lock_sha256": (baseline.get("lock_sha256"), current.get("lock_sha256")),
        "packages": (baseline.get("packages"), current.get("packages")),
    }
    for field in ["id", "revision", "split", "parquet_sha256", "sample_ids"]:
        checks[f"dataset.{field}"] = (baseline["dataset"].get(field), current["dataset"].get(field))
    for field in ["instruction", "max_new_tokens", "do_sample", "precision", "batch_size", "scoring"]:
        checks[f"protocol.{field}"] = (baseline["protocol"].get(field), current["protocol"].get(field))
    # Local checkpoints lack a weight fingerprint; remote pinned snapshots are
    # the tested route. Keep the same declared source for local comparisons.
    for field in ["source", "model_id", "revision", "config_sha256"]:
        checks[f"checkpoint.{field}"] = (baseline["checkpoint"].get(field), current["checkpoint"].get(field))
    for label, (before, after) in checks.items():
        if before != after:
            raise ValueError(f"Cannot compare runs: {label} differs")


def compare_runs(baseline, current, baseline_rows, current_rows):
    validate_compatible(baseline, current)
    if current.get("status") != "passed":
        raise ValueError("Current evaluation did not complete")
    if set(baseline_rows) != set(current_rows):
        raise ValueError("Prediction sample IDs differ")
    gained, lost, unchanged, changed = [], [], [], []
    for index in baseline["dataset"]["sample_ids"]:
        before, after = baseline_rows[index], current_rows[index]
        if before["gold"] != after["gold"] or before["question"] != after["question"]:
            raise ValueError(f"Question or ground truth differs for sample {index}")
        if not before["correct"] and after["correct"]:
            gained.append(index)
        if before["correct"] and not after["correct"]:
            lost.append(index)
        (unchanged if before["response"] == after["response"] else changed).append(index)
    compressed = current.get("compression", {})
    identity = compressed.get("enabled", False) and compressed.get("keep_tokens") == compressed.get("original_tokens")
    if identity and changed:
        raise ValueError(f"Identity adapter changed raw responses for samples {changed}")
    return {
        "paired_samples": len(baseline_rows),
        "baseline_accuracy": baseline["summary"]["accuracy"],
        "current_accuracy": current["summary"]["accuracy"],
        "accuracy_delta": current["summary"]["accuracy"] - baseline["summary"]["accuracy"],
        "gained_correct_ids": gained, "lost_correct_ids": lost,
        "same_raw_response_count": len(unchanged), "changed_raw_response_ids": changed,
        "baseline_unparsed": [{"index": row["index"], "response": row["response"]}
                              for row in baseline_rows.values() if row["prediction"] is None],
        "current_unparsed": [{"index": row["index"], "response": row["response"]}
                             for row in current_rows.values() if row["prediction"] is None],
        "baseline_median_generation_seconds": baseline["summary"]["median_generation_seconds"],
        "current_median_generation_seconds": current["summary"]["median_generation_seconds"],
        "identity_check": "passed" if identity else "not applicable",
        "note": "Paired subset diagnostic; generation length and instrumentation affect timing. "
                "This does not establish paper accuracy or speedup.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline, current = json.loads(args.baseline.read_text()), json.loads(args.current.read_text())
    comparison = compare_runs(baseline, current, read_predictions(baseline, args.baseline),
                              read_predictions(current, args.current))
    text = json.dumps(comparison, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
