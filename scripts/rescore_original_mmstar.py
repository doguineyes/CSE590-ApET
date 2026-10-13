"""Rescore a saved native MMStar JSON/ZIP on CPU without changing source evidence."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_original_env import command_output, verify_native_reference
from eval_original_mmstar import update_summary
from mmstar_scoring import score_answer, scoring_metadata


def load_saved(input_path):
    """Read explicit files; ZIP members are never extracted as filesystem paths."""
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            names = archive.namelist()
            reports = [name for name in names if name.endswith(".json") and name[:-5] + ".jsonl" in names]
            if len(reports) != 1:
                raise ValueError("Expected one paired MMStar report and JSONL in the ZIP")
            report_bytes = archive.read(reports[0])
            prediction_bytes = archive.read(reports[0][:-5] + ".jsonl")
    else:
        report_bytes = input_path.read_bytes()
        report = json.loads(report_bytes)
        prediction_name = Path(report.get("predictions_file", input_path.with_suffix(".jsonl").name)).name
        prediction_bytes = (input_path.parent / prediction_name).read_bytes()
    report = json.loads(report_bytes)
    rows = [json.loads(line) for line in prediction_bytes.decode().splitlines() if line.strip()]
    return report, rows, hashlib.sha256(prediction_bytes).hexdigest()


def validate_saved(report, rows):
    if report.get("status") != "passed":
        raise ValueError("Rescoring expects a completed native evaluation")
    configurations = list(report["protocol"]["configurations"])
    if "baseline" not in configurations:
        raise ValueError("Missing baseline configuration")
    ids = report["dataset"]["sample_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate manifest sample IDs")
    expected = {(name, index) for name in configurations for index in ids}
    actual = [(row["configuration"], row["index"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Prediction IDs/configurations do not match the completed manifest")
    reference = {}
    for row in rows:
        if row["gold"] not in "ABCD" or len(row["gold"]) != 1 or not isinstance(row["response"], str):
            raise ValueError("Invalid ground truth or response")
        identity = tuple(row[key] for key in ["gold", "original_question", "prompt", "image_sha256", "seed"])
        if row["index"] in reference and reference[row["index"]] != identity:
            raise ValueError("Paired generation inputs differ")
        reference[row["index"]] = identity
    verify_native_reference(report)
    return configurations


def image_bytes(input_path, relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Expected a relative image path within the saved result")
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            return archive.read(relative)
    return (input_path.parent / name).read_bytes()


def rescore(input_path, output, include_images=True):
    input_path, output = Path(input_path), Path(output)
    image_dir = output.parent / (output.stem + "-images")
    if output.exists() or output.with_suffix(".jsonl").exists() or (include_images and image_dir.exists()):
        raise ValueError("Output exists; choose a new name to preserve existing evidence")
    source, rows, predictions_sha = load_saved(input_path)
    report = copy.deepcopy(source)
    configurations = validate_saved(report, rows)
    report["generation_status"] = source["status"]
    report["rescoring"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path.resolve()), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_predictions_sha256": predictions_sha, "original_scoring": source.get("scoring", source["protocol"]["scoring"]),
        "original_summaries": source["summaries"], "original_comparisons": source["comparisons"],
        "scorer_commit": command_output(["git", "rev-parse", "HEAD"]),
        "scorer_git_status": command_output(["git", "status", "--short"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_inference_rerun": False, "images_included": include_images,
        "note": "Generation provenance/timing and raw answers are retained. Only scoring and portable image paths change.",
    }
    report["scoring"] = scoring_metadata()
    report["protocol"]["scoring"] = report["scoring"]["rule"]
    report["predictions_file"] = output.with_suffix(".jsonl").name
    output.parent.mkdir(parents=True, exist_ok=True)
    if include_images:
        image_dir.mkdir()
    copied = {}
    changed = []
    for row in rows:
        previous = {key: row.get(key) for key in ["prediction", "correct", "scoring_version", "format_compliant"]}
        row["previous_scoring"] = previous
        row.update(score_answer(row["response"], row["original_question"], row["gold"]))
        if (previous["prediction"], previous["correct"]) != (row["prediction"], row["correct"]):
            changed.append({"index": row["index"], "configuration": row["configuration"],
                            "response": row["response"], "before": previous,
                            "after": {"prediction": row["prediction"], "correct": row["correct"]}})
        if include_images:
            if row["index"] not in copied:
                raw = image_bytes(input_path, row["image"])
                if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
                    raise ValueError(f"Image hash mismatch for sample {row['index']}")
                target = image_dir / (str(row["index"]) + Path(row["image"]).suffix)
                target.write_bytes(raw)
                copied[row["index"]] = str(target.relative_to(output.parent))
            row["image"] = copied[row["index"]]
    if include_images:
        report["dataset"]["image_directory"] = image_dir.name
    report["rescoring"]["changed_scores"] = changed
    update_summary(report, rows, configurations)
    with output.with_suffix(".jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved original MMStar result ZIP or JSON report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-images", action="store_true", help="Score only; retain old image references")
    args = parser.parse_args()
    try:
        report = rescore(args.input, args.output, not args.without_images)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Rescoring failed: {error}\n")
    print(json.dumps({"output": str(args.output), "scoring": report["scoring"]["version"],
                      "changed_scores": report["rescoring"]["changed_scores"],
                      "summaries": report["summaries"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
