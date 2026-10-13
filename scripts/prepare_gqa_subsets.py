"""Freeze image-disjoint GQA development/evaluation subsets and nested diagnostics."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_gqa_full import QUESTION_COUNT, IMAGE_COUNT, full_questions
from prepare_gqa_pilot import (
    DATASET_ID, DATASET_REVISION, FILES, SCHEMA_VERSION as PILOT_SCHEMA,
    fetch_sources, image_seed, select_groups,
)

STUDY_SCHEMA = "gqa-fixed-image-splits-v1"
SUBSET_SCHEMA = "gqa-fixed-subset-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ranked_ids(ids, seed, role):
    return sorted(ids, key=lambda image_id: hashlib.sha256(f"gqa-subsets:{seed}:{role}:{image_id}".encode()).hexdigest())


def select_sets(questions, images, seed=590, development_images=100, evaluation_images=100,
                questions_per_image=8, diagnostic_images=20, diagnostic_questions=4):
    questions = full_questions(questions, images)
    grouped = defaultdict(list)
    for row in questions:
        grouped[row["imageId"]].append(row)
    candidates = {}
    for image_id, rows in grouped.items():
        unique = {" ".join(row["question"].split()).casefold() for row in rows}
        if (len(unique) >= questions_per_image
                and {"query", "verify"} <= {row["types"]["structural"] for row in rows}
                and len({row["types"]["semantic"] for row in rows}) >= 2):
            chosen, _ = select_groups(rows, {image_id}, 1, questions_per_image, seed)
            candidates[image_id] = chosen
    if (questions_per_image < diagnostic_questions or diagnostic_questions < 2
            or diagnostic_images > development_images or diagnostic_images < 1):
        raise ValueError("Diagnostics must fit inside development, with at least two questions per image")
    if development_images < 1 or evaluation_images < 1 or len(candidates) < development_images + evaluation_images:
        raise ValueError("Not enough eligible images for disjoint development/evaluation")
    # The already inspected pilot belongs to development, never reserved evaluation.
    # Recompute its IDs from source metadata; no local result files are required.
    pilot, _ = select_groups(questions, images, 10, 4, 590)
    pilot_ids = sorted({row["imageId"] for row in pilot})
    seen_eligible = sorted(set(pilot_ids) & set(candidates))
    if len(seen_eligible) > development_images:
        raise ValueError("Development must include all eligible previously inspected pilot images")
    development = sorted(seen_eligible + ranked_ids(set(candidates) - set(pilot_ids), seed, "development")[:development_images - len(seen_eligible)])
    available = set(candidates) - set(development) - set(pilot_ids)
    if len(available) < evaluation_images:
        raise ValueError("Insufficient unseen eligible images for evaluation")
    evaluation = sorted(ranked_ids(available, seed, "evaluation")[:evaluation_images])
    diagnostics = sorted(seen_eligible[:diagnostic_images] + ranked_ids(set(development) - set(seen_eligible), seed, "diagnostic")[:max(0, diagnostic_images - len(seen_eligible))])
    diagnostic_rows = []
    for image_id in diagnostics:
        chosen, _ = select_groups(grouped[image_id], {image_id}, 1, diagnostic_questions, seed)
        if not {row["id"] for row in chosen} <= {row["id"] for row in candidates[image_id]}:
            raise ValueError("Diagnostic questions must be nested in development")
        diagnostic_rows.extend(chosen)
    sets = {"development": [row for image_id in development for row in candidates[image_id]],
            "evaluation": [row for image_id in evaluation for row in candidates[image_id]],
            "diagnostic": diagnostic_rows}
    assignment = {"development": development, "evaluation": evaluation,
                  "unused": sorted(set(images) - set(development) - set(evaluation))}
    # Prevent accidental leakage through different IDs for identical raw photos.
    development_hashes = {images[image_id]["sha256"] for image_id in development}
    if development_hashes & {images[image_id]["sha256"] for image_id in evaluation}:
        raise ValueError("Duplicate raw photos cross development/evaluation; revise image grouping")
    return sets, assignment, pilot_ids, len(candidates)


def prepare(args):
    import pyarrow.parquet as pq
    from PIL import Image

    if args.output_dir.exists() or args.output_dir.with_suffix(".zip").exists():
        raise ValueError("Output exists; use a new versioned directory")
    paths = fetch_sources(args.dataset_cache, args.source_dir, args.download_budget_mib)
    questions = pq.read_table(paths[next(iter(FILES))]).to_pylist()
    raw_images = pq.read_table(paths[list(FILES)[1]]).to_pylist()
    if len(questions) != QUESTION_COUNT or len(raw_images) != IMAGE_COUNT:
        raise ValueError("Unexpected pinned source row counts")
    images = {row["id"]: {"bytes": row["image"]["bytes"],
        "sha256": hashlib.sha256(row["image"]["bytes"]).hexdigest()} for row in raw_images}
    if len(images) != IMAGE_COUNT:
        raise ValueError("Duplicate source image IDs")
    sets, assignment, pilot_ids, eligible = select_sets(questions, images, args.seed,
        args.development_images, args.evaluation_images, args.questions_per_image,
        args.diagnostic_images, args.diagnostic_questions)
    source = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": "testdev",
        "tier": "testdev_balanced", "files": FILES, "total_questions": QUESTION_COUNT,
        "total_images": IMAGE_COUNT, "original_dataset": "https://cs.stanford.edu/people/dorarad/gqa/"}
    rule = ("Image groups assigned by seeded SHA256 ranking. Eligible images have query/verify and >=2 semantic types. "
            "Previously inspected pilot images excluded from evaluation. Questions chosen for type diversity; "
            "no model predictions or correctness used. Equal questions per selected image; not population-weighted full GQA.")
    selected_rows = {row["id"]: row for rows in sets.values() for row in rows}
    study = {"schema": STUDY_SCHEMA, "source": source, "seed": args.seed, "selection_rule": rule,
        "prepared_by_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "eligible_images": eligible, "previously_inspected_pilot_images": pilot_ids,
        "assignment": assignment,
        "question_ids": {role: [row["id"] for row in rows] for role, rows in sets.items()},
        "question_sha256": {qid: canonical_hash(row) for qid, row in sorted(selected_rows.items())},
        "image_sha256": {image_id: image["sha256"] for image_id, image in sorted(images.items())},
        "counts": {role: {"images": len({row["imageId"] for row in rows}), "questions": len(rows)}
                   for role, rows in sets.items()},
        "note": "Reserved evaluation is a project split within labeled Test-Dev, not the official hidden test set. "
                "Keep evaluation out of tuning; diagnostics are nested in development."}
    study["id"] = canonical_hash(study)
    args.output_dir.mkdir(parents=True)
    study_path = args.output_dir / "study.json"
    study_path.write_text(json.dumps(study, indent=2) + "\n")
    study_sha = hashlib.sha256(study_path.read_bytes()).hexdigest()
    image_records = {}
    for image_id in sorted(set(assignment["development"]) | set(assignment["evaluation"])):
        raw = images[image_id]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")
            record = {"file": "images/" + image_id + suffix, "image_size": list(image.size),
                "sha256": images[image_id]["sha256"],
                "rgb_pixels_sha256": hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
                "compression_seed": image_seed(args.seed, image_id)}
        image_records[image_id] = record
    for role, rows in sets.items():
        directory = args.output_dir / role
        (directory / "images").mkdir(parents=True)
        records = {image_id: image_records[image_id] for image_id in sorted({row["imageId"] for row in rows})}
        for image_id, record in records.items():
            (directory / record["file"]).write_bytes(images[image_id]["bytes"])
        per_image = args.diagnostic_questions if role == "diagnostic" else args.questions_per_image
        manifest = {"schema": PILOT_SCHEMA if role == "diagnostic" else SUBSET_SCHEMA, "source": source,
            "study": {"id": study["id"], "role": role, "manifest_sha256": study_sha},
            "selection": {"seed": args.seed, "image_count": len(records), "question_count": len(rows),
                "questions_per_image": per_image, "image_ids": list(records),
                "question_ids": [row["id"] for row in rows],
                "questions_by_image": dict(sorted(Counter(row["imageId"] for row in rows).items())),
                "eligible_images": eligible, "rule": rule,
                "type_counts": {field: dict(sorted(Counter(row["types"][field] for row in rows).items()))
                                for field in ["semantic", "structural", "detailed"]}},
            "images": records, "questions": rows, "prepared_by_sha256": study["prepared_by_sha256"],
            "note": "Frozen metadata-selected subset. Report this subset's score, not full GQA accuracy."}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_zip:
        with zipfile.ZipFile(args.output_dir.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(args.output_dir)))
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--development-images", type=int, default=100)
    parser.add_argument("--evaluation-images", type=int, default=100)
    parser.add_argument("--questions-per-image", type=int, default=8)
    parser.add_argument("--diagnostic-images", type=int, default=20)
    parser.add_argument("--diagnostic-questions", type=int, default=4)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-gqa-cache"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--download-budget-mib", type=float, default=100)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if (min(args.development_images, args.evaluation_images, args.diagnostic_images) < 1
            or args.questions_per_image < 4 or args.diagnostic_questions < 2
            or not math.isfinite(args.download_budget_mib) or args.download_budget_mib < 0):
        parser.error("Use positive image counts, >=4 questions/image, >=2 diagnostic questions and a finite nonnegative budget")
    try:
        study = prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Subset preparation failed: {error}\n")
    print(json.dumps({"study": str(args.output_dir / "study.json"), "study_id": study["id"], "counts": study["counts"]}, indent=2))


if __name__ == "__main__":
    main()
