"""Prepare all labeled GQA balanced Test-Dev questions and raw images on CPU."""

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, fetch_sources, image_seed

SCHEMA_VERSION = "gqa-testdev-full-v1"
QUESTION_COUNT = 12578
IMAGE_COUNT = 398


def full_questions(questions, image_ids):
    """Retain every source row, including repeated question text with distinct IDs."""
    seen = set()
    for row in questions:
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("Missing or duplicate GQA question ID")
        seen.add(row["id"])
        if (row.get("imageId") not in image_ids or row.get("isBalanced") is not True
                or not isinstance(row.get("types"), dict)
                or any(not isinstance(row["types"].get(field), str)
                       for field in ["semantic", "structural", "detailed"])
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or "<image" in row["question"]
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()):
            raise ValueError(f"Invalid labeled balanced GQA row: {row['id']}")
    if {row["imageId"] for row in questions} != set(image_ids):
        raise ValueError("Source questions do not cover exactly the source images")
    return sorted(questions, key=lambda row: (row["imageId"], row["id"]))


def prepare(args):
    import pyarrow.parquet as pq
    from PIL import Image

    if args.output_dir.exists() or args.output_dir.with_suffix(".zip").exists():
        raise ValueError("Output exists; choose a new directory")
    paths = fetch_sources(args.dataset_cache, args.source_dir, args.download_budget_mib)
    questions = pq.read_table(paths[next(iter(FILES))]).to_pylist()
    images = pq.read_table(paths[list(FILES)[1]]).to_pylist()
    by_image = {row["id"]: row["image"] for row in images}
    if len(questions) != QUESTION_COUNT or len(images) != IMAGE_COUNT or len(by_image) != IMAGE_COUNT:
        raise ValueError("Unexpected pinned Test-Dev row counts or duplicate image IDs")
    questions = full_questions(questions, by_image)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "images").mkdir()
    records = {}
    for image_id, source in sorted(by_image.items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError("Unsafe image ID")
        raw = source["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")
            size = list(image.size)
            pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        relative = "images/" + image_id + suffix
        (args.output_dir / relative).write_bytes(raw)
        records[image_id] = {"file": relative, "image_size": size,
            "sha256": hashlib.sha256(raw).hexdigest(), "rgb_pixels_sha256": pixels,
            "compression_seed": image_seed(args.seed, image_id)}
    manifest = {
        "schema": SCHEMA_VERSION,
        "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": "testdev",
            "tier": "testdev_balanced", "files": FILES, "total_questions": QUESTION_COUNT,
            "total_images": IMAGE_COUNT, "original_dataset": "https://cs.stanford.edu/people/dorarad/gqa/"},
        "selection": {"rule": "All labeled balanced Test-Dev questions; no sampling or text deduplication.",
            "seed": args.seed, "image_count": IMAGE_COUNT, "question_count": QUESTION_COUNT,
            "image_ids": list(records), "question_ids": [row["id"] for row in questions],
            "questions_by_image": dict(sorted(Counter(row["imageId"] for row in questions).items())),
            "type_counts": {field: dict(sorted(Counter(row["types"][field] for row in questions).items()))
                            for field in ["semantic", "structural", "detailed"]}},
        "images": records, "questions": questions,
        "prepared_by_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Full balanced Test-Dev, not the GQA training set, all unbalanced questions, or hidden test set.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_zip:
        with zipfile.ZipFile(args.output_dir.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(args.output_dir)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-gqa-cache"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--download-budget-mib", type=float, default=100)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.download_budget_mib) or args.download_budget_mib < 0:
        parser.error("Download budget must be finite and nonnegative")
    try:
        manifest = prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"GQA preparation failed: {error}\n")
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
        "images": len(manifest["images"]), "questions": len(manifest["questions"]),
        "type_counts": manifest["selection"]["type_counts"]}, indent=2))


if __name__ == "__main__":
    main()
