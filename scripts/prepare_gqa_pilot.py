"""Prepare a small, reproducible same-image GQA subset on CPU; no model imports."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import zipfile

DATASET_ID = "lmms-lab-encoder/GQA"
DATASET_REVISION = "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
FILES = {
    "testdev_balanced_instructions/testdev-00000-of-00001.parquet": {
        "bytes": 2090335, "sha256": "e967085f896deee3b99b7c7a830e74fc0662a61cdbcbd0154ffbb8aea4ab67ed"},
    "testdev_balanced_images/testdev-00000-of-00001.parquet": {
        "bytes": 65647745, "sha256": "ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb"},
}
SCHEMA_VERSION = "gqa-grouped-pilot-v1"


def image_seed(seed, image_id):
    digest = hashlib.sha256(f"gqa:{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def fetch_sources(cache_dir, source_dir=None, download_budget_mib=100):
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    paths = {}
    for name in FILES:
        if source_dir:
            paths[name] = Path(source_dir) / name
            if not paths[name].is_file():
                raise ValueError(f"Missing local source parquet: {paths[name]}")
        else:
            cached = try_to_load_from_cache(DATASET_ID, name, repo_type="dataset",
                                            revision=DATASET_REVISION, cache_dir=str(cache_dir))
            if isinstance(cached, str) and Path(cached).is_file():
                paths[name] = Path(cached)
    missing = sum(FILES[name]["bytes"] for name in FILES if name not in paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if missing > download_budget_mib * 1024 ** 2 or missing + 50 * 1024 ** 2 > shutil.disk_usage(cache_dir).free:
        raise ValueError("GQA source downloads exceed the budget or free space")
    for name, expected in FILES.items():
        if name not in paths:
            paths[name] = Path(hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=DATASET_REVISION,
                                               cache_dir=str(cache_dir)))
        if paths[name].stat().st_size != expected["bytes"] or hashlib.sha256(paths[name].read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Pinned GQA source hash/size mismatch: {name}")
    return paths


def select_groups(questions, available_images, image_count, questions_per_image, seed):
    groups = defaultdict(list)
    seen = set()
    for row in questions:
        required = {"id", "imageId", "question", "answer", "types", "isBalanced"}
        if not required.issubset(row) or not isinstance(row["types"], dict):
            raise ValueError("Unexpected GQA question schema")
        if row["id"] in seen:
            raise ValueError("Duplicate GQA question IDs")
        seen.add(row["id"])
        if not row["isBalanced"] or not row["answer"].strip() or not row["question"].strip():
            raise ValueError("Expected labeled balanced GQA questions")
        if row["imageId"] not in available_images:
            raise ValueError(f"Question image missing from source: {row['imageId']}")
        groups[row["imageId"]].append(row)
    candidates = {}
    for image_id, rows in groups.items():
        unique = {}
        for row in sorted(rows, key=lambda row: row["id"]):
            unique.setdefault(" ".join(row["question"].split()).casefold(), row)
        rows = list(unique.values())
        structural = {row["types"]["structural"] for row in rows}
        semantic = {row["types"]["semantic"] for row in rows}
        if len(rows) >= questions_per_image and {"query", "verify"} <= structural and len(semantic) >= 2:
            candidates[image_id] = rows
    if len(candidates) < image_count:
        raise ValueError(f"Need {image_count} eligible image groups; found {len(candidates)}")
    selected_ids = sorted(random.Random(seed).sample(sorted(candidates), image_count))
    selected = []
    for image_id in selected_ids:
        rows = candidates[image_id][:]
        random.Random(image_seed(seed, image_id)).shuffle(rows)
        chosen = [next(row for row in rows if row["types"]["structural"] == kind) for kind in ["query", "verify"]]
        while len(chosen) < questions_per_image:
            def novelty(row):
                return tuple(row["types"][field] not in {item["types"][field] for item in chosen}
                             for field in ["semantic", "structural", "detailed"])
            chosen.append(max((row for row in rows if row not in chosen), key=novelty))
        selected.extend(sorted(chosen, key=lambda row: row["id"]))
    return selected, len(candidates)


def prepare(args):
    import pyarrow.parquet as pq
    from PIL import Image

    if args.output_dir.exists() or args.output_dir.with_suffix(".zip").exists():
        raise ValueError("Output exists; choose a new directory to preserve earlier data")
    paths = fetch_sources(args.dataset_cache, args.source_dir, args.download_budget_mib)
    questions = pq.read_table(paths[next(iter(FILES))]).to_pylist()
    images = pq.read_table(paths[list(FILES)[1]]).to_pylist()
    if len(questions) != 12578 or len(images) != 398:
        raise ValueError("Unexpected row counts for pinned GQA Test-Dev sources")
    by_image = {row["id"]: row["image"] for row in images}
    if len(by_image) != len(images):
        raise ValueError("Duplicate GQA image IDs")
    selected, eligible = select_groups(questions, by_image, args.image_count, args.questions_per_image, args.seed)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "images").mkdir()
    records = {}
    for image_id in sorted({row["imageId"] for row in selected}):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError("Unexpected image ID for a portable image filename")
        raw = by_image[image_id]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")
            size = list(image.size)
            pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        target = args.output_dir / "images" / (image_id + extension)
        target.write_bytes(raw)
        records[image_id] = {"file": str(target.relative_to(args.output_dir)), "image_size": size,
                             "sha256": hashlib.sha256(raw).hexdigest(), "rgb_pixels_sha256": pixels,
                             "compression_seed": image_seed(args.seed, image_id)}
    manifest = {
        "schema": SCHEMA_VERSION,
        "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": "testdev",
                   "tier": "testdev_balanced", "files": FILES,
                   "original_dataset": "https://cs.stanford.edu/people/dorarad/gqa/",
                   "total_questions": len(questions), "total_images": len(images)},
        "selection": {"seed": args.seed, "image_count": args.image_count,
                      "questions_per_image": args.questions_per_image, "eligible_images": eligible,
                      "image_ids": list(records), "question_ids": [row["id"] for row in selected],
                      "rule": "Seeded image sampling among groups with query/verify and >=2 semantic types; "
                              "unique questions chosen for type diversity without consulting model predictions.",
                      "type_counts": {field: dict(Counter(row["types"][field] for row in selected))
                                      for field in ["semantic", "structural", "detailed"]}},
        "images": records, "questions": selected,
        "prepared_by_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Diagnostic subset, not a population benchmark. Original question metadata and image bytes retained.",
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_zip:
        with zipfile.ZipFile(args.output_dir.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(args.output_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, str(file.relative_to(args.output_dir)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--questions-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-gqa-cache"))
    parser.add_argument("--source-dir", type=Path, help="Optional directory containing the two pinned parquet subdirectories")
    parser.add_argument("--download-budget-mib", type=float, default=100)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if args.image_count < 1 or args.questions_per_image < 2:
        parser.error("Use positive image count and at least two questions per image")
    if not math.isfinite(args.download_budget_mib) or args.download_budget_mib < 0:
        parser.error("Download budget must be finite and nonnegative")
    try:
        manifest = prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"GQA preparation failed: {error}\n")
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "selection": manifest["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
