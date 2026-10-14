#!/usr/bin/env python3
"""Prepare provisional RQ1 datasets and image galleries; never run inference."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import tempfile
import zipfile

from rq1_candidates import (Archive, CONFIGURATIONS, EXCLUDED_GQA_IMAGES, REPO,
                            digest, rank, save_image, write_manifest)

TEXTVQA_REPO = "maoxx241/textvqa_val"
TEXTVQA_REVISION = "ddf4bd3dab5428e3ecbb4c0fc88e1c5bcfaf59cd"
TEXTVQA_ANNOTATION = "TextVQA_0.5.1_val.json"
TEXTVQA_ANNOTATION_SHA = "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d"
GQA_ARCHIVE = "research/results/original-gqa-development-20261013T221805402373Z-complete.zip"
MMSTAR_ARCHIVE = "research/results/original-mmstar-1500-20261013T182319548616Z.zip"


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def skeleton(dataset, seed, note, source):
    return {"schema": "rq1-candidates-v1", "status": "candidate", "dataset": dataset,
            "selection_seed": seed, "selection_note": note, "source": source,
            "cases": [], "images": {}, "context_questions": [],
            "freeze_status": "Awaiting human review; not a registered RQ1 evaluation set"}


def prepare_archived(path, dataset, seed, directory):
    from rq1_candidates import select_gqa, select_mmstar

    archive = Archive(path, dataset)
    try:
        picks = (select_gqa if dataset == "gqa" else select_mmstar)(archive.rows, seed)
        note = ("Half the images are selected without correctness; half enrich input-only loss/gain cases. "
                "Three questions per image; other saved questions are context only. " if dataset == "gqa" else
                "Five questions per category: three selected without correctness and two input-only diagnostic slots. ")
        manifest = skeleton(dataset, seed, note + "This mixed candidate collection is not a population accuracy sample.",
            {"archive_file": path.name, "archive_sha256": digest(path),
             "predictions_file": archive.report["predictions_file"],
             "predictions_sha256": archive.predictions_sha256,
             "recorded_prediction_hash_verified": bool(archive.report.get("predictions_sha256")),
             "dataset": archive.report["dataset"], "protocol": archive.report["protocol"],
             "implementation": archive.report["implementation"],
             "report_file": "source-report.json", "native_source_verified": True})
        directory.mkdir()
        dump(directory / "source-report.json", archive.report)
        manifest["cases"] = [archive.case(*pick) for pick in picks]
        for case in manifest["cases"]:
            if case["image_id"] in manifest["images"]:
                continue
            row = archive.rows[case["question_id"]]["baseline"]
            pixels = (archive.images[case["image_id"]]["rgb_pixels_sha256"] if dataset == "gqa"
                      else row["rgb_pixels_sha256"])
            manifest["images"][case["image_id"]] = save_image(
                archive.raw_image(case["question_id"]), directory, case["image_sha256"], pixels)
        if dataset == "gqa":
            manifest["exclusions"] = EXCLUDED_GQA_IMAGES
            manifest["reserved_evaluation_used"] = False
            # Controls are explicitly configuration-specific, not interchangeable.
            for case in manifest["cases"]:
                case["preserved_controls_by_configuration"] = {
                    cfg: [c["question_id"] for c in manifest["cases"] if c["image_id"] == case["image_id"]
                          and c["case_id"] != case["case_id"] and c["outcomes"][cfg] == "preserved_correct"]
                    for cfg in CONFIGURATIONS[1:]}
                case["same_image_preserved_controls"] = case["preserved_controls_by_configuration"]["input-only"]
            selected = {c["question_id"] for c in manifest["cases"]}
            manifest["context_questions"] = [archive.case(qid, "context_only", "Unselected saved question on a candidate image")
                for qid in sorted(archive.rows) if qid not in selected and
                archive.rows[qid]["baseline"]["image_id"] in manifest["images"]]
        manifest["observed_input_outcomes"] = dict(Counter(c["outcomes"]["input-only"] for c in manifest["cases"]))
        write_manifest(directory, manifest)
        return manifest
    finally:
        archive.close()


def select_textvqa(data, seed, count=100):
    """Select distinct images and one question per image without model outcomes."""
    grouped = defaultdict(list)
    ids = set()
    for row in data:
        qid = str(row["question_id"])
        if qid in ids or len(row["answers"]) != 10 or not row["question"].strip():
            raise ValueError("Invalid TextVQA question or reference-answer count")
        ids.add(qid)
        if not all(isinstance(a, str) and a.strip() for a in row["answers"]):
            raise ValueError("TextVQA reference answers must be nonempty strings")
        grouped[row["image_id"]].append(row)
    if len(grouped) < count:
        raise ValueError("Too few distinct TextVQA images")
    images = sorted(grouped, key=lambda image: rank(seed, "textvqa-image", image))[:count]
    return [min(grouped[image], key=lambda r: rank(seed, "textvqa-question", str(r["question_id"]))) for image in images]


def textvqa_case(row, candidate):
    counts = Counter(row["answers"])
    target = min(counts, key=lambda answer: (-counts[answer], answer))
    return {"case_id": "textvqa:" + str(row["question_id"]), "question_id": str(row["question_id"]),
            "image_id": row["image_id"], "question": row["question"],
            "generation_question": row["question"] + "\nAnswer the question using a single word or phrase.",
            "archived_prompt": None, "gold_answers": row["answers"], "canonical_target": target,
            "target_rule": "Most frequent raw reference answer; lexicographic tie-break; fixed before inference",
            "category": "text_reading", "task_metadata": {"dataset_split": row.get("set_name", "val")},
            "source_question": row, "predictions": {name: None for name in CONFIGURATIONS},
            "outcomes": {name: "unmeasured" for name in CONFIGURATIONS[1:]},
            "compression_seed": None, "baseline_screen_status": "pending",
            "selection": {"stratum": "metadata_reference", "reason":
                "First 30 of a fixed, seeded 100-image screening pool; no baseline-correct filter yet" if candidate else
                "Seeded image and question selection without model outcomes; one question per image"},
            "cautions": ["No model predictions yet. Visual grounding, legibility and baseline competence require review/screening."],
            "review": {"status": "pending", "notes": ""}}


def textvqa_files(cache, seed, max_bytes):
    """Bounded pinned downloads, cached for later rebuilds; no archive downloads."""
    from huggingface_hub import HfApi, hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / (TEXTVQA_REVISION + "-files.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["revision"] != TEXTVQA_REVISION or metadata["repo"] != TEXTVQA_REPO:
            raise ValueError("TextVQA cache identity mismatch")
    else:
        info = HfApi().dataset_info(TEXTVQA_REPO, revision=TEXTVQA_REVISION, files_metadata=True)
        if info.sha != TEXTVQA_REVISION:
            raise ValueError("TextVQA mirror revision differs")
        metadata = {"repo": TEXTVQA_REPO, "revision": info.sha, "files": {
            f.rfilename: {"bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None, "git_blob": f.blob_id}
            for f in info.siblings}}
        dump(metadata_path, metadata)

    def fetch(filename):
        entry = metadata["files"][filename]
        target = cache / "verified" / filename
        if not target.exists():
            downloaded = Path(hf_hub_download(TEXTVQA_REPO, filename, repo_type="dataset",
                                            revision=TEXTVQA_REVISION, cache_dir=str(cache / "hub")))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, target)
        raw = target.read_bytes()
        if len(raw) != entry["bytes"]:
            raise ValueError("Mirror file size mismatch: " + filename)
        if entry["sha256"]:
            actual = hashlib.sha256(raw).hexdigest()
            expected = entry["sha256"]
        else:
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            expected = entry["git_blob"]
        if actual != expected:
            raise ValueError("Pinned mirror content hash mismatch: " + filename)
        return target

    annotation_size = metadata["files"][TEXTVQA_ANNOTATION]["bytes"]
    if annotation_size > max_bytes:
        raise ValueError("Annotation exceeds download budget")
    annotation = fetch(TEXTVQA_ANNOTATION)
    if digest(annotation) != TEXTVQA_ANNOTATION_SHA:
        raise ValueError("TextVQA annotations differ from official v0.5.1 validation file")
    data = json.loads(annotation.read_text())["data"]
    if len(data) != 5000 or any(row.get("set_name") != "val" for row in data):
        raise ValueError("Expected 5,000 TextVQA validation questions")
    selected = select_textvqa(data, seed)
    filenames = ["val_images/" + row["image_id"] + ".jpg" for row in selected]
    total_bytes = annotation_size + sum(metadata["files"][name]["bytes"] for name in filenames)
    if total_bytes > max_bytes:
        raise ValueError(f"Selected source files need {total_bytes} bytes, above budget {max_bytes}")
    print(f"TextVQA: 100 individual images + annotations, {total_bytes / 1024**2:.1f} MiB source bytes", flush=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        paths = list(workers.map(fetch, filenames))
    return selected, dict(zip((row["image_id"] for row in selected), paths)), metadata, total_bytes


def prepare_textvqa(cache, seed, max_bytes, directory):
    selected, paths, metadata, total_bytes = textvqa_files(cache, seed, max_bytes)
    directory.mkdir()
    source = {"dataset": "TextVQA", "version": "0.5.1", "split": "val", "total_questions": 5000,
              "annotation_official_url": "https://dl.fbaipublicfiles.com/textvqa/data/" + TEXTVQA_ANNOTATION,
              "annotation_sha256": TEXTVQA_ANNOTATION_SHA, "mirror_repo": TEXTVQA_REPO,
              "mirror_revision": TEXTVQA_REVISION, "source_bytes": total_bytes,
              "image_verification": "Pinned mirror LFS SHA256 or Git blob hash; original bytes copied",
              "scoring_plan": {"method": "TextVQA multi-answer VQA consensus; not GQA exact match",
                               "implementation": "llava/eval/m4c_evaluator.py",
                               "implementation_sha256": digest(REPO / "llava/eval/m4c_evaluator.py")},
              "model_screen": "not_run", "auxiliary_ocr_supplied_to_model": False}
    screen = skeleton("textvqa", seed, "Fixed 100-image metadata-selected screening pool; all model outcomes pending.", source)
    for index, row in enumerate(selected):
        case = textvqa_case(row, index < 30)
        meta = save_image(paths[row["image_id"]].read_bytes(), directory)
        meta["mirror_file"] = "val_images/" + row["image_id"] + ".jpg"
        meta["mirror_identity"] = metadata["files"][meta["mirror_file"]]
        if meta["size"] != [row["image_width"], row["image_height"]]:
            case["cautions"].append("Image dimensions differ from annotation metadata; inspect before freezing.")
        case["image_sha256"] = meta["sha256"]
        screen["cases"].append(case)
        screen["images"][row["image_id"]] = meta
    dump(directory / "screening-source-questions.json", {"data": selected})
    write_manifest(directory, screen, "screening-manifest.json", "screening-gallery.html")
    candidates = skeleton("textvqa", seed, "30 provisional candidates nested in the fixed 100-image pool. "
                          "No baseline competence or compression outcome has been measured.", source)
    candidates["cases"] = screen["cases"][:30]
    candidates["images"] = {c["image_id"]: screen["images"][c["image_id"]] for c in candidates["cases"]}
    candidates["screening_pool"] = {"manifest_file": "screening-manifest.json",
                                   "manifest_sha256": digest(directory / "screening-manifest.json"), "questions": 100}
    write_manifest(directory, candidates)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa-archive", type=Path, default=REPO / GQA_ARCHIVE)
    parser.add_argument("--mmstar-archive", type=Path, default=REPO / MMSTAR_ARCHIVE)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--output-dir", type=Path, default=REPO / "research/results/rq1-candidates-v1-s590")
    parser.add_argument("--textvqa-cache", type=Path, default=Path(tempfile.gettempdir()) / "apet-rq1-textvqa-cache")
    parser.add_argument("--max-source-mib", type=int, default=100)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    bundle = output.with_suffix(".zip")
    if output.exists() or bundle.exists():
        parser.error("Output or bundle exists; use a new --output-dir to preserve earlier reviews")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rq1-preparation-", dir=output.parent) as temp:
        work = Path(temp) / "collection"
        work.mkdir()
        collections = [prepare_archived(args.gqa_archive, "gqa", args.seed, work / "gqa"),
                       prepare_archived(args.mmstar_archive, "mmstar", args.seed, work / "mmstar"),
                       prepare_textvqa(args.textvqa_cache, args.seed, args.max_source_mib * 1024**2, work / "textvqa")]
        sources = ["scripts/prepare_rq1_candidates.py", "scripts/rq1_candidates.py", "research/templates/rq1_gallery.html",
                   "research/RQ1_CANDIDATES.md", "research/provenance/original_llava_reference.json"]
        for relative in sources:
            target = work / "preparation-source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        import PIL
        collection = {"schema": "rq1-candidate-collection-v1", "status": "candidate",
            "prepared_utc": datetime.now(timezone.utc).isoformat(), "selection_seed": args.seed,
            "preparer": {"python": platform.python_version(), "pillow": PIL.__version__,
                         "source_sha256": {relative: digest(REPO / relative) for relative in sources}},
            "datasets": {m["dataset"]: {"manifest": m["dataset"] + "/manifest.json",
                "manifest_sha256": digest(work / m["dataset"] / "manifest.json"), "counts": m["counts"]} for m in collections},
            "excluded_gqa_images": EXCLUDED_GQA_IMAGES, "reserved_gqa_evaluation_used": False,
            "gpu_inference_run": False, "native_apet_modified": False}
        dump(work / "collection.json", collection)
        (work / "index.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RQ1 candidate review</title>
<style>body{max-width:850px;margin:40px auto;padding:0 20px;font:18px/1.6 system-ui;color:#17202a}a{color:#174d86}li{margin:16px 0}</style>
<h1>RQ1 candidate review</h1><p>Provisional selections for all three datasets. Review images, questions,
reference answers and saved predictions; download your decisions from each gallery. Labels and original results stay unchanged.</p>
<ul><li><a href="gqa/gallery.html">GQA: 60 questions / 20 images</a> — same-image controls and other saved questions.</li>
<li><a href="mmstar/gallery.html">MMStar: 30 questions / 30 images</a> — five per category.</li>
<li><a href="textvqa/gallery.html">TextVQA: 30 provisional questions / 30 images</a> — awaiting baseline screening.</li>
<li><a href="textvqa/screening-gallery.html">TextVQA: fixed 100-question screening pool</a> — includes the 30 candidates.</li></ul>
<p>Review exports are separate overlays, not frozen datasets. Numerical debugging remains deferred;
GQA image n497658 is excluded. These selections support mechanism analysis, not population accuracy estimates.</p>
<p><a href="collection.json">Collection provenance</a> · <a href="preparation-source/research/RQ1_CANDIDATES.md">Instructions</a></p></html>
""", encoding="utf-8")
        work.rename(output)
    with zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(json.dumps({"output": str(output), "bundle": str(bundle), "datasets": collection["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
