"""Package accepted working inputs for Kaggle; no new selection or downloads."""

import argparse
from pathlib import Path
import shutil
import zipfile

from rq1_protocol import GQA_IDS, REPO, SOURCES, VERSION, load_manifest, sha256, write_json


def prepare(results, output):
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise ValueError("Choose a new output path; existing inputs are preserved")
    # Validate all sources before copying anything.
    sources = [(name, results / rel, load_manifest(results / rel, name))
               for name, (rel, _) in SOURCES.items()]
    output.mkdir(parents=True)
    counts = {}
    for name, source, data in sources:
        destination = output / name
        destination.mkdir()
        shutil.copy2(source, destination / "manifest.json")
        for meta in data["images"].values():
            target = destination / meta["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.parent / meta["file"], target)
        counts[name] = {"questions": len(data["cases"]), "images": len(data["images"]),
                        "manifest_sha256": sha256(source)}
    write_json(output / "protocol.json", {
        "version": VERSION, "gqa_pilot_question_ids": GQA_IDS,
        "calibration": {"questions": 2, "uniform_sampled_tokens_per_image": 32},
        "pilot": {"questions": 12, "tokens_per_image": 576},
        "inputs": counts, "textvqa": "All 100 retained; baseline/ApET generation only at this stage",
        "note": "Accepted manifests copied byte-for-byte; this does not revise dataset curation.",
    })
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(output))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO / "research/results")
    parser.add_argument("--output", type=Path, default=REPO / "research/results/rq1-inputs-v1")
    args = parser.parse_args()
    archive = prepare(args.results, args.output)
    print(f"Prepared {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")
