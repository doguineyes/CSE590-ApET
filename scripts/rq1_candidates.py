"""CPU-only selection and review rendering for provisional RQ1 collections."""

from collections import Counter, defaultdict
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("baseline", "input-only", "decoder-only", "combined")
EXCLUDED_GQA_IMAGES = {"n497658": "Deferred airplane repeatability investigation; exclude every question on this image."}
KNOWN_CAUTIONS = {
    "20119066": "Previous image review: Tie and Suit both describe visible clothing; scored gain may reflect specificity.",
    "201570646": "Previous image review: cabinet has glass doors and a dark frame/base; intended material is ambiguous.",
    "201713417": "Previous image review: both a cup and mirrors are in front of the woman; reference is ambiguous.",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rank(seed, namespace, value):
    return hashlib.sha256(f"rq1:{seed}:{namespace}:{value}".encode()).hexdigest()


def outcome(before, after):
    if before is None or after is None:
        return "unmeasured"
    return "preserved_correct" if before and after else "loss" if before else "gain" if after else "both_wrong"


class Archive:
    def __init__(self, path, dataset):
        self.path, self.dataset = Path(path), dataset
        self.zip = zipfile.ZipFile(path)
        try:
            names = self.zip.namelist()
            if len(names) != len(set(names)) or self.zip.testzip() is not None:
                raise ValueError("Duplicate members or failed ZIP CRC")
            reports = [json.loads(self.zip.read(n)) for n in names if "/" not in n and n.endswith(".json")]
            self.report = next(r for r in reports if "predictions_file" in r)
            report = self.report
            if report["status"] != "passed" or not report["implementation"]["native_matches_reference"]:
                raise ValueError("Use a completed native ApET reference archive")
            model = report["sessions"][0]["model"] if "sessions" in report else report["model"]
            if model["class"] != "LlavaLlamaForCausalLM_X":
                raise ValueError("Expected the original native LLaVA baseline")
            reference = json.loads((REPO / "research/provenance/original_llava_reference.json").read_text())
            for name, expected in reference["native_source_sha256"].items():
                candidates = [n for n in names if n.endswith("/" + name)]
                if len(candidates) != 1 or hashlib.sha256(self.zip.read(candidates[0])).hexdigest() != expected:
                    raise ValueError("Native source snapshot mismatch: " + name)
            raw = self.zip.read(report["predictions_file"])
            self.predictions_sha256 = hashlib.sha256(raw).hexdigest()
            if report.get("predictions_sha256") and self.predictions_sha256 != report["predictions_sha256"]:
                raise ValueError("Prediction hash mismatch")
            self.rows = {}
            for row in [json.loads(line) for line in raw.decode().splitlines()]:
                qid = str(row.get("question_id", row["index"]))
                config = row["configuration"]
                if config not in CONFIGURATIONS or config in self.rows.setdefault(qid, {}) or row["status"] != "passed":
                    raise ValueError("Invalid/duplicate question-configuration record")
                self.rows[qid][config] = row
            expected_count = 800 if dataset == "gqa" else 1500
            if len(self.rows) != expected_count or any(set(v) != set(CONFIGURATIONS) for v in self.rows.values()):
                raise ValueError("Expected a complete four-configuration development baseline")
            for records in self.rows.values():
                baseline = records["baseline"]
                for row in records.values():
                    if any(row[k] != baseline[k] for k in ["question", "gold", "category", "seed", "image_sha256"]):
                        raise ValueError("Paired identities differ")
            self.source_questions = {}
            self.images = {}
            if dataset == "gqa":
                if report["dataset"]["study"]["role"] != "development":
                    raise ValueError("Reserved GQA evaluation cannot be selected")
                raw = self.zip.read(report["dataset"]["manifest_file"])
                if hashlib.sha256(raw).hexdigest() != report["dataset"]["manifest_sha256"]:
                    raise ValueError("Development manifest changed")
                manifest = json.loads(raw)
                self.source_questions = {r["id"]: r for r in manifest["questions"]}
                self.images = manifest["images"]
        except BaseException:
            self.zip.close()
            raise

    def close(self):
        self.zip.close()

    def raw_image(self, qid):
        row = self.rows[qid]["baseline"]
        raw = self.zip.read(row["image"])
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError("Archived image hash differs: " + qid)
        return raw

    def case(self, qid, stratum, reason):
        records = self.rows[qid]
        baseline = records["baseline"]
        image_id = baseline["image_id"] if self.dataset == "gqa" else baseline["image_sha256"]
        outcomes = {name: outcome(baseline["correct"], row["correct"]) for name, row in records.items() if name != "baseline"}
        predictions = {name: {k: row[k] for k in ["response", "prediction", "correct", "scoring_version",
                                                 "seed", "settings", "generated_tokens"]} for name, row in records.items()}
        flags = []
        if self.dataset == "gqa" and qid in KNOWN_CAUTIONS:
            flags.append(KNOWN_CAUTIONS[qid])
        if any(row["generated_tokens"] >= self.report["protocol"]["max_new_tokens"] for row in records.values()):
            flags.append("At least one archived answer reached the output-token cap.")
        if self.dataset == "mmstar" and any(not row["format_compliant"] for row in records.values()):
            flags.append("At least one answer required the explicit-choice parser rather than a single option letter.")
        return {"case_id": f"{self.dataset}:{qid}", "question_id": qid, "image_id": image_id,
                "question": baseline["original_question"], "generation_question": baseline["question"],
                "archived_prompt": baseline["prompt"], "gold_answers": [baseline["gold"]],
                "canonical_target": baseline["gold"], "target_rule": "Unchanged archived gold label",
                "category": baseline["category"], "task_metadata": baseline.get("types", {"l2_category": baseline.get("l2_category")}),
                "source_question": self.source_questions.get(qid), "predictions": predictions,
                "outcomes": outcomes, "compression_seed": baseline["seed"],
                "selection": {"stratum": stratum, "reason": reason}, "cautions": flags,
                "review": {"status": "pending", "notes": ""}, "image_sha256": baseline["image_sha256"]}


def choose_diverse(rows, selected, count, seed, namespace):
    selected = list(selected)
    ordered = sorted(rows, key=lambda row: rank(seed, namespace, row["question_id"]))
    while len(selected) < count:
        remaining = [r for r in ordered if r["question_id"] not in {x["question_id"] for x in selected}
                     and r["original_question"] not in {x["original_question"] for x in selected}]
        if not remaining:
            raise ValueError("Not enough distinct questions for the requested image group")
        def novelty(row):
            return tuple(row.get("types", {}).get(field) not in {r.get("types", {}).get(field) for r in selected}
                         for field in ["semantic", "structural", "detailed"])
        selected.append(max(remaining, key=novelty))
    return selected


def select_gqa(rows, seed):
    groups = defaultdict(list)
    for qid, predictions in rows.items():
        row = predictions["baseline"]
        if row["image_id"] not in EXCLUDED_GQA_IMAGES:
            groups[row["image_id"]].append({**row, "question_id": qid})
    eligible = {i: group for i, group in groups.items() if len({r["original_question"] for r in group}) >= 3}
    if len(eligible) < 20:
        raise ValueError("Need 20 eligible development images")
    reference_images = sorted(eligible, key=lambda i: rank(seed, "gqa-reference-image", i))[:10]
    chosen = [(i, "metadata_reference", "Image chosen by seeded hash without correctness", None) for i in reference_images]

    def ids(image, role):
        return [r["question_id"] for r in eligible[image] if
                outcome(rows[r["question_id"]]["baseline"]["correct"], rows[r["question_id"]]["input-only"]["correct"]) == role
                and (role not in {"loss", "gain"} or r["question_id"] not in KNOWN_CAUTIONS)]
    for role, count in [("mixed", 4), ("loss", 3), ("gain", 2), ("preserved", 1)]:
        for _ in range(count):
            unused = [i for i in eligible if i not in {x[0] for x in chosen}]
            candidates = [i for i in unused if ids(i, "preserved_correct") and
                          (ids(i, "loss") and ids(i, "gain") if role == "mixed" else
                           ids(i, "preserved_correct") if role == "preserved" else ids(i, role))]
            fallback = not candidates
            if fallback:
                candidates = unused
            image = min(candidates, key=lambda i: rank(seed, "gqa-diagnostic-image:" + role, i))
            chosen.append((image, "outcome_diagnostic", f"Requested {role} image" + ("; unavailable, metadata fallback" if fallback else ""), role))
    selections = []
    for image, stratum, reason, role in chosen:
        group = eligible[image]
        initial = []
        if stratum == "metadata_reference":
            for kind in ["query", "verify"]:
                pool = [r for r in group if r["types"]["structural"] == kind]
                if pool:
                    initial.append(min(pool, key=lambda r: rank(seed, "gqa-reference-question", r["question_id"])))
        else:
            requested = ["loss", "gain", "preserved_correct"] if role == "mixed" else [role, "preserved_correct"]
            for target in requested:
                pool = [r for r in group if r["question_id"] in ids(image, target)
                        and r["question_id"] not in {x["question_id"] for x in initial}]
                if pool:
                    initial.append(min(pool, key=lambda r: rank(seed, "gqa-diagnostic-question", r["question_id"])))
        picked = choose_diverse(group, initial, 3, seed, "gqa-diverse-question")
        selections.extend((r["question_id"], stratum, reason) for r in picked)
    return selections


def select_mmstar(rows, seed):
    selections = []
    categories = sorted({records["baseline"]["category"] for records in rows.values()})
    if len(categories) != 6:
        raise ValueError("Expected six MMStar categories")
    for category in categories:
        pool = [qid for qid, records in rows.items() if records["baseline"]["category"] == category]
        reference = sorted(pool, key=lambda qid: rank(seed, "mmstar-reference", qid))[:3]
        selections.extend((qid, "metadata_reference", "Seeded category-balanced choice without correctness") for qid in reference)
        used = set(reference)
        for role in ["loss", "gain"]:
            candidates = [qid for qid in pool if qid not in used and
                          outcome(rows[qid]["baseline"]["correct"], rows[qid]["input-only"]["correct"]) == role
                          and all(row["generated_tokens"] < 16 for row in rows[qid].values())]
            reason = "Input-only scored " + role
            if not candidates:
                candidates = [qid for qid in pool if qid not in used and rows[qid]["baseline"]["correct"]
                              and rows[qid]["input-only"]["correct"]]
                reason += " unavailable after reference selection; preserved-correct fallback"
            if not candidates:
                candidates = [qid for qid in pool if qid not in used]
                reason += "; metadata fallback"
            qid = min(candidates, key=lambda qid: rank(seed, "mmstar-diagnostic:" + role, qid))
            used.add(qid)
            selections.append((qid, "outcome_diagnostic", reason))
    return selections


def save_image(raw, directory, expected_sha=None, expected_pixels=None):
    from PIL import Image

    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and sha != expected_sha:
        raise ValueError("Image bytes differ from reference")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if expected_pixels and pixels != expected_pixels:
            raise ValueError("Decoded pixels differ from reference")
        extension, mime = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"),
                           "WEBP": (".webp", "image/webp")}.get(image.format, (".img", "application/octet-stream"))
        size = list(image.size)
    relative = Path("images") / (sha + extension)
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(raw)
    elif digest(target) != sha:
        raise ValueError("Existing copied image differs")
    return {"file": str(relative), "sha256": sha, "rgb_pixels_sha256": pixels, "size": size, "mime": mime}


def validate_manifest(manifest, directory):
    cases = manifest["cases"]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)) or manifest["status"] != "candidate":
        raise ValueError("Duplicate candidate IDs or incorrect collection state")
    if manifest["dataset"] == "gqa" and (set(manifest["images"]) & set(EXCLUDED_GQA_IMAGES) or
            any(c["image_id"] in EXCLUDED_GQA_IMAGES for c in cases + manifest.get("context_questions", []))):
        raise ValueError("Deferred GQA image leaked into candidates")
    for case in cases:
        if case["review"]["status"] != "pending" or not case["canonical_target"] or not case["gold_answers"]:
            raise ValueError("Candidates need pending review and fixed reference answers")
        meta = manifest["images"][case["image_id"]]
        relative = Path(meta["file"])
        if (relative.is_absolute() or ".." in relative.parts or digest(directory / relative) != meta["sha256"]
                or case["image_sha256"] != meta["sha256"]):
            raise ValueError("Candidate image integrity failed")


def write_manifest(directory, manifest, filename="manifest.json", gallery="gallery.html"):
    manifest["counts"] = {"questions": len(manifest["cases"]), "images": len(manifest["images"]),
        "categories": dict(Counter(c["category"] for c in manifest["cases"])),
        "strata": dict(Counter(c["selection"]["stratum"] for c in manifest["cases"]))}
    validate_manifest(manifest, directory)
    path = directory / filename
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    render_gallery(directory, manifest, digest(path), gallery, filename)
    return path


def render_gallery(directory, manifest, manifest_sha, filename, manifest_filename):
    import base64

    esc = html.escape
    grouped = defaultdict(list)
    for case in manifest["cases"]:
        grouped[case["image_id"]].append(case)
    content = []

    def card(case, reviewable=True):
        role = case["outcomes"].get("input-only", "unmeasured")
        badge = "Candidate" if reviewable else "Context only"
        predictions = "".join(f"<tr><th>{esc(name)}</th><td>{esc(str((case['predictions'].get(name) or {}).get('response', 'Not measured')))}</td>"
                              f"<td>{esc(str((case['predictions'].get(name) or {}).get('correct', 'Pending')))}</td></tr>" for name in CONFIGURATIONS)
        references = ", ".join(f"{answer} × {count}" for answer, count in Counter(case["gold_answers"]).items())
        cautions = "".join(f"<p class='caution'>{esc(reason)}</p>" for reason in case["cautions"])
        controls = ", ".join(case.get("same_image_preserved_controls", [])) or "None among selected candidates"
        controls_html = f"<p class='small'>Selected input-only preserved controls: {esc(controls)}</p>" if manifest["dataset"] == "gqa" else ""
        form = (f"<div class='review' data-reviewcase='{esc(case['case_id'], quote=True)}'><label>Review "
                "<select><option value='pending'>Pending</option><option value='keep'>Keep</option>"
                "<option value='exclude'>Exclude</option><option value='ambiguous'>Ambiguous</option>"
                "<option value='annotation_issue'>Possible label issue</option><option value='uncertain'>Uncertain</option></select></label>"
                "<textarea rows='2' placeholder='Reason or observation'></textarea></div>") if reviewable else ""
        return (f"<section class='question'><div class='tags'>{badge} · {esc(case['question_id'])} · {esc(case['category'])} · "
                f"<span class='{role}'>{esc(role.replace('_', ' '))}</span></div><p class='q'>{esc(case['question'])}</p>"
                f"<p>Reference answers: {esc(references)}<br>Fixed target: <strong>{esc(case['canonical_target'])}</strong></p>"
                f"<p class='small'>{esc(case['selection']['stratum'])}: {esc(case['selection']['reason'])}</p>"
                f"{cautions}{controls_html}<table><thead><tr><th>Configuration</th><th>Raw answer</th><th>Original score</th></tr></thead>"
                f"<tbody>{predictions}</tbody></table>{form}</section>")

    for image_id, cases in grouped.items():
        meta = manifest["images"][image_id]
        encoded = base64.b64encode((directory / meta["file"]).read_bytes()).decode()
        selected_ids = {c["case_id"] for c in cases}
        context = [c for c in manifest.get("context_questions", []) if c["image_id"] == image_id and c["case_id"] not in selected_ids]
        others = "<details><summary>Other saved questions on this image (context only)</summary>" + "".join(card(c, False) for c in context) + "</details>" if context else ""
        content.append(f"<article><div class='picture'><img src='data:{meta['mime']};base64,{encoded}' alt='Dataset image {esc(image_id, quote=True)}'>"
                       f"<p>{esc(image_id)} · {meta['size'][0]} × {meta['size'][1]}</p></div>"
                       f"<div>{''.join(card(c) for c in cases)}{others}</div></article>")
    template = (REPO / "research/templates/rq1_gallery.html").read_text()
    replacements = {"__TITLE__": esc(manifest["dataset"].upper() + " RQ1 candidate review"),
        "__SUMMARY__": esc(f"{len(manifest['cases'])} candidate questions · {len(grouped)} images · selection seed {manifest['selection_seed']}"),
        "__NOTE__": esc(manifest["selection_note"]), "__BODY__": "\n".join(content),
        "__MANIFEST_FILE__": esc(manifest_filename, quote=True),
        "__REVIEW_IDENTITY__": json.dumps({"schema": "rq1-review-overlay-v1", "dataset": manifest["dataset"],
                                           "manifest_sha256": manifest_sha}).replace("<", "\\u003c")}
    for key, value in replacements.items():
        template = template.replace(key, value)
    (directory / filename).write_text(template)
