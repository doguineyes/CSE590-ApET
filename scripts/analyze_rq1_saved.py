"""CPU-only group and same-image analysis of frozen RQ1 input-stage results.

Python standard library only; reads original exports and writes a new directory.
Existing PNGs are reused, so no model or plotting packages are needed locally.
"""

import argparse
from collections import defaultdict
import html
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import statistics

from rq1_protocol import average_ranks, sha256, spearman, write_json
from summarize_rq1_pilot import summarize, write_csv

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "research/protocols/rq1_saved_analysis_v1.json"


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def error_groups(tokens, basis, bins=3):
    """Rank groups exclude basis sources and never split equal residuals."""
    remaining = [t for t in tokens if t["token_index"] not in basis]
    ranks = average_ranks([t["reconstruction_error"] for t in remaining])
    groups = [[] for _ in range(bins)]
    for token, rank in zip(remaining, ranks):
        # Ranks are one-based; a rank exactly on a boundary goes to the lower bin.
        group = min(bins - 1, math.ceil(rank * bins / len(remaining)) - 1)
        groups[group].append(token)
    return groups, [t for t in tokens if t["token_index"] in basis]


def distribution(tokens):
    if not tokens:
        return {"n": 0, **{key: None for key in (
            "E_min", "E_max", "I_mean", "I_median", "I_q25", "I_q75",
            "I_min", "I_max", "positive_fraction", "abs_I_mean_auxiliary")}}
    effects = [t["intervention_score"] for t in tokens]
    return {"n": len(tokens), "E_min": min(t["reconstruction_error"] for t in tokens),
            "E_max": max(t["reconstruction_error"] for t in tokens),
            "I_mean": statistics.mean(effects), "I_median": statistics.median(effects),
            "I_q25": quantile(effects, .25), "I_q75": quantile(effects, .75),
            "I_min": min(effects), "I_max": max(effects),
            "positive_fraction": sum(x > 0 for x in effects) / len(effects),
            "abs_I_mean_auxiliary": statistics.mean(abs(x) for x in effects)}


def contrast(tokens, basis, bins=3):
    groups, basis_tokens = error_groups(tokens, basis, bins)
    rows = [distribution(group) for group in groups]
    delta = None if not rows[0]["n"] or not rows[-1]["n"] else rows[-1]["I_mean"] - rows[0]["I_mean"]
    median_delta = None if delta is None else rows[-1]["I_median"] - rows[0]["I_median"]
    return rows, distribution(basis_tokens), delta, median_delta


def cohort(rows):
    valid = [r for r in rows if r["high_minus_low_mean_I"] is not None]
    images = defaultdict(list)
    for row in valid:
        images[row["image_id"]].append(row["high_minus_low_mean_I"])
    values = [r["high_minus_low_mean_I"] for r in valid]
    return {"questions": len(rows), "images": len({r["image_id"] for r in rows}),
            "questions_with_defined_contrast": len(valid),
            "high_mean_exceeds_low_questions": sum(x > 0 for x in values),
            "median_question_high_minus_low_mean_I": statistics.median(values) if values else None,
            "median_question_high_minus_low_median_I": statistics.median(r["high_minus_low_median_I"] for r in valid) if valid else None,
            "median_image_mean_high_minus_low_I": statistics.median(statistics.mean(v) for v in images.values()) if images else None,
            "median_question_nonbasis_rho": statistics.median(r["rho_nonbasis"] for r in rows),
            "median_quartile_high_minus_low_mean_I": statistics.median(r["quartile_high_minus_low_mean_I"] for r in valid) if valid else None}


def pair_summary(a, b):
    for field in ("image_id", "image_sha256", "seed", "clip_sha256", "projected_sha256"):
        if a["case"][field] != b["case"][field]:
            raise ValueError(f"Same-image pair differs in {field}")
    if a["native"] != b["native"]:
        raise ValueError("Same-image native reconstruction/merge records differ")
    case_a, case_b = a["case"], b["case"]
    first = {t["token_index"]: t for t in case_a["tokens"]}
    second = {t["token_index"]: t for t in case_b["tokens"]}
    if set(first) != set(second) or any(first[i]["reconstruction_error"] != second[i]["reconstruction_error"] for i in first):
        raise ValueError("Same-image E or measured indices differ")
    ids = sorted(first)
    nb = [i for i in ids if i not in a["basis"]]
    top_count = math.ceil(len(nb) * .10)
    def top(lookup):
        ordered = sorted(nb, key=lambda i: (-lookup[i]["intervention_score"], i))
        cutoff = lookup[ordered[top_count - 1]]["intervention_score"]
        return ordered[:top_count], sum(lookup[i]["intervention_score"] == cutoff for i in nb)
    ta, ties_a = top(first); tb, ties_b = top(second)
    overlap = len(set(ta) & set(tb))
    row = {"image_id": case_a["image_id"], "question_a": case_a["case_id"], "question_b": case_b["case_id"],
           "target_a": case_a["measurement_target"], "target_b": case_b["measurement_target"],
           "reference_mean_logprob_a": case_a["controls"]["reference"]["mean_logprob"],
           "reference_mean_logprob_b": case_b["controls"]["reference"]["mean_logprob"],
           "answer_tokens_a": len(case_a["target_token_ids"]), "answer_tokens_b": len(case_b["target_token_ids"]),
           "rho_all": spearman([first[i]["intervention_score"] for i in ids], [second[i]["intervention_score"] for i in ids]),
           "rho_nonbasis": spearman([first[i]["intervention_score"] for i in nb], [second[i]["intervention_score"] for i in nb]),
           "nonbasis_tokens": len(nb), "top_set_size": top_count, "top_overlap": overlap,
           "top_jaccard": overlap / (2 * top_count - overlap),
           "top_boundary_ties_a": ties_a, "top_boundary_ties_b": ties_b,
           "opposite_sign_beyond_1e_minus4": sum(first[i]["intervention_score"] * second[i]["intervention_score"] < 0
               and min(abs(first[i]["intervention_score"]), abs(second[i]["intervention_score"])) > 1e-4 for i in nb)}
    ranks_a = dict(zip(nb, average_ranks([first[i]["intervention_score"] for i in nb])))
    ranks_b = dict(zip(nb, average_ranks([second[i]["intervention_score"] for i in nb])))
    tokens = [{"image_id": row["image_id"], "question_a": row["question_a"], "question_b": row["question_b"],
               "token_index": i, "E": first[i]["reconstruction_error"],
               "I_a": first[i]["intervention_score"], "I_b": second[i]["intervention_score"],
               "basis": i in a["basis"], "direct_original_retained": i in a["retained"],
               "I_rank_a_nonbasis": ranks_a.get(i), "I_rank_b_nonbasis": ranks_b.get(i),
               "top_a": i in ta, "top_b": i in tb} for i in ids]
    return row, tokens


def analyze(results, output, plan_path=PLAN):
    results, output = Path(results).resolve(), Path(output).resolve()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if output.exists():
        raise ValueError("Choose a new output directory; original and earlier analyses are preserved")
    for info in plan["sources"].values():
        source = results / info["run"]
        if source == output or source in output.parents:
            raise ValueError("Analysis output must be outside each original run")
        if sha256(source.with_suffix(".zip")) != info["archive_sha256"]:
            raise ValueError(f"Original ZIP hash differs: {source.name}")
    output.mkdir(parents=True)
    records, source_records = [], []
    for dataset, info in plan["sources"].items():
        source = results / info["run"]
        checked = summarize(source, output / "validation" / dataset)
        report = json.loads((source / "report.json").read_text())
        if report["version"] != "rq1-input-zero-v2" or report["dataset"]["name"] != dataset:
            raise ValueError("Unexpected input protocol/dataset")
        manifest = json.loads((source / "evidence/manifest.json").read_text())
        lookup = {c["case_id"]: c for c in manifest["cases"]}
        for p in sorted((source / "questions").glob("*.json")):
            case = json.loads(p.read_text())
            if {t["token_index"] for t in case["tokens"]} != set(range(576)):
                raise ValueError("This analysis requires all 576 patches")
            feature_path = source / "features" / (case["image_sha256"] + ".json")
            feature = json.loads(feature_path.read_text())
            if (case["clip_sha256"], case["projected_sha256"], case["seed"]) != (feature["clip_sha256"], feature["projected_sha256"], feature["seed"]):
                raise ValueError("Question features/seed differ from cached record")
            encoded = next(p for p in (source / "images").glob(case["image_sha256"] + ".*") if p.suffix in (".jpg", ".png", ".jpeg"))
            if sha256(encoded) != case["image_sha256"]:
                raise ValueError("Saved encoded image hash differs")
            native = feature["diagnostics"]
            if not case["free_generation"]["scoring"]["correct"]:
                raise ValueError("This frozen panel expects a correct fresh reference answer")
            records.append({"dataset": dataset, "source": source, "case": case, "native": native,
                            "basis": set(native["input"]["basis_indices"]),
                            "retained": set(native["input_merge"]["target_indices"]),
                            "category": lookup[case["case_id"]].get("category", "text reading"),
                            "feature_metadata_sha256": sha256(feature_path)})
        source_records.append({"dataset": dataset, "run": info["run"],
            "archive_sha256": info["archive_sha256"], "report_sha256": checked["source_report_sha256"],
            "source_commit": report["commit"], "original_lock_sha256": report["lock_sha256"]})
    if len({r["case"]["case_id"] for r in records}) != len(records):
        raise ValueError("Duplicate question across canonical runs")
    groups, questions, token_rows = [], [], []
    for record in records:
        case, basis = record["case"], record["basis"]
        bins, fps, delta, median_delta = contrast(case["tokens"], basis)
        _, _, quartile_delta, _ = contrast(case["tokens"], basis, 4)
        nonbasis = [t for t in case["tokens"] if t["token_index"] not in basis]
        rho = spearman([t["reconstruction_error"] for t in nonbasis], [t["intervention_score"] for t in nonbasis])
        common = {"dataset": record["dataset"], "case_id": case["case_id"], "image_id": case["image_id"],
                  "task": record["category"], "archived_role": case["outcomes"]["input-only"]}
        reference = case["controls"]["reference"]["token_logprobs"]
        control_delta = max(abs(a-b) for key in ("reference_repeat", "noop", "native_multimodal", "reference_after")
            for a,b in zip(reference, case["controls"][key]["token_logprobs"]))
        questions.append({**common, "question": case["question"], "target": case["measurement_target"],
            "rho_nonbasis": rho, "high_minus_low_mean_I": delta, "high_minus_low_median_I": median_delta,
            "quartile_high_minus_low_mean_I": quartile_delta,
            "target_tokens": len(case["target_token_ids"]), "reference_mean_logprob": case["controls"]["reference"]["mean_logprob"],
            "full_forward_control_delta": control_delta,
            "cached_generation_prefix_delta_diagnostic": case["controls"]["generated_prefix_score_max_delta"],
            "question_record_sha256": sha256(record["source"] / "questions" / (case["question_id"] + ".json")),
            "feature_metadata_sha256": record["feature_metadata_sha256"]})
        for name, stats in zip(["low", "middle", "high", "fps_basis"], bins + [fps]):
            groups.append({**common, "group": name, **stats})
        partition, _ = error_groups(case["tokens"], basis)
        group_of = {t["token_index"]: name for name, items in zip(["low", "middle", "high"], partition) for t in items}
        ranks = dict(zip([t["token_index"] for t in nonbasis], average_ranks([t["reconstruction_error"] for t in nonbasis])))
        for t in sorted(case["tokens"], key=lambda t: t["token_index"]):
            i = t["token_index"]
            token_rows.append({**common, "token_index": i, "E": t["reconstruction_error"], "I": t["intervention_score"],
                "group": group_of.get(i, "fps_basis"), "E_percentile_nonbasis": (ranks[i]-1)/(len(nonbasis)-1) if i in ranks else None,
                "direct_original_retained": i in record["retained"],
                "basis_replacement_slot": i in record["native"]["input_merge"]["basis_slots"]})
    pairs, pair_tokens = [], []
    by_image = defaultdict(list)
    for record in records:
        if record["dataset"] == "gqa":
            by_image[record["case"]["image_id"]].append(record)
    for image in by_image.values():
        for a, b in combinations(image, 2):
            pair, tokens = pair_summary(a, b)
            pairs.append(pair); pair_tokens.extend(tokens)
    cohorts = {dataset: cohort([r for r in questions if r["dataset"] == dataset]) for dataset in plan["sources"]}
    dataset_groups = []
    for dataset in cohorts:
        for group in ("low", "middle", "high", "fps_basis"):
            subset = [r for r in groups if r["dataset"] == dataset and r["group"] == group and r["n"]]
            dataset_groups.append({"dataset": dataset, "group": group, "questions": len(subset),
                "median_question_mean_I": statistics.median(r["I_mean"] for r in subset),
                "median_question_median_I": statistics.median(r["I_median"] for r in subset),
                "median_question_positive_fraction": statistics.median(r["positive_fraction"] for r in subset)})
    strata = []
    for dataset in cohorts:
        subset = [r for r in questions if r["dataset"] == dataset]
        for dimension in ("task", "archived_role"):
            for label in sorted({r[dimension] for r in subset}):
                strata.append({"dataset": dataset, "dimension": dimension, "label": label,
                               **cohort([r for r in subset if r[dimension] == label])})
    mm_sensitivity = cohort([r for r in questions if r["dataset"] == "mmstar" and r["case_id"] not in ("mmstar:604", "mmstar:617")])
    exemplars = []
    for entry in plan["visual_examples"]:
        for i in entry["tokens"]:
            row = next(t for t in token_rows if t["case_id"] == entry["case_id"] and t["token_index"] == i)
            paired = next((t for t in token_rows if t["case_id"] == entry.get("paired_case_id") and t["token_index"] == i), None)
            exemplars.append({**row, "reason": entry["reason"], "paired_case_id": entry.get("paired_case_id"),
                              "paired_I": paired["I"] if paired else None})
    result = {"schema": "rq1-saved-group-pair-analysis-v1", "plan_sha256": sha256(plan_path),
        "analysis_script_sha256": sha256(Path(__file__)), "sources": source_records,
        "question_count": len(questions), "image_count": len({(r['dataset'], r['image_id']) for r in questions}),
        "intervention_count": len(token_rows), "cohorts": cohorts, "dataset_groups": dataset_groups, "gqa_pairs": pairs,
        "mmstar_without_flagged_cases": mm_sensitivity, "illustrations": exemplars,
        "interpretation": plan["status"], "limitations": plan["limitations"]}
    for name, rows in [("question-groups", groups), ("questions", questions), ("tokens", token_rows),
                       ("gqa-pairs", pairs), ("gqa-paired-tokens", pair_tokens), ("strata", strata),
                       ("dataset-groups", dataset_groups), ("illustrations", exemplars)]:
        write_csv(output / (name + ".csv"), rows)
    write_json(output / "plan.json", plan)
    write_json(output / "summary.json", result)
    render(output, result, records, questions, groups)
    return result


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def fmt(value):
    return "undefined" if value is None else f"{value:+.5f}"


def render(output, result, records, questions, groups):
    lookup = {r["case"]["case_id"]: r for r in records}
    figure_dir = output / "figures"
    figure_dir.mkdir()
    figure_files = {}
    for case_id, r in lookup.items():
        name = case_id.replace(":", "-") + ".png"
        source = r["source"] / "analysis" / (r["case"]["question_id"] + ".png")
        shutil.copyfile(source, figure_dir / name)
        figure_files[name] = sha256(source)
    write_json(output / "figure-sha256.json", figure_files)
    def figure(case_id):
        path = "figures/" + case_id.replace(":", "-") + ".png"
        return f'<img loading="lazy" src="{html.escape(path, quote=True)}" alt="Saved image, E and I maps for {case_id}">'
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ1 saved-data group and pair analysis</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto;padding:0 1em}table{border-collapse:collapse;margin:1em 0}th,td{padding:.55em;border-bottom:1px solid #ccc;text-align:left}img{width:100%;height:auto}section{border-top:1px solid #aaa;margin-top:2em;padding-top:1em}a{color:#145b99}</style>',
        '<h1>RQ1 saved-data analysis — 2026-10-17</h1><p>30 questions / 24 images / 17,280 singleton effects. No GPU inference. Low/middle/high are average-rank thirds of non-basis E within each image; FPS basis sources are separate. Signed I is reference support under singleton zeroing, not an accuracy change. Mean and median group effects do not measure jointly removing a group.</p>',
        table(['Dataset','Questions / images','High mean > low','Median per-question high − low mean I','Median per-image mean contrast','Quartile sensitivity contrast'],[
            [d,f'{c["questions"]} / {c["images"]}',f'{c["high_mean_exceeds_low_questions"]} / {c["questions_with_defined_contrast"]}',fmt(c['median_question_high_minus_low_mean_I']),fmt(c['median_image_mean_high_minus_low_I']),fmt(c['median_quartile_high_minus_low_mean_I'])] for d,c in result['cohorts'].items()]),
        '<p>All contrasts use nats per reference-answer token. Descriptive curated cohorts; no pooled-token p-values. Answer format/confidence and task differ. Original PNGs below are reused without alteration. CSV tables expose all observations, including opposite effects.</p>',
        '<p><a href="summary.json">Summary JSON</a> · <a href="plan.json">Post-hoc analysis plan</a> · <a href="question-groups.csv">All group distributions</a> · <a href="questions.csv">Question contrasts</a> · <a href="strata.csv">Task/role strata</a> · <a href="tokens.csv">All token groups</a></p>',
        '<h2>GQA: identical image features and E, different questions</h2>',
        table(['Image','Question A / target','Question B / target','I rho (non-basis)','Top-57 overlap','Opposite sign beyond 0.0001'],[
            [p['image_id'],p['question_a']+' / '+p['target_a'],p['question_b']+' / '+p['target_b'],fmt(p['rho_nonbasis']),str(p['top_overlap'])+' / '+str(p['top_set_size']),p['opposite_sign_beyond_1e_minus4']] for p in result['gqa_pairs']]),
        '<p>Top sets use the highest signed I, not absolute influence. Opposite sign uses a descriptive 0.0001 cutoff, not an estimated noise floor. Wording, target and confidence differ within pairs. No claim of isolated ground-truth relevance follows.</p>']
    for pair in result['gqa_pairs']:
        parts.append(f'<section><h3>{html.escape(pair["image_id"])}</h3>')
        for field in ['question_a','question_b']:
            cid=pair[field];case=lookup[cid]['case']
            parts.append(f'<p><b>{cid}</b>: {html.escape(case["question"])} Reference: {html.escape(case["measurement_target"])}.</p>'+figure(cid))
        parts.append('</section>')
    parts.append('<h2>Previously observed qualitative examples</h2><p>Selected after observing E/I, explicitly illustrative. No new frozen dataset is created. Outside direct-original retention does not mean absent from the compressed representation; merging/basis replacement can carry information. Border patches contain contextualized features.</p>')
    for entry in json.loads((output/'plan.json').read_text())['visual_examples']:
        cid=entry['case_id'];selected=[r for r in result['illustrations'] if r['case_id']==cid]
        parts.append(f'<section><h3>{cid}</h3><p>{html.escape(entry["reason"])}</p>'+table(
            ['Patch','E','Non-basis E percentile','Group','Signed I','Direct original retained','Paired-question I'],[
                [r['token_index'],f'{r["E"]:.4f}',f'{100*r["E_percentile_nonbasis"]:.1f}%' if r['E_percentile_nonbasis'] is not None else 'FPS basis',r['group'],fmt(r['I']),r['direct_original_retained'],fmt(r['paired_I'])] for r in selected])+figure(cid)+'</section>')
    parts.append('<h2>All question distributions</h2><p>Each row reports separate singleton effects. Q25/Q75 describe patch-score distributions, not sampling confidence intervals.</p>')
    for q in questions:
        parts.append(f'<h3>{q["case_id"]}: {html.escape(q["question"])}</h3><p><a href="figures/{q["case_id"].replace(":", "-")}.png">Saved image/E/I figure</a></p>'+table(
            ['E group','Patches','Mean signed I','Median signed I','Q25','Q75','Positive-I fraction'],[
                [r['group'],r['n'],fmt(r['I_mean']),fmt(r['I_median']),fmt(r['I_q25']),fmt(r['I_q75']),f'{100*r["positive_fraction"]:.1f}%' if r['n'] else 'undefined'] for r in groups if r['case_id']==q['case_id']]))
    (output/'index.html').write_text('\n'.join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO / "research/results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args()
    result = analyze(args.results, args.output, args.plan)
    print(json.dumps({k: result[k] for k in ("question_count", "image_count", "intervention_count", "cohorts", "gqa_pairs")}, indent=2))
