"""CPU-only finding inventory from saved JSON/CSV, without loading tensors.

The broad routing summaries are new, post-hoc descriptive calculations from
existing native records. They are not additional model experiments.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "research/results"
OUT = Path(__file__).with_name("findings_audit_2026-10-30")
SCHEDULES = ["96-to-32", "288-to-96"]
ARCHIVES = {
    "textvqa": "rq2-staged-textvqa100-v1-20261021T134751714241Z.zip",
    "gqa": "rq2-staged-gqa800-v1-20261021T142303483991Z.zip",
    "mmstar": "rq2-staged-mmstar1500-v1-20261021T153730115642Z.zip",
}


def read_csv(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def write_json(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def graph_record(ds, boundary, schedule, key, image, errors, graph):
    targets = graph["target_indices"]
    basis = graph["basis_slots"]
    counts = graph["merge_counts"]
    dropped = graph["dropped_indices"]
    assignments = graph["assigned_target_indices"]
    K = len(targets) + len(basis)
    N = graph["source_tokens"]
    assert len(errors) == N and len(basis) == 10
    assert not (set(targets) & set(basis))
    assert len(set(targets + basis)) == K
    assert len(dropped) == len(assignments) == sum(counts) == N - K
    assert all(t in targets for t in assignments) and graph["scaling"] == 1
    counter = Counter(assignments)
    assert [counter[t] for t in targets] == counts
    ranked = sorted(zip(targets, counts), key=lambda x: (errors[x[0]], x[0]))
    row = {"dataset": ds, "boundary": boundary, "schedule": schedule,
           "record_id": key, "image_sha256": image,
           "source_states": N, "output_states": K, "ordinary_targets": len(targets),
           "basis_slots": len(basis), "dropped_sources": len(dropped),
           "ordinary_targets_with_incoming": sum(c > 0 for c in counts),
           "ordinary_targets_without_incoming": sum(c == 0 for c in counts),
           "outputs_not_averaged_by_merge": len(basis) + sum(c == 0 for c in counts),
           "fraction_outputs_not_averaged": (len(basis) + sum(c == 0 for c in counts)) / K,
           "max_incoming": max(counts)}
    for q in range(4):
        group = ranked[len(ranked) * q // 4:len(ranked) * (q + 1) // 4]
        mass = sum(c for _, c in group)
        row[f"q{q + 1}_targets"] = len(group)
        row[f"q{q + 1}_incoming"] = mass
        row[f"q{q + 1}_fraction_incoming"] = mass / len(dropped)
        row[f"q{q + 1}_targets_without_incoming"] = sum(c == 0 for _, c in group)
    assert sum(row[f"q{i}_incoming"] for i in range(1, 5)) == len(dropped)
    return row


def collect_broad():
    graph_rows, questions, budget_rows, transition_rows, inventories = [], [], [], [], []
    for ds, filename in ARCHIVES.items():
        archive = RESULTS / filename
        print("Reading", ds, flush=True)
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            encodings = [n for n in names if n.startswith("encodings/") and n.endswith(".json")]
            q_names = [n for n in names if n.startswith("questions/") and n.endswith(".json")]
            for name in encodings:
                e = json.loads(z.read(name))
                for schedule in SCHEDULES:
                    arm = e["encodings"]["native-" + schedule]
                    graph_rows.append(graph_record(ds, "input", schedule, name,
                        e["image_sha256"], arm["input"]["errors"], arm["input_merge"]))
            for name in q_names:
                q = json.loads(z.read(name))
                assert q["status"] == "passed"
                c = q["case"]
                base = q["arms"]["baseline576"]
                row = {"dataset": ds, "case_id": q["case_id"], "image_sha256": c["image_sha256"],
                       "question": c["question"], "gold": c["gold_answers"], "arms": {}}
                for condition, arm in q["arms"].items():
                    if condition != "baseline576" and condition not in ["native-" + s for s in SCHEDULES] and not any(condition == f"pure-random-{s}-s{seed}" for s in SCHEDULES for seed in [101, 202, 303]):
                        continue
                    scoring = arm["scoring"]
                    row["arms"][condition] = {"correct": bool(scoring["correct"]),
                        "score": float(scoring["consensus_score"] if ds == "textvqa" else scoring["correct"]),
                        "response": arm["response"], "prediction": scoring["prediction"]}
                assert len(row["arms"]) == 9
                b = row["arms"]["baseline576"]
                small, large = (row["arms"]["native-" + s] for s in SCHEDULES)
                role = ("both_correct" if small["correct"] and large["correct"] else
                        "recovered_at_larger" if large["correct"] else
                        "lost_at_larger" if small["correct"] else "both_wrong")
                budget_rows.append({"dataset": ds, "case_id": q["case_id"], "image_sha256": c["image_sha256"],
                    "question": c["question"], "gold": " | ".join(c["gold_answers"]),
                    "baseline_correct": b["correct"], "baseline_response": b["response"],
                    "severe_correct": small["correct"], "moderate_correct": large["correct"],
                    "severe_response": small["response"], "moderate_response": large["response"],
                    "severe_score": small["score"], "moderate_score": large["score"], "budget_role": role})
                for schedule in SCHEDULES:
                    arm = q["arms"]["native-" + schedule]
                    dec = arm["decoder"]
                    graph_rows.append(graph_record(ds, "decoder16", schedule, q["case_id"],
                        c["image_sha256"], dec["E"]["errors"], dec["merge"]))
                    a = row["arms"]["native-" + schedule]
                    transition = ("preserved" if b["correct"] and a["correct"] else
                                  "loss" if b["correct"] else "gain" if a["correct"] else "both_wrong")
                    if transition in ["loss", "gain"] or a["score"] != b["score"]:
                        transition_rows.append({"dataset": ds, "case_id": q["case_id"],
                            "image_sha256": c["image_sha256"], "schedule": schedule,
                            "question": c["question"], "gold": " | ".join(c["gold_answers"]),
                            "baseline_response": b["response"], "native_response": a["response"],
                            "baseline_score": b["score"], "native_score": a["score"],
                            "transition": transition})
                questions.append(row)
            inventories.append({"dataset": ds, "archive": filename, "sha256": fingerprint(archive),
                "encoding_records": len(encodings), "question_records": len(q_names)})
    write_csv("broad_routing_per_record.csv", graph_rows)
    write_csv("native_budget_transitions.csv", budget_rows)
    write_csv("native_losses_gains_and_score_changes.csv", transition_rows)
    return graph_rows, questions, budget_rows, inventories


def summarize_graphs(rows):
    summaries = []
    for ds in ARCHIVES:
        for boundary in ["input", "decoder16"]:
            for schedule in SCHEDULES:
                group = [r for r in rows if (r["dataset"], r["boundary"], r["schedule"]) == (ds, boundary, schedule)]
                by_image = defaultdict(list)
                for r in group:
                    by_image[r["image_sha256"]].append(r)
                r0 = group[0]
                summary = {"dataset": ds, "boundary": boundary, "schedule": schedule,
                    "records": len(group), "unique_images": len(by_image),
                    "ordinary_targets_per_record": r0["ordinary_targets"],
                    "dropped_sources_per_record": r0["dropped_sources"]}
                for q in range(1, 5):
                    values = [np.mean([r[f"q{q}_fraction_incoming"] for r in image]) for image in by_image.values()]
                    summary[f"q{q}_target_count"] = r0[f"q{q}_targets"]
                    summary[f"q{q}_pooled_fraction_incoming"] = sum(r[f"q{q}_incoming"] for r in group) / sum(r["dropped_sources"] for r in group)
                    summary[f"q{q}_image_equal_mean_fraction_incoming"] = float(np.mean(values))
                    summary[f"q{q}_image_median_fraction_incoming"] = float(np.median(values))
                values = [np.mean([r["fraction_outputs_not_averaged"] for r in image]) for image in by_image.values()]
                summary["image_equal_fraction_outputs_not_averaged"] = float(np.mean(values))
                summaries.append(summary)
    write_csv("broad_routing_summary.csv", summaries)
    return summaries


def summarize_answers(questions, budgets):
    conditions, budget_summary, decompositions, mixed_images, mm_labels = [], [], [], [], []
    for ds in ARCHIVES:
        qs = [q for q in questions if q["dataset"] == ds]
        br = [r for r in budgets if r["dataset"] == ds]
        bsum = Counter(r["budget_role"] for r in br)
        budget_summary.append({"dataset": ds, "questions": len(qs), **dict(bsum),
            "baseline_correct_lost_at_larger": sum(r["baseline_correct"] and r["budget_role"] == "lost_at_larger" for r in br),
            "baseline_correct_recovered_at_larger": sum(r["baseline_correct"] and r["budget_role"] == "recovered_at_larger" for r in br)})
        for condition in qs[0]["arms"]:
            B = sum(q["arms"]["baseline576"]["correct"] for q in qs)
            preserved = sum(q["arms"]["baseline576"]["correct"] and q["arms"][condition]["correct"] for q in qs)
            gains = sum(not q["arms"]["baseline576"]["correct"] and q["arms"][condition]["correct"] for q in qs)
            correct = sum(q["arms"][condition]["correct"] for q in qs)
            assert correct == B - (B - preserved) + gains
            conditions.append({"dataset": ds, "condition": condition, "questions": len(qs),
                "score_percent": 100 * np.mean([q["arms"][condition]["score"] for q in qs]),
                "baseline_correct": B, "correct": correct, "preserved": preserved,
                "losses": B - preserved, "gains": gains, "APR_percent": 100 * preserved / B})
        for schedule in SCHEDULES:
            native = "native-" + schedule
            def masses(condition):
                loss = sum(max(q["arms"]["baseline576"]["score"] - q["arms"][condition]["score"], 0) for q in qs)
                gain = sum(max(q["arms"][condition]["score"] - q["arms"]["baseline576"]["score"], 0) for q in qs)
                return np.array([loss, gain]) * 100 / len(qs)
            loss_n, gain_n = masses(native)
            loss_r, gain_r = np.mean([masses(f"pure-random-{schedule}-s{seed}") for seed in [101, 202, 303]], axis=0)
            decompositions.append({"dataset": ds, "schedule": schedule,
                "lower_lost_credit_contribution_pp": float(loss_r - loss_n),
                "higher_gained_credit_contribution_pp": float(gain_n - gain_r),
                "native_minus_random_pp": float(loss_r - loss_n + gain_n - gain_r)})
            image_roles = defaultdict(list)
            for q in qs:
                if q["arms"]["baseline576"]["correct"]:
                    image_roles[q["image_sha256"]].append(q["arms"][native]["correct"])
            mixed_images.append({"dataset": ds, "schedule": schedule,
                "baseline_correct_image_groups": len(image_roles),
                "images_with_both_lost_and_preserved": sum(any(v) and not all(v) for v in image_roles.values())})
            if ds == "mmstar":
                for role in ["loss", "gain"]:
                    subset = [q for q in qs if (q["arms"]["baseline576"]["correct"] and not q["arms"][native]["correct"] if role == "loss" else not q["arms"]["baseline576"]["correct"] and q["arms"][native]["correct"])]
                    mm_labels.append({"schedule": schedule, "role": role, "questions": len(subset),
                        "gold_label_counts": dict(Counter(q["gold"][0] for q in subset)),
                        "baseline_to_native_prediction_counts": dict(Counter(q["arms"]["baseline576"]["prediction"] + "->" + q["arms"][native]["prediction"] for q in subset))})
    write_csv("condition_checks.csv", conditions)
    write_csv("native_budget_summary.csv", budget_summary)
    write_csv("native_random_credit_decomposition.csv", decompositions)
    write_csv("same_image_loss_preserved_counts.csv", mixed_images)
    write_json("mmstar_label_transitions.json", mm_labels)
    return conditions, budget_summary, decompositions, mixed_images, mm_labels


def selected_component_checks():
    folder = RESULTS / "rq2-crossed-full-v1-review-2026-10-19"
    rows = read_csv(folder / "answers.csv")
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    ids = sorted({r["case_id"] for r in rows})
    paired = []
    for selection in ["E", "random-101", "random-202", "random-303"]:
        before = [lookup[cid, selection + "-no-merge"]["correct"] == "True" for cid in ids]
        after = [lookup[cid, selection + "-merge"]["correct"] == "True" for cid in ids]
        paired.append({"selection": selection, "rescued_by_merging": sum(not a and b for a, b in zip(before, after)),
            "lost_by_merging": sum(a and not b for a, b in zip(before, after)),
            "correct_without_merge": sum(before), "correct_with_merge": sum(after)})
    write_csv("selected_panel_merge_interactions.csv", paired)
    folder = RESULTS / "rq2-routing-full-v1-review-2026-10-19"
    rank = read_csv(folder / "native-rank-activity.csv")
    rank_counts = [sum(int(r[k]) for r in rank) for k in ["lowest_E_quartile_incoming", "lower_middle_E_quartile_incoming", "upper_middle_E_quartile_incoming", "highest_E_quartile_incoming"]]
    assert rank_counts == [4569, 1909, 308, 126]
    activity = read_csv(folder / "routing-activity.csv")
    answers = read_csv(folder / "answers.csv")
    native = {r["case_id"]: r for r in answers if r["condition"] == "native-nearest"}
    routing = []
    for condition in sorted({r["condition"] for r in activity}):
        a = [r for r in answers if r["condition"] == condition]
        unique = {r["image_sha256"]: r for r in activity if r["condition"] == condition}
        rows = list(unique.values())
        routing.append({"condition": condition, "unique_images": len(unique),
            "changed_assignments": sum(int(r["changed_assignments"]) for r in rows),
            "mean_native_cosine_image_equal": float(np.mean([float(r["mean_native_cosine"]) for r in rows])),
            "mean_new_destination_cosine_image_equal": float(np.mean([float(r["mean_new_destination_cosine"]) for r in rows])),
            "raw_answer_changes_vs_native": sum(r["response"] != native[r["case_id"]]["response"] for r in a),
            "correctness_changes_vs_native": sum(r["correct"] != native[r["case_id"]]["correct"] for r in a)})
    write_csv("selected_panel_routing_checks.csv", routing)
    return {"incoming_counts_by_quartile": rank_counts, "images": len(rank),
            "top28_incoming": sum(int(r["top28_incoming"]) for r in rank),
            "bottom28_incoming": sum(int(r["bottom28_incoming"]) for r in rank),
            "paired_merge_effects": paired, "routing": routing}


def rank(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="stable")
    values = a[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    ends = np.r_[starts[1:], len(a)]
    r = np.empty(len(a))
    for lo, hi in zip(starts, ends):
        r[order[lo:hi]] = (lo + hi - 1) / 2
    return r


def correlation_checks():
    summaries = []
    folders = [RESULTS / "rq1-input-ih-synthesis-v1-2026-10-18",
               RESULTS / "rq1-layer16-ih-full-review-v1-2026-10-17"]
    for boundary, folder in zip(["input", "decoder16"], folders):
        records = read_csv(folder / "verified-comparison.csv")
        for cid in sorted({r["case_id"] for r in records}):
            group = [r for r in records if r["case_id"] == cid]
            E = np.array([float(r["E"]) for r in group])
            zero = np.array([float(r["zero_first_probability"]) for r in group])
            ih = np.array([float(r["ih_first_probability"]) for r in group])
            corr = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
            summaries.append({"boundary": boundary, "case_id": cid,
                "E_vs_zero_signed": corr(E, zero), "E_vs_retention_signed": corr(E, ih),
                "E_vs_absolute_retention": corr(E, np.abs(ih)), "zero_vs_retention_signed": corr(zero, ih),
                "positions": len(group)})
    write_csv("correlation_measure_checks.csv", summaries)
    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    graphs, questions, budgets, inventory = collect_broad()
    graph_summaries = summarize_graphs(graphs)
    conditions, budget_summaries, decomposition, mixed, labels = summarize_answers(questions, budgets)
    selected = selected_component_checks()
    correlations = correlation_checks()
    write_json("audit_checks.json", {"date": "2026-10-30", "raw_archives": inventory,
        "scope": "Existing JSON/CSV reanalysis; no GPU/model execution or tensor deserialization; broad routing summaries are post-hoc descriptive analyses",
        "broad_questions": len(questions), "routing_records": len(graphs),
        "routing_summary": graph_summaries, "budget_transitions": budget_summaries,
        "native_random_credit_decomposition": decomposition, "same_image_outcomes": mixed,
        "selected_panel": selected, "correlations_recomputed": len(correlations) * 4,
        "limitations": ["Source archives were previously audited; this pass reads JSON and hashes ZIP bytes, not numerical tensor contents.",
            "Routing summaries are observational; no performance mechanism is established by the flow distribution.",
            "Quartiles use ascending E among eligible ordinary retained targets, excluding basis slots; floor boundaries yield group sizes differing by one.",
            "Image-equal summaries average records within the same image hash before averaging images; pooled record counts are also exported.",
            "No incoming merge means no averaging at that operation, not unchanged computation relative to the full model.",
            "No new confirmatory tests or significance claims were added."]})
    print("Budget transitions:", budget_summaries, flush=True)
    print("Routing summary:", flush=True)
    for r in graph_summaries:
        print(r["dataset"], r["boundary"], r["schedule"],
              "low/high incoming %", round(100 * r["q1_image_equal_mean_fraction_incoming"], 2), round(100 * r["q4_image_equal_mean_fraction_incoming"], 2),
              "output not averaged %", round(100 * r["image_equal_fraction_outputs_not_averaged"], 2), flush=True)
    print("Output", OUT, flush=True)


if __name__ == "__main__":
    main()
