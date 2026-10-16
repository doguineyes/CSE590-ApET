"""Export per-token RQ1 tables, per-question Spearman values and aligned plots.

Analyze saved measurements only: no checkpoint loading or inference. No pooled
token p-values; repeated tokens/questions from one image are not independent.
"""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_protocol import spearman, write_json


def quadrant_examples(tokens):
    """Descriptive outer-quartile exemplars, never relabel negative I as a loss."""
    def quantile(values, q):
        values = sorted(values)
        return values[int((len(values)-1)*q)]
    errors = [r["reconstruction_error"] for r in tokens]
    impacts = [r["intervention_score"] for r in tokens]
    if not tokens:
        return {}
    low_e, high_e = quantile(errors, .25), quantile(errors, .75)
    low_i, high_i = quantile(impacts, .25), quantile(impacts, .75)
    result = {}
    for name, candidates in {
        "low_E_high_I": [r for r in tokens if r["reconstruction_error"] <= low_e and r["intervention_score"] >= high_i and r["intervention_score"] > 0],
        "high_E_high_I": [r for r in tokens if r["reconstruction_error"] >= high_e and r["intervention_score"] >= high_i and r["intervention_score"] > 0],
        "low_E_low_I": [r for r in tokens if r["reconstruction_error"] <= low_e and r["intervention_score"] <= low_i],
        "high_E_low_I": [r for r in tokens if r["reconstruction_error"] >= high_e and r["intervention_score"] <= low_i],
    }.items():
        reverse = name.endswith("high_I")
        result[name] = sorted(candidates, key=lambda r: r["intervention_score"], reverse=reverse)[:2]
    return {"thresholds": {"E_q25": low_e, "E_q75": high_e, "I_q25": low_i, "I_q75": high_i},
            "examples": result}


def plot_case(folder, case, errors, exemplars):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image

    canvas = Image.open(folder / "images" / f"{case['image_sha256']}-canvas.png")
    impacts = np.full(576, np.nan)
    for row in case["tokens"]:
        impacts[row["token_index"]] = row["intervention_score"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True)
    axes[0].imshow(canvas)
    axes[0].set_title("Actual CLIP input (336 × 336)")
    seen = set()
    for rows in exemplars.get("examples", {}).values():
        for row in rows:
            index = row["token_index"]
            if index in seen:
                continue
            seen.add(index)
            x, y = index % 24 * 14, index // 24 * 14
            axes[0].add_patch(Rectangle((x-.5, y-.5), 14, 14, fill=False, edgecolor="yellow", linewidth=1.3))
            axes[0].text(x, y, str(index), color="yellow", fontsize=6,
                         bbox=dict(facecolor="black", alpha=.5, pad=0))
    axes[0].axis("off")
    heat = axes[1].imshow(np.array(errors).reshape(24, 24), cmap="viridis")
    axes[1].set_title("Native reconstruction error E")
    fig.colorbar(heat, ax=axes[1], shrink=.7)
    maximum = max(float(np.nanmax(np.abs(impacts))), 1e-7)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    heat = axes[2].imshow(impacts.reshape(24, 24), cmap=cmap, vmin=-maximum, vmax=maximum)
    axes[2].set_title(f"Sensitivity I ({len(case['tokens'])}/576)\nGray = unmeasured")
    fig.colorbar(heat, ax=axes[2], shrink=.7)
    tokens = case["tokens"]
    axes[3].scatter([r["reconstruction_error"] for r in tokens],
                    [r["intervention_score"] for r in tokens], s=10, alpha=.6)
    axes[3].axhline(0, color="gray", linewidth=.7)
    drift = max(case["controls"].get("max_pre_token_logprob_delta", 0),
                case["controls"].get("post_token_logprob_delta", 0))
    if drift:
        axes[3].axhspan(-drift, drift, color="gray", alpha=.15, label="Observed control drift")
        axes[3].legend(fontsize=7)
    axes[3].set_xlabel("Reconstruction error E")
    axes[3].set_ylabel("S(full) − S(zero), nats/token")
    axes[3].set_title(f"Per-question Spearman ρ = {case['spearman']:.3f}" if case["spearman"] is not None else "Spearman undefined")
    title = case["question"] if len(case["question"]) < 140 else case["question"][:137] + "..."
    target = case.get("measurement_target", case["canonical_target"])
    fig.suptitle(f"{case['case_id']}: {title}\nMeasured reference: {target}; controls: {case['status']}", fontsize=10)
    destination = folder / "analysis" / case["question_id"]
    fig.savefig(destination.with_suffix(".png"), dpi=150)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def analyze(folder):
    folder = Path(folder)
    analysis = folder / "analysis"
    analysis.mkdir(exist_ok=True)
    summaries = []
    sections = []
    pairs = defaultdict(list)
    with (analysis / "tokens.jsonl").open("w") as flat:
        for path in sorted((folder / "questions").glob("*.json")):
            case = json.loads(path.read_text())
            if not case.get("tokens"):
                continue
            features = json.loads((folder / "features" / f"{case['image_sha256']}.json").read_text())
            errors = features["diagnostics"]["input"]["errors"]
            lookup = {r["token_index"]: r for r in case["tokens"]}
            for index, error in enumerate(errors):
                measured = lookup.get(index)
                flat.write(json.dumps({"sample_id": case["case_id"], "image_id": case["image_id"],
                    "measurement_version": case.get("measurement_version", "rq1-input-zero-v1"),
                    "canonical_target": case["canonical_target"],
                    "measurement_target": case.get("measurement_target", case["canonical_target"]),
                    "stage": "input_post_projector", "layer": None, "token_index": index,
                    "reconstruction_error": error, "measured": measured is not None,
                    "intervention_score": measured["intervention_score"] if measured else None,
                    "reference_mean_logprob": case["controls"]["reference"]["mean_logprob"],
                    "controls_passed": case["controls"]["passed"],
                    "question_status": case["status"]}, allow_nan=False) + "\n")
            rho = spearman([r["reconstruction_error"] for r in case["tokens"]],
                           [r["intervention_score"] for r in case["tokens"]])
            case["spearman"] = rho
            exemplars = quadrant_examples(case["tokens"])
            summary = {"case_id": case["case_id"], "image_id": case["image_id"], "status": case["status"],
                       "measurement_version": case.get("measurement_version", "rq1-input-zero-v1"),
                       "measurement_target": case.get("measurement_target", case["canonical_target"]),
                       "target_alignment": case.get("target_alignment"),
                       "measured_tokens": len(lookup), "spearman": rho, "quadrants": exemplars,
                       "sensitivity_min": min(r["intervention_score"] for r in case["tokens"]),
                       "sensitivity_max": max(r["intervention_score"] for r in case["tokens"]),
                       "controls": case["controls"], "free_generation": case["free_generation"]}
            summaries.append(summary)
            pairs[case["image_id"]].append(case)
            plot_case(folder, case, errors, exemplars)
            examples = "".join(f"<li>{html.escape(name)}: " + ", ".join(
                f"token {r['token_index']} (E={r['reconstruction_error']:.3g}, I={r['intervention_score']:.3g})" for r in rows) + "</li>"
                for name, rows in exemplars.get("examples", {}).items())
            sections.append(f"<section><h2>{html.escape(case['case_id'])}: {html.escape(case['question'])}</h2>"
                f"<p>Annotation: {html.escape(case['canonical_target'])}; measured reference: "
                f"{html.escape(case.get('measurement_target', case['canonical_target']))}; fresh baseline: "
                f"{html.escape(case['free_generation']['response'])}; control status: {html.escape(case['status'])}</p>"
                f"<a href='{case['question_id']}.pdf'><img src='{case['question_id']}.png' width='100%'></a><ul>{examples}</ul></section>")
    same_image = []
    for image_id, cases in pairs.items():
        for i, first in enumerate(cases):
            for second in cases[i+1:]:
                a = {r["token_index"]: r["intervention_score"] for r in first["tokens"]}
                b = {r["token_index"]: r["intervention_score"] for r in second["tokens"]}
                common = sorted(set(a) & set(b))
                same_image.append({"image_id": image_id, "case_ids": [first["case_id"], second["case_id"]],
                    "same_features": first["clip_sha256"] == second["clip_sha256"],
                    "common_tokens": len(common), "sensitivity_spearman_between_questions": spearman(
                        [a[k] for k in common], [b[k] for k in common])})
    write_json(analysis / "summary.json", {"questions": summaries, "same_image_comparisons": same_image,
        "limitations": "Descriptive within-question correlations on an outcome-enriched subset; no pooled independent-token inference. "
                       "I is a signed zeroing proxy. Gray/missing tokens are unmeasured. Check reference drift before interpretation."})
    (analysis / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>RQ1 input pilot</title>"
        "<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:15px}section{border-top:1px solid #ccc;padding:20px 0}</style>"
        "<h1>RQ1 input-stage pilot</h1><p>Exploratory zero-embedding sensitivity; no token deletion or decoder-layer claim. "
        "Images show the actual padded/preprocessed CLIP canvas. Positive I means the fixed reference becomes less probable. "
        "Near-zero effects should be compared with control drift. Correlations are per question, not population estimates.</p>" + "".join(sections))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
