"""MMStar identity/scoring/statistical checks; no model or Torch required."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import rq2_staged_mmstar_protocol as p
from rq2_protocol import score_response
from rq2_staged_protocol import CONDITIONS


def primary_rows(ids):
    cases = {c["case_id"]: c for c in p.load_reference()["cases"]}
    return [dict(case_id=cid, condition=name, image_id=cases[cid]["image_id"],
        category=cases[cid]["category"], l2_category=cases[cid]["l2_category"],
        index=cases[cid]["index"], response=cases[cid]["canonical_target"],
        scoring=score_response(cases[cid], cases[cid]["canonical_target"]), control_only=False,
        generated_tokens=2) for cid in ids for name in CONDITIONS]


class MMStarStagedProtocolTests(unittest.TestCase):
    def test_full1500_image_disjoint_balanced_parts_and_controls(self):
        profile = p.load_profile()
        parts = [p.selected_ids(profile, [i]) for i in range(6)]
        self.assertEqual([len(v) for v in parts], [250] * 6)
        self.assertEqual(len(set(c for v in parts for c in v)), 1500)
        self.assertEqual(p.selected_ids(profile, list(range(6))), profile["case_ids"])
        seen = set()
        for part in parts:
            images = {profile["case_image_ids"][c] for c in part}
            self.assertFalse(seen & images)
            seen.update(images)
            self.assertTrue(set(part) & set(profile["identity_case_ids"]))
        self.assertEqual(len(seen), 1430)
        for invalid in ([], [0, 0], [2, 1], [6], [-1], [True]):
            with self.assertRaises(ValueError):
                p.selected_ids(profile, invalid)

    def test_changed_profile_and_filtered_manifest_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            profile = p.load_profile()
            profile["shards"]["0"].pop()
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "parts or controls"):
                p.load_profile(path)
            manifest = p.manifest_data()
            manifest["cases"].pop()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "filtered/edited"):
                p.validate_inputs(path)

    def test_preparation_preserves_question_image_prompt_and_rejects_label_drift(self):
        from PIL import Image
        from eval_mmstar import prepare_question
        blob = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(blob, format="PNG")
        raw = blob.getvalue()
        image_hash = hashlib.sha256(raw).hexdigest()
        pixel_hash = hashlib.sha256(Image.open(io.BytesIO(raw)).convert("RGB").tobytes()).hexdigest()
        question = "Which?\nA: red, B: blue, C: green, D: yellow"
        ref = dict(dataset={"fixture": True}, cases=[dict(case_id="mmstar:0", index=0, row=0,
            image_id=image_hash, image_sha256=image_hash, rgb_pixels_sha256=pixel_hash,
            image_size=[3, 2], question=question, generation_question=prepare_question(question),
            canonical_target="A", category="coarse perception", l2_category="fixture")])
        rows = [dict(index=0, question=question, answer="A", category="coarse perception",
            l2_category="fixture", image={"bytes": raw})]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = p.prepare_rows(rows, root / "good", ref)
            self.assertEqual(json.loads(path.read_text()), p.manifest_data(ref))
            self.assertEqual((path.parent / "images" / (image_hash + ".img")).read_bytes(), raw)
            wrong = copy.deepcopy(rows)
            wrong[0]["answer"] = "B"
            with self.assertRaisesRegex(ValueError, "question/label"):
                p.prepare_rows(wrong, root / "wrong", ref)
            with self.assertRaisesRegex(ValueError, "Preserve existing"):
                p.prepare_rows(rows, root / "good", ref)

    def test_image_bootstrap_is_question_weighted_for_unequal_clusters(self):
        ids = list(map(str, range(10)))
        left = dict.fromkeys(ids, 1.0)
        right = {c: 0.0 if int(c) < 9 else 1.0 for c in ids}
        images = {c: "large" if int(c) < 9 else "small" for c in ids}
        result = p.paired_difference(left, right, ids, images, intervals=True)
        self.assertEqual(result["questions"], 10)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["mean_accuracy_delta"], .9)  # Not the unweighted image mean .5.
        self.assertEqual(result["paired_image_bootstrap_95_ci"], [0.0, 1.0])

    def test_scoring_categories_seed_mean_and_baseline_wrong_cases(self):
        ids = ["mmstar:0", "mmstar:250"]
        rows = primary_rows(ids)
        cases = {c["case_id"]: c for c in p.load_reference()["cases"]}
        for r in rows:
            if r["condition"] in ("baseline576", "pure-random-96-to-32-s101"):
                r.update(response="unclear explanation", scoring=score_response(cases[r["case_id"]], "unclear explanation"))
        result = p.summarize(rows, expected_ids=ids)
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["baseline_correct"], 0)
        native = result["conditions"]["native-96-to-32"]
        self.assertIsNone(native["APR"])
        self.assertEqual(len(native["gains"]), 2)
        self.assertEqual(len(result["category_summaries"]["category"]), 2)
        pair = result["paired_comparisons"]["pure-random-96-to-32-seed-mean minus native-96-to-32"]
        self.assertAlmostEqual(pair["mean_accuracy_delta"], -1/3)
        self.assertEqual(pair["questions"], 2)
        self.assertEqual(len(result["conditions"]["baseline576"]["unparsed_ids"]), 2)

    def test_edited_scores_duplicates_controls_and_incomplete_arms_rejected(self):
        rows = primary_rows(["mmstar:0"])
        bad_score = copy.deepcopy(rows)
        bad_score[0]["scoring"]["correct"] = False
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], "control_only": True}], bad_score):
            with self.assertRaises(ValueError):
                p.summarize(bad)

    def test_historical_baseline_drift_is_recorded_without_retargeting(self):
        ref = p.load_reference()
        c = p.manifest_data(ref)["cases"][0]
        old = copy.deepcopy(c)
        r = dict(response="different", scoring=score_response(c, "different"))
        change = p.historical_comparison(c, {"baseline576": r}, ref)
        self.assertFalse(change["response_equal"])
        self.assertEqual(c, old)


if __name__ == "__main__":
    unittest.main()
