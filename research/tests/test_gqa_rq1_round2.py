"""CPU validation of second-round curation and preservation of review evidence."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_gqa_rq1_round2 import (POLICY, build, quality_flags, render_decisions,
                                  review_input, select_round2, validated_assistant_review)
from prepare_rq1_candidates import GQA_ARCHIVE
from rq1_candidates import Archive, CONFIGURATIONS, EXCLUDED_GQA_IMAGES, digest

FIRST = ROOT / "research/results/rq1-candidates-v1-s590/gqa/manifest.json"
ASSISTANT = ROOT / "research/curation/gqa_rq1_round2_assistant_review.json"
# Portable fixture when the Downloads export has been moved: a byte-identical source copy.
HUMAN = ROOT / "research/results/rq1-gqa-reviewed-v1-s591/round1-human-review.json"


class QualityRulesTests(unittest.TestCase):
    def test_alias_and_question_filters_are_symmetric_in_outcome(self):
        rows = {cfg: {"original_question": "What is the person holding?", "question_id": "example",
                      "response": "Racket", "generated_tokens": 1, "correct": True} for cfg in CONFIGURATIONS}
        rows["input-only"]["response"] = "Tennis racket"
        before = quality_flags(rows)
        self.assertIn("likely_answer_alias", before)
        rows["baseline"]["response"], rows["input-only"]["response"] = "Tennis racket", "Racket"
        rows["baseline"]["correct"], rows["input-only"]["correct"] = False, True
        self.assertEqual(before, quality_flags(rows))
        for row in rows.values():
            row["original_question"] = "Is the statue large?"
        self.assertIn("subjective_attribute", quality_flags(rows))


@unittest.skipUnless(FIRST.exists() and HUMAN.exists() and (ROOT / GQA_ARCHIVE).exists(), "Local curation source artifacts absent")
class SavedCurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = Archive(ROOT / GQA_ARCHIVE, "gqa")
        cls.first, cls.human = review_input(FIRST, HUMAN)
        picks, cls.audit, cls.shortfalls = select_round2(cls.archive.rows, set(cls.first["images"]))
        cls.picks = picks
        cls.manifest = {"cases": [cls.archive.case(*p) for p in picks],
            "images": {cls.archive.rows[qid]["baseline"]["image_id"]:
                {"sha256": cls.archive.rows[qid]["baseline"]["image_sha256"]} for qid, _, _ in picks}}

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    def test_selection_is_reproducible_new_and_audited(self):
        again, audit, shortfalls = select_round2(dict(reversed(list(self.archive.rows.items()))), set(self.first["images"]))
        self.assertEqual(self.picks, again)
        self.assertEqual(self.audit, audit)
        self.assertEqual(self.shortfalls, shortfalls)
        self.assertEqual(len(self.picks), 39)
        images = Counter(self.archive.rows[p[0]]["baseline"]["image_id"] for p in self.picks)
        self.assertEqual(len(images), 13)
        self.assertEqual(set(images.values()), {3})
        self.assertFalse(set(images) & set(self.first["images"]))
        self.assertFalse(set(images) & set(EXCLUDED_GQA_IMAGES))
        self.assertEqual(len(audit), 800)
        self.assertTrue(shortfalls)
        self.assertTrue(all(not quality_flags(self.archive.rows[p[0]]) for p in self.picks))

    def test_human_review_identity_mismatch_fails(self):
        changed = copy.deepcopy(self.human)
        changed["manifest_sha256"] = "wrong"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "does not match"):
                review_input(FIRST, path)

    def test_assistant_image_binding_and_complete_coverage(self):
        review = validated_assistant_review(ASSISTANT, self.archive, self.manifest, digest(FIRST))
        self.assertFalse(review["human_validated"])
        for mutate in [lambda r: r["image_sha256"].update({next(iter(r["image_sha256"])): "wrong"}),
                       lambda r: r["decisions"].pop("gqa:" + r["initial_candidate_question_ids"][0])]:
            changed = copy.deepcopy(review)
            mutate(changed)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "review.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    validated_assistant_review(path, self.archive, self.manifest, digest(FIRST))

    def test_complete_build_keeps_sources_scores_and_review_separate(self):
        original_hashes = {path: digest(path) for path in [ROOT / GQA_ARCHIVE, FIRST, HUMAN, ASSISTANT]}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "reviewed"
            args = SimpleNamespace(first_manifest=FIRST, human_review=HUMAN, archive=ROOT / GQA_ARCHIVE,
                                   output=output, seed=591, assistant_review=ASSISTANT)
            with redirect_stdout(io.StringIO()):
                build(args)
            proposed = json.loads((output / "recommended-manifest.json").read_text())
            report = json.loads((output / "review-report.json").read_text())
            self.assertEqual((output / "round1-human-review.json").read_bytes(), HUMAN.read_bytes())
            self.assertEqual(report["human_keeps_preserved"], 30)
            self.assertEqual(report["human_keeps_adopted"], 22)
            self.assertEqual(len(report["human_keeps_set_aside_by_stricter_policy"]), 8)
            self.assertEqual(proposed["counts"]["questions"], 49)
            self.assertEqual(len({c["case_id"] for c in proposed["cases"]}), 49)
            self.assertFalse(proposed["frozen_for_intervention"])
            lookup = {c["question_id"]: c for c in proposed["cases"]}
            for case in proposed["cases"]:
                self.assertEqual(case["curation_review"]["status"], "keep")
                self.assertFalse(case["policy_flags"])
                source = self.archive.rows[case["question_id"]]
                for cfg in CONFIGURATIONS:
                    self.assertEqual(case["predictions"][cfg]["response"], source[cfg]["response"])
                    self.assertEqual(case["predictions"][cfg]["correct"], source[cfg]["correct"])
                    self.assertEqual(case["predictions"][cfg]["settings"], source[cfg]["settings"])
                for qid in case["same_image_preserved_controls"]:
                    self.assertEqual(case["image_id"], lookup[qid]["image_id"])
                    self.assertEqual(lookup[qid]["outcomes"]["input-only"], "preserved_correct")
            index = (output / "index.html").read_text()
            self.assertNotIn("<select", index)
            self.assertIn("Assistant visual review", index)
            self.assertIn("Human review", index)
            self.assertIn("data:image/", index)
            # User/assistant notes are escaped even when displayed as trusted review evidence.
            case = copy.deepcopy(proposed["cases"][0])
            case["curation_review"]["notes"] = "<script>unsafe & text</script>"
            page = output / "escaping-check.html"
            render_decisions(page, "test", [case], proposed["images"], output, "test")
            self.assertNotIn("<script>unsafe", page.read_text())
            self.assertIn("&lt;script&gt;unsafe &amp; text", page.read_text())
            saved_index_hash = digest(output / "index.html")
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                build(args)
            self.assertEqual(digest(output / "index.html"), saved_index_hash)
            with zipfile.ZipFile(output.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(z.read("recommended-manifest.json"), (output / "recommended-manifest.json").read_bytes())
        self.assertEqual(original_hashes, {path: digest(path) for path in original_hashes})


if __name__ == "__main__":
    unittest.main()
