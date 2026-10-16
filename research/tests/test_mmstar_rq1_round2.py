"""CPU curation checks: competence gating, evidence binding and review provenance."""

from collections import Counter
from contextlib import redirect_stdout
import copy
from html.parser import HTMLParser
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
from prepare_mmstar_rq1_round2 import build, flags, load_human, select_additions, validate_review
from prepare_rq1_candidates import MMSTAR_ARCHIVE
from rq1_candidates import Archive, digest

FIRST = ROOT / "research/results/rq1-candidates-v1-s590/mmstar/manifest.json"
HUMAN = ROOT / "research/results/rq1-mmstar-reviewed-v1-s592/round1-human-review.json"
ASSISTANT = ROOT / "research/curation/mmstar_rq1_round2_assistant_review.json"


@unittest.skipUnless(FIRST.exists() and HUMAN.exists() and (ROOT / MMSTAR_ARCHIVE).exists(), "Local MMStar review artifacts absent")
class MMStarCurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first, cls.human = load_human(FIRST, HUMAN)
        cls.archive = Archive(ROOT / MMSTAR_ARCHIVE, "mmstar")
        cls.picks, cls.shortfalls = select_additions(cls.archive.rows, cls.first, cls.human)

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    def test_exact_human_review_binding(self):
        self.assertEqual(Counter(d["status"] for d in self.human["decisions"].values()),
                         {"keep": 17, "exclude": 12, "ambiguous": 1})
        for mutation in [lambda r: r.update(manifest_sha256="changed"),
                         lambda r: r["decisions"].pop(next(iter(r["decisions"])) )]:
            review = copy.deepcopy(self.human)
            mutation(review)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "review.json"
                path.write_text(json.dumps(review))
                with self.assertRaises(ValueError):
                    load_human(FIRST, path)

    def test_small_draw_is_baseline_correct_distinct_and_order_independent(self):
        again, shortfalls = select_additions(dict(reversed(list(self.archive.rows.items()))), self.first, self.human)
        self.assertEqual(self.picks, again)
        self.assertEqual(self.shortfalls, shortfalls)
        self.assertEqual(len(self.picks), 12)
        hashes = set()
        for qid, slot in self.picks:
            row = self.archive.rows[qid]["baseline"]
            self.assertTrue(row["correct"])
            self.assertFalse(flags(self.archive.rows[qid]))
            self.assertNotIn(row["image_sha256"], self.first["images"])
            self.assertNotIn(row["image_sha256"], hashes)
            hashes.add(row["image_sha256"])

    def test_assistant_review_is_bound_and_not_human_validation(self):
        review = validate_review(ASSISTANT, self.archive, FIRST, self.first, self.picks)
        self.assertFalse(review["human_validated"])
        self.assertFalse(review["outcome_blinded"])
        self.assertEqual(len(review["decisions"]), 18)
        for mutation in [lambda r: r.update(predictions_sha256="wrong"),
                         lambda r: r["decisions"]["mmstar:1030"].update(image_sha256="wrong")]:
            changed = copy.deepcopy(review)
            mutation(changed)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "review.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    validate_review(path, self.archive, FIRST, self.first, self.picks)

    def test_primary_appendix_sources_and_rendering(self):
        before = {p: digest(p) for p in [ROOT / MMSTAR_ARCHIVE, FIRST, HUMAN, ASSISTANT]}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "reviewed"
            args = SimpleNamespace(first_manifest=FIRST, human_review=HUMAN, archive=ROOT / MMSTAR_ARCHIVE,
                                   assistant_review=ASSISTANT, output=output, seed=592)
            with redirect_stdout(io.StringIO()):
                report = build(args)
            core = json.loads((output / "recommended-manifest.json").read_text())
            context = json.loads((output / "qualitative-manifest.json").read_text())
            self.assertEqual(core["counts"]["questions"], 18)
            self.assertEqual(len(core["counts"]["categories"]), 6)
            self.assertEqual(core["counts"]["input_outcomes"], {"loss": 6, "preserved_correct": 12})
            self.assertEqual(context["counts"]["input_outcomes"], {"gain": 5, "both_wrong": 5})
            self.assertEqual(report["full_native_baseline"]["correct"], 521)
            self.assertFalse(core["frozen_for_intervention"])
            self.assertEqual((output / "round1-human-review.json").read_bytes(), HUMAN.read_bytes())
            combined = core["cases"] + context["cases"]
            ids = {c["case_id"] for c in combined}
            self.assertEqual(len(combined), len(ids))
            human_keep = {i for i, d in self.human["decisions"].items() if d["status"] == "keep"}
            self.assertTrue(human_keep <= ids)
            for case in core["cases"]:
                self.assertTrue(case["predictions"]["baseline"]["correct"])
                self.assertNotIn(case["outcomes"]["input-only"], {"gain", "both_wrong"})
            for case in combined:
                source = self.archive.rows[case["question_id"]]
                self.assertEqual(case["question"], source["baseline"]["original_question"])
                self.assertEqual(case["canonical_target"], source["baseline"]["gold"])
                for cfg, pred in case["predictions"].items():
                    for key in ["response", "prediction", "correct", "seed", "settings", "generated_tokens"]:
                        self.assertEqual(pred[key], source[cfg][key])
                if case["curation_review"]["reviewer"] == "user":
                    self.assertEqual(case["curation_review"]["notes"], self.human["decisions"][case["case_id"]]["notes"])
            beak = next(c for c in core["cases"] if c["question_id"] == "617")
            self.assertTrue(any("typo" in flag for flag in beak["sensitivity_flags"]))
            binary = next(c for c in core["cases"] if c["question_id"] == "604")
            self.assertTrue(any("25%" in flag for flag in binary["sensitivity_flags"]))
            class Links(HTMLParser):
                def handle_starttag(self, tag, attrs):
                    fields = dict(attrs)
                    if tag == "a":
                        self_target = fields.get("href", "")
                        if self_target:
                            assert (output / self_target).is_file(), self_target
            for filename in ["index.html", "qualitative-gallery.html", "review-audit.html"]:
                page = (output / filename).read_text()
                self.assertNotIn("<select", page)
                self.assertIn("data:image/", page)
                Links().feed(page)
            audit = (output / "review-audit.html").read_text()
            self.assertIn("Human review status: exclude. No written reason supplied", audit)
            saved = digest(output / "index.html")
            with self.assertRaises(FileExistsError):
                build(args)
            self.assertEqual(digest(output / "index.html"), saved)
            with zipfile.ZipFile(output.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                for path in output.rglob("*"):
                    if path.is_file():
                        self.assertEqual(z.read(path.relative_to(output).as_posix()), path.read_bytes())
        self.assertEqual(before, {p: digest(p) for p in before})


if __name__ == "__main__":
    unittest.main()
