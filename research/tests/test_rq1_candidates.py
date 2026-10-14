"""CPU checks for scientific selection, provenance and review safety."""

from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from rq1_candidates import (Archive, CONFIGURATIONS, EXCLUDED_GQA_IMAGES, digest,
                            save_image, select_gqa, select_mmstar, validate_manifest, write_manifest)
from prepare_rq1_candidates import (GQA_ARCHIVE, MMSTAR_ARCHIVE, prepare_archived,
                                   select_textvqa, skeleton, textvqa_case)


def text_rows(count=120):
    return [{"question_id": 2 * i + j, "image_id": f"image{i}",
             "question": f"What does sign {i} say?", "answers": ["alpha", "beta"] * 5,
             "set_name": "val", "image_width": 3, "image_height": 2}
            for i in range(count) for j in range(2)]


class CandidateTests(unittest.TestCase):
    def test_text_pool_is_distinct_nested_and_order_independent(self):
        rows = text_rows()
        picked = select_textvqa(rows, 590)
        self.assertEqual(picked, select_textvqa(list(reversed(rows)), 590))
        self.assertEqual(len({r["image_id"] for r in picked}), 100)
        self.assertNotEqual(picked, select_textvqa(rows, 591))
        cases = [textvqa_case(r, True) for r in picked[:30]]
        self.assertEqual(len(cases), 30)
        for case in cases:
            self.assertEqual(case["canonical_target"], "alpha")
            self.assertEqual(len(case["gold_answers"]), 10)
            self.assertTrue(all(v is None for v in case["predictions"].values()))
            self.assertEqual(set(case["outcomes"].values()), {"unmeasured"})
        invalid = text_rows()
        invalid[0]["answers"] = ["alpha"]
        with self.assertRaises(ValueError):
            select_textvqa(invalid, 590)

    def test_mmstar_unavailable_gain_is_explicit_not_fabricated(self):
        rows = {}
        for category in ["coarse", "fine", "instance", "logical", "math", "science"]:
            for i in range(30):
                qid = f"{category}-{i}"
                baseline = {"category": category, "correct": i != 1, "generated_tokens": 1}
                compressed = {**baseline, "correct": i != 0 and (i != 1 or category != "math")}
                rows[qid] = {cfg: dict(baseline if cfg == "baseline" else compressed) for cfg in CONFIGURATIONS}
        picked = select_mmstar(rows, 590)
        self.assertEqual(picked, select_mmstar(dict(reversed(list(rows.items()))), 590))
        self.assertEqual(len(set(p[0] for p in picked)), 30)
        self.assertEqual(Counter(rows[p[0]]["baseline"]["category"] for p in picked),
                         {c: 5 for c in ["coarse", "fine", "instance", "logical", "math", "science"]})
        fallback = next(p for p in picked if p[0].startswith("math-") and "gain unavailable" in p[2])
        self.assertTrue(rows[fallback[0]]["baseline"]["correct"])
        self.assertTrue(rows[fallback[0]]["input-only"]["correct"])

    def test_images_and_gallery_preserve_identity_escape_text_and_pending_scores(self):
        from PIL import Image

        stream = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
        raw = stream.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            meta = save_image(raw, directory)
            self.assertEqual((directory / meta["file"]).read_bytes(), raw)
            self.assertEqual(meta["sha256"], hashlib.sha256(raw).hexdigest())
            with self.assertRaises(ValueError):
                save_image(raw, directory, expected_pixels="incorrect")
            row = text_rows(1)[0]
            row["question"] = "<script>question & answer</script>"
            case = textvqa_case(row, True)
            case["image_sha256"] = meta["sha256"]
            manifest = skeleton("textvqa", 590, "provisional", {})
            manifest["cases"], manifest["images"] = [case], {row["image_id"]: meta}
            path = write_manifest(directory, manifest)
            gallery = (directory / "gallery.html").read_text()
            self.assertNotIn("<script>question", gallery)
            self.assertIn("&lt;script&gt;question &amp; answer", gallery)
            self.assertIn("Not measured", gallery)
            self.assertIn(digest(path), gallery)
            self.assertIn("data:image/png;base64,", gallery)
            self.assertEqual(gallery.count("data-reviewcase="), 1)
            bad = copy.deepcopy(manifest)
            bad["cases"][0]["image_sha256"] = "changed"
            with self.assertRaises(ValueError):
                validate_manifest(bad, directory)
            bad = copy.deepcopy(manifest)
            bad["images"][row["image_id"]]["file"] = "../outside.png"
            with self.assertRaises(ValueError):
                validate_manifest(bad, directory)

    @unittest.skipUnless((ROOT / GQA_ARCHIVE).exists() and (ROOT / MMSTAR_ARCHIVE).exists(), "Local result archives absent")
    def test_saved_archive_selection_controls_exclusion_and_unchanged_sources(self):
        original = {p: digest(ROOT / p) for p in [GQA_ARCHIVE, MMSTAR_ARCHIVE]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            gqa = prepare_archived(ROOT / GQA_ARCHIVE, "gqa", 590, directory / "gqa")
            mmstar = prepare_archived(ROOT / MMSTAR_ARCHIVE, "mmstar", 590, directory / "mmstar")
            self.assertEqual(gqa["counts"]["questions"], 60)
            self.assertEqual(gqa["counts"]["images"], 20)
            self.assertEqual(gqa["counts"]["strata"], {"metadata_reference": 30, "outcome_diagnostic": 30})
            self.assertFalse(set(gqa["images"]) & set(EXCLUDED_GQA_IMAGES))
            for image in gqa["images"]:
                cases = [c for c in gqa["cases"] if c["image_id"] == image]
                self.assertEqual(len({c["question"] for c in cases}), 3)
            by_id = {c["question_id"]: c for c in gqa["cases"]}
            for case in gqa["cases"]:
                for control in case["same_image_preserved_controls"]:
                    self.assertEqual(case["image_id"], by_id[control]["image_id"])
                    self.assertEqual(by_id[control]["outcomes"]["input-only"], "preserved_correct")
                if case["selection"]["stratum"] == "outcome_diagnostic" and case["outcomes"]["input-only"] in {"loss", "gain"}:
                    self.assertTrue(case["same_image_preserved_controls"])
            gallery = (directory / "gqa/gallery.html").read_text()
            self.assertEqual(gallery.count("data-reviewcase="), 60)
            self.assertTrue(gqa["context_questions"])
            self.assertEqual(mmstar["counts"]["questions"], 30)
            self.assertEqual(set(mmstar["counts"]["categories"].values()), {5})
            self.assertFalse(mmstar["source"]["recorded_prediction_hash_verified"])
            archive = Archive(ROOT / GQA_ARCHIVE, "gqa")
            try:
                selected = select_gqa(archive.rows, 590)
                self.assertEqual(selected, select_gqa(dict(reversed(list(archive.rows.items()))), 590))
            finally:
                archive.close()
        self.assertEqual(original, {p: digest(ROOT / p) for p in original})


if __name__ == "__main__":
    unittest.main()
