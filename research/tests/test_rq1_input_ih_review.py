"""CPU archive review: real calibration, preservation and source-gate rejection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from analyze_rq1_input_ih import analyze
from rq1_protocol import sha256

ARCHIVE=REPO/'research/results/rq1-input-ih-calibration-v1-20261018T135403155797Z.zip'
DIGEST='05d1fc5c78682d94ab645a386f1dde7eb23f8e51fa72f3e70714dbf22c2a5a8c'


class InputIHReviewTests(unittest.TestCase):
    def setUp(self):
        if not ARCHIVE.exists():self.skipTest('Private calibration unavailable')

    def test_complete_review_verifies_raw_scores_and_preserves_originals(self):
        log=ARCHIVE.with_suffix('.log');before=sha256(ARCHIVE),sha256(log)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'review';v=analyze(ARCHIVE,out,log,DIGEST)
            self.assertEqual((v['questions'],v['images'],v['positions_per_question']),(6,5,32))
            self.assertEqual(v['new_retention_scores'],192);self.assertEqual(v['exact_zero_audit_repeats'],18)
            self.assertEqual(v['verified_evidence_files'],95);self.assertEqual(v['fresh_feature_files'],5)
            self.assertTrue(v['source_gate']['passed']);self.assertTrue(v['archived_reuse_exact'])
            self.assertEqual(v['max_retention_control_drift'],0)
            self.assertFalse(v['inference_performed']);self.assertFalse(v['tensor_contents_loaded'])
            rows=json.loads((out/'effect-summary.json').read_text())
            dominion=next(r for r in rows if r['case_id']=='textvqa:35670')
            self.assertGreater(dominion['rho_E_ih_first'],.53)
            self.assertLess(dominion['median_absolute_IH'],1e-6)
            self.assertLess(dominion['best_measured_retained_probability'],.00002)
            self.assertAlmostEqual(v['full_timing_linear_minutes'],27.010118706649683)
        self.assertEqual(before,(sha256(ARCHIVE),sha256(log)))

    def test_wrong_hash_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):analyze(ARCHIVE,directory)
            with self.assertRaisesRegex(ValueError,'SHA256'):
                analyze(ARCHIVE,Path(directory)/'new',expected_sha256='wrong')

    def test_report_source_mismatch_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bad=root/'bad.zip'
            with zipfile.ZipFile(ARCHIVE) as src,zipfile.ZipFile(bad,'w',zipfile.ZIP_DEFLATED) as dst:
                for name in src.namelist():
                    raw=src.read(name)
                    if name=='report.json':
                        r=json.loads(raw);r['measurement_sources_sha256']['scripts/rq1_input_ih_native.py']='changed'
                        raw=json.dumps(r).encode()
                    dst.writestr(name,raw)
            with self.assertRaisesRegex(ValueError,'measurement source'):
                analyze(bad,root/'review')
            self.assertFalse((root/'review').exists())


if __name__=='__main__':unittest.main()
