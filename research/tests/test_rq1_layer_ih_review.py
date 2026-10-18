"""Saved IH arithmetic, geometry, controls and archival integrity; no inference."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from analyze_rq1_layer_ih import analyze, verify_ih
from rq1_protocol import sha256

ARCHIVE=REPO/'research/results/rq1-layer16-ih-calibration-v1-20261018T002129035074Z.zip'
DIGEST='84aa8999e9593f9cd2b436cbce794c205f088083ff2a104b511989e201a40e84'


class IHReviewTests(unittest.TestCase):
    def fixture(self):
        if not ARCHIVE.exists(): self.skipTest('Private IH calibration unavailable')
        with zipfile.ZipFile(ARCHIVE) as z:
            q=json.loads(z.read('questions/gqa-202218829.json'))
            entry=json.loads(z.read('evidence/layer-profile.json'))['cases'][0]
        return q,entry

    def test_wrong_effect_or_probability_is_rejected(self):
        q,e=self.fixture()
        self.assertEqual(verify_ih(q,e,32),0)
        for key in ['ih_first_probability','retained_probability']:
            bad=copy.deepcopy(q);bad['ih']['tokens'][0][key]+=.01
            with self.assertRaises(ValueError):verify_ih(bad,e,32)

    def test_wrong_retention_geometry_and_missing_positions_are_rejected(self):
        q,e=self.fixture()
        bad=copy.deepcopy(q);bad['ih']['tokens'][0]['kept_positions'].pop()
        with self.assertRaises(ValueError):verify_ih(bad,e,32)
        bad=copy.deepcopy(q);bad['ih']['tokens'].pop()
        with self.assertRaises(ValueError):verify_ih(bad,e,32)

    def test_control_drift_and_failed_boundary_are_rejected(self):
        q,e=self.fixture()
        bad=copy.deepcopy(q);bad['ih']['controls']['final_boundary_exact']=False
        with self.assertRaises(ValueError):verify_ih(bad,e,32)
        bad=copy.deepcopy(q);s=bad['ih']['controls']['remove_all_repeat']
        s['token_logprobs'][0]-=.01
        s['sum_logprob']=sum(s['token_logprobs']);s['mean_logprob']=s['sum_logprob']/len(s['token_logprobs'])
        with self.assertRaises(ValueError):verify_ih(bad,e,32)

    def test_overwrite_and_wrong_archive_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):analyze('missing.zip',directory)
            if ARCHIVE.exists():
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    analyze(ARCHIVE,Path(directory)/'new',expected_sha256='wrong')

    def test_complete_saved_review_preserves_originals_and_checks_all_scores(self):
        self.fixture();log=ARCHIVE.with_suffix('.log');before=sha256(ARCHIVE),sha256(log)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'review';result=analyze(ARCHIVE,out,log,DIGEST)
            self.assertEqual(result['new_ih_scores'],192)
            self.assertEqual(result['zeroing_scores_repeated_exactly'],384)
            self.assertEqual(result['verified_evidence_files'],49)
            self.assertEqual(result['verified_feature_file_bytes'],11)
            self.assertEqual(result['max_control_drift'],0)
            self.assertTrue(result['full_run_source_eligibility']['passed'])
            self.assertFalse(result['inference_performed']);self.assertFalse(result['tensor_contents_loaded'])
            rows=json.loads((out/'effect-summary.json').read_text())
            dominion=next(r for r in rows if r['case_id']=='textvqa:35670')
            self.assertGreater(dominion['full_probability'],.94)
            self.assertLess(dominion['best_sampled_retained_probability'],.001)
        self.assertEqual(before,(sha256(ARCHIVE),sha256(log)))


if __name__=='__main__':unittest.main()
