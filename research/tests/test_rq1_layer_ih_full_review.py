"""Real full-grid archive checks: rerun prerequisite, nested repeats and tails."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from analyze_rq1_layer_ih_full import analyze, exact_repeat, verify_prerequisite
from rq1_protocol import sha256

ARCHIVE = REPO/'research/results/rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip'
DIGEST = 'e396b7a1c0f0c554e2642ac4e0ac9a83d4f9c4b2a4ecdfbfee6e86bb5a845b37'


class FullReviewTests(unittest.TestCase):
    def test_exact_repeats_reject_missing_positions_or_small_drift(self):
        prior = [{'token_index': 0, 'value': .1}]
        with self.assertRaises(ValueError): exact_repeat([], prior, ['value'], 'repeat')
        with self.assertRaises(ValueError):
            exact_repeat([{'token_index': 0, 'value': .1+1e-14}], prior, ['value'], 'repeat')

    def test_actual_rerun_prerequisite_accepted_but_wrong_hash_rejected(self):
        if not ARCHIVE.exists(): self.skipTest('Private full archive unavailable')
        with zipfile.ZipFile(ARCHIVE) as z:
            report = json.loads(z.read('report.json'))
            self.assertIn('013751203709Z', verify_prerequisite(z, report)['original_output'])
            bad = copy.deepcopy(report); bad['calibration_prerequisite']['report_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'Embedded calibration'):
                verify_prerequisite(z, bad)

    def test_full_grid_preserves_raw_files_and_recovers_unsampled_strong_states(self):
        if not ARCHIVE.exists(): self.skipTest('Private full archive unavailable')
        log = ARCHIVE.with_suffix('.log'); before = sha256(ARCHIVE), sha256(log)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'review'
            v = analyze(ARCHIVE, output, log, DIGEST)
            self.assertEqual(v['total_patch_method_scores'], 10368)
            self.assertEqual(v['verified_zeroing_scores'], 6912)
            self.assertEqual(v['verified_IH_scores'], 3456)
            self.assertEqual(v['exact_zero_calibration_repeats'], 384)
            self.assertEqual(v['exact_IH_calibration_repeats'], 192)
            self.assertEqual(v['exact_canonical_input_repeats'], 3456)
            self.assertEqual(v['verified_evidence_files'], 50)
            self.assertEqual(v['max_control_drift'], 0)
            self.assertFalse(v['inference_performed']); self.assertFalse(v['tensor_contents_loaded'])
            summaries = json.loads((output/'effect-summary.json').read_text())
            dominion = next(r for r in summaries if r['case_id'] == 'textvqa:35670')
            self.assertEqual(dominion['positions'], 576)
            self.assertEqual(dominion['best_token_index'], 131)
            self.assertFalse(dominion['best_in_previous_32'])
            self.assertGreater(dominion['best_retained_probability'], .98)
            self.assertLess(dominion['previous_32_best_retained_probability'], .001)
            self.assertAlmostEqual(dominion['best_E_percentile_nonbasis'], 11.327433628318584)
            page = (output/'full-review.html').read_text(encoding='utf-8')
            self.assertIn('actual padded 336×336', page)
            with self.assertRaises(ValueError): analyze(ARCHIVE, output)
        self.assertEqual(before, (sha256(ARCHIVE), sha256(log)))


if __name__ == '__main__': unittest.main()
