"""CPU integration checks on the private full sweep and matched layer evidence."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from rq1_protocol import sha256
from summarize_rq1_input_ih_full import FULL, FULL_SHA, LAYER, synthesize


class FullInputIHReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = REPO / 'research/results'
        cls.sources = [results / (FULL + '.zip'), results / (FULL + '.log'), results / LAYER]
        if not all(p.exists() for p in cls.sources):
            raise unittest.SkipTest('Private full input/layer archives unavailable')
        cls.before = [sha256(p) for p in cls.sources]
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name) / 'review'
        cls.result = synthesize(results, cls.output)

    def rows(self, filename):
        with (self.output / filename).open(newline='') as f:
            return list(csv.DictReader(f))

    def test_full_cohort_and_exact_reuse_preserve_raw_archives(self):
        v = json.loads((self.output / 'verification.json').read_text())
        self.assertEqual(v['archive_sha256'], FULL_SHA)
        self.assertEqual((v['questions'], v['images'], v['positions_per_question']), (30, 24, 576))
        self.assertEqual((v['new_retention_scores'], v['reused_zero_scores']), (17280, 17280))
        self.assertEqual(v['exact_zero_audit_repeats'], 90)
        self.assertEqual((v['verified_evidence_files'], v['fresh_feature_files']), (102, 24))
        self.assertTrue(v['source_gate']['passed'])
        self.assertTrue(v['archived_reuse_exact'])
        self.assertEqual(v['max_retention_control_drift'], 0)
        self.assertFalse(v['inference_performed'])
        self.assertFalse(v['tensor_contents_loaded'])
        self.assertEqual(self.before, [sha256(p) for p in self.sources])

    def test_cohort_denominators_and_matched_stage_identity(self):
        cohort = json.loads((self.output / 'cohort-summary.json').read_text())
        ds = {r['group']: r for r in cohort['datasets']}
        self.assertEqual([ds[d]['questions'] for d in ['gqa', 'mmstar', 'textvqa']], [12, 12, 6])
        self.assertEqual(sum(r['rho_E_ih_first_positive'] for r in ds.values()), 11)
        self.assertEqual(sum(r['rho_E_ih_first_negative'] for r in ds.values()), 19)
        # Photo-only filtering changes GQA's sign; don't silently export the primary value.
        self.assertLess(ds['gqa']['rho_E_ih_first_median'], 0)
        self.assertGreater(ds['gqa']['photo_nonbasis_rho_E_ih_first_median'], 0)
        self.assertEqual(len(self.rows('same-image-pairs.csv')), 6)
        self.assertEqual(len(self.rows('depth-comparison.csv')), 6)
        v = json.loads((self.output / 'synthesis-verification.json').read_text())
        self.assertEqual(v['layer_input_zero_scores_repeated_exactly'], 3456)

    def test_full_grid_sparse_tail_and_basis_source_are_not_mislabelled(self):
        calibration = {r['case_id']: r for r in self.rows('sampled-full-comparison.csv')}
        dominion = calibration['textvqa:35670']
        self.assertLess(float(dominion['best_retained_32']), .00002)
        self.assertGreater(float(dominion['best_retained_576']), .20)
        top = [r for r in self.rows('top-retained-tokens.csv')
               if r['case_id'] == 'textvqa:35962' and r['rank'] == '1'][0]
        self.assertEqual(top['token_index'], '149')
        self.assertEqual(top['fps_basis_source'], 'True')
        self.assertEqual(top['E_percentile_nonbasis'], '')
        page = (self.output / 'synthesis.html').read_text()
        payload = page.split('<script type="application/json" id="data">', 1)[1].split('</script>', 1)[0]
        cases = json.loads(payload)
        self.assertEqual(len(cases), 30)
        self.assertEqual(sum(len(c['tokens']) for c in cases), 17280)
        self.assertTrue(all(sum(t[4] for t in c['tokens']) == 10 for c in cases))
        self.assertTrue(all((self.output / c['image']).is_file() for c in cases))


if __name__ == '__main__':
    unittest.main()
