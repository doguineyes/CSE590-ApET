"""Development-only identity, image shards and clustered inference tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import rq2_staged_gqa_protocol as p
from rq2_staged_protocol import CONDITIONS

def primary_rows(ids=None):
    profile = p.load_profile()
    ids = profile['case_ids'] if ids is None else ids
    return [dict(case_id=c, condition=condition, image_id=profile['case_image_ids'][c], response='yes', scoring={'correct': True}, control_only=False, types=dict(structural='query', semantic='attr', detailed='color'), generated_tokens=2) for c in ids for condition in CONDITIONS]

class GQAStagedProtocolTests(unittest.TestCase):

    def test_frozen800_disjoint_shards_and_distributed_identity_controls(self):
        profile = p.load_profile()
        parts = [p.selected_ids(profile, [i]) for i in range(4)]
        self.assertEqual([len(x) for x in parts], [200] * 4)
        self.assertEqual([cid for part in parts for cid in part], profile['case_ids'])
        for i, part in enumerate(parts):
            self.assertTrue(set(profile['identity_case_ids']) & set(part))
            self.assertEqual(len({profile['case_image_ids'][c] for c in part}), 25)
        self.assertIn('gqa:2058428', profile['case_ids'])
        self.assertEqual(profile['role'], 'development')
        for bad in ([], [0, 0], [2, 1], [4], [-1], [True]):
            with self.assertRaises(ValueError):
                p.selected_ids(profile, bad)

    def test_evaluation_or_edited_data_refused_before_loading_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text('{"study":{"role":"evaluation"}}')
            with self.assertRaisesRegex(ValueError, 'development800 manifest'):
                p.validate_inputs(path)
        profile = p.load_profile()
        profile['role'] = 'evaluation'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, 'Frozen GQA'):
                p.load_profile(path)

    def test_adapter_preserves_image_seed_question_types_and_short_answer_instruction(self):
        manifest = dict(questions=[dict(id='2058428', imageId='image', question='Is there a plane?', answer='yes', types=dict(structural='verify', semantic='obj', detailed='exist'))], images={'image': dict(file='images/image.jpg', image_size=[5, 7], sha256='a' * 64, compression_seed=123)})
        case = p.adapt_manifest(manifest)['cases'][0]
        self.assertEqual(case['compression_seed'], 123)
        self.assertEqual(case['generation_question'], 'Is there a plane?\n' + p.INSTRUCTION)
        self.assertTrue(case['known_repeatability_case'])
        self.assertEqual(case['types'], manifest['questions'][0]['types'])
        self.assertEqual(p.adapt_manifest(manifest)['images']['image']['size'], [5, 7])

    def test_cluster_resampling_uses_images_instead_of16_independent_questions(self):
        ids = p.load_profile()['case_ids'][:16]
        image_for = p.load_profile()['case_image_ids']
        left = {c: 1.0 if i < 8 else 0.0 for i, c in enumerate(ids)}
        right = {c: 0.0 if i < 8 else 1.0 for i, c in enumerate(ids)}
        result = p.clustered_difference(left, right, ids, image_for, intervals=True)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['questions'], 16)
        self.assertEqual(result['mean_accuracy_delta'], 0.0)
        self.assertEqual(result['paired_image_bootstrap_95_ci'], [-1.0, 1.0])

    def test_seed_means_APR_types_and_whole_image_completeness(self):
        ids = p.load_profile()['case_ids'][:8]
        rows = primary_rows(ids)
        for r in rows:
            if r['condition'] == 'pure-random-96-to-32-s101':
                r['scoring']['correct'] = False
        s = p.summarize(rows, expected_ids=ids)
        self.assertEqual(s['conditions']['pure-random-96-to-32-s101']['APR'], 0.0)
        pair = s['paired_comparisons']['pure-random-96-to-32-seed-mean minus native-96-to-32']
        self.assertAlmostEqual(pair['mean_accuracy_delta'], -1 / 3)
        self.assertEqual(pair['images'], 1)
        self.assertEqual(pair['questions'], 8)
        self.assertEqual(s['type_summaries']['structural']['query']['questions'], 8)
        self.assertFalse(s['full_development800'])
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], 'control_only': True}]):
            with self.assertRaises(ValueError):
                p.summarize(bad)
        with self.assertRaisesRegex(ValueError, 'eight-question'):
            p.summarize(primary_rows(ids[:7]))

    def test_baseline_wrong_questions_stay_in_denominator_and_zero_APR_is_undefined(self):
        rows = primary_rows(p.load_profile()['case_ids'][:8])
        for r in rows:
            if r['condition'] == 'baseline576':
                r['scoring']['correct'] = False
        s = p.summarize(rows)
        self.assertEqual(s['questions'], 8)
        self.assertIsNone(s['conditions']['native-96-to-32']['APR'])
        self.assertEqual(len(s['conditions']['native-96-to-32']['gains']), 8)
if __name__ == '__main__':
    unittest.main()
