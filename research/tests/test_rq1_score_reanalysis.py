"""Meaningful scoring/provenance regressions for CPU-only saved-data reanalysis."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from analyze_rq1_scores import PLAN, analyze, effects, load_sources, paired_scores
from rq1_protocol import sha256, spearman


def score(logprobs):
    return {'token_logprobs':logprobs,'sum_logprob':sum(logprobs),'mean_logprob':sum(logprobs)/len(logprobs)}


class ScoreReanalysisTests(unittest.TestCase):
    def test_single_token_identity_and_probability_rank_invariance(self):
        ref=score([math.log(.6)])
        rows=[effects(ref,score([math.log(p)])) for p in [.1,.3,.3,.5,.7,.9]]
        self.assertTrue(all(r['whole_log']==r['first_log'] for r in rows))
        self.assertAlmostEqual(spearman([r['first_log'] for r in rows],[r['first_probability'] for r in rows]),1)
        self.assertAlmostEqual(rows[0]['first_probability'],.5)
        self.assertLess(rows[-1]['first_log'],0)

    def test_multi_token_first_and_whole_effect_can_have_opposite_signs(self):
        ref=score([math.log(.8),math.log(.6)])
        changed=score([math.log(.9),math.log(.1)])
        result=effects(ref,changed)
        self.assertLess(result['first_log'],0)
        self.assertGreater(result['whole_log'],0)
        self.assertAlmostEqual(result['whole_log'],sum(a-b for a,b in zip(ref['token_logprobs'],changed['token_logprobs']))/2)

    def test_missing_length_nonfinite_or_incorrect_saved_mean_rejected(self):
        ref=score([-.1,-.2])
        with self.assertRaises(ValueError): effects(ref,score([-.3]))
        with self.assertRaises(ValueError): effects(ref,score([float('nan'),-.3]))
        bad=score([-.3,-.4]); bad['mean_logprob']+=.1
        with self.assertRaises(ValueError): effects(ref,bad)
        with self.assertRaises(ValueError): effects(score([.2]),score([-.3]))

    def test_wrong_archive_hash_rejected_before_results_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake.zip'; path.write_bytes(b'not-original')
            plan={'sources':{'gqa':{'run':'fake','archive_sha256':'wrong'}}}
            with self.assertRaisesRegex(ValueError,'archive hash'):
                load_sources(Path(directory),plan)

    def test_existing_analysis_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'new derived'):
                analyze(REPO/'research/results',Path(directory))

    def test_real_canonical_results_are_verified_without_inference_or_source_edits(self):
        plan=json.loads(PLAN.read_text()); results=REPO/'research/results'
        paths=[results/(p['run']+'.zip') for p in plan['sources'].values()]
        paths.append(results/(plan['validity_source']['run']+'.zip'))
        if not all(p.exists() for p in paths): self.skipTest('Private source ZIPs unavailable')
        if importlib.util.find_spec('tqdm') is None: self.skipTest('Use the existing editor environment for TextVQA scoring')
        before={p:sha256(p) for p in paths}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new'
            summary=analyze(results,output)
            self.assertEqual((summary['question_count'],summary['image_count'],summary['intervention_count']),(30,24,17280))
            self.assertEqual((summary['one_token_questions'],summary['multi_token_questions']),(21,9))
            self.assertEqual(summary['max_abs_first_log_probability_E_rho_difference_nonbasis'],0)
            self.assertAlmostEqual(summary['cohorts']['textvqa']['first_log']['median_rho_nonbasis'],-.037260829693088623)
            self.assertTrue(all(a['max_control_drift']==0 for a in summary['sources']))
            self.assertTrue(summary['no_inference']); self.assertFalse(summary['tensor_contents_loaded'])
            for relative,digest in summary['analysis_evidence_sha256'].items():
                self.assertEqual(sha256(output/'evidence'/relative),digest)
            self.assertIn('first TOKENIZER PIECE',(output/'index.html').read_text(encoding='utf-8'))
        self.assertEqual({p:sha256(p) for p in paths},before)


if __name__=='__main__': unittest.main()
