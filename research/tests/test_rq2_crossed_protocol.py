"""CPU-only scientific design checks; no tensor/runtime dependencies."""
import copy
from pathlib import Path
import random
import sys
import unittest

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'scripts'))
from rq2_crossed_protocol import (load_profile,random_indices,assert_selector_identity,
                                  CONDITIONS,RANDOM_SEEDS,summarize)


def reference():
    return {'source_clip':'clip','source_patches':'patches','pre_merge':'basis-replaced',
        'input':{'errors':[float(576-i) for i in range(576)],'basis_indices':list(range(10)),
                 'selected_indices':list(range(288))},
        'basis_slot_to_source':[{'source_index':i,'assignment_slot':278+i,'compact_output_position':278+i} for i in range(10)]}


class CrossedProtocolTests(unittest.TestCase):
    def test_profile_freezes_nine_conditions_three_seeds_and_unchanged_cohort(self):
        p=load_profile();self.assertEqual(len(CONDITIONS),9);self.assertEqual(len(p['case_ids']),30)
        self.assertEqual(p['random_seeds'],[101,202,303]);self.assertEqual(len(p['calibration_case_ids']),7)
        self.assertEqual(p['oracle']['target'],317)

    def test_random_selection_preserves_tail_and_is_isolated_from_global_rng(self):
        state=random.getstate();ref=reference();sets=[]
        for seed in RANDOM_SEEDS:
            indices,meta=random_indices(ref,'image-sha',590,seed)
            self.assertEqual(len(indices),len(set(indices)));self.assertEqual(len(indices),288)
            self.assertEqual(indices[-10:],ref['input']['selected_indices'][-10:])
            self.assertFalse(set(indices[:-10]) & set(indices[-10:]));self.assertEqual(meta['pool_size'],566)
            self.assertEqual(indices,random_indices(ref,'image-sha',590,seed)[0]);sets.append(set(indices))
        self.assertNotEqual(sets[0],sets[1]);self.assertEqual(state,random.getstate())
        self.assertNotEqual(random_indices(ref,'image-sha',590,101)[0],random_indices(ref,'other-image',590,101)[0])
        with self.assertRaises(ValueError):random_indices(ref,'image-sha',590,999)

    def test_compact_basis_positions_may_change_but_spatial_source_assignment_may_not(self):
        ref=reference();obs=copy.deepcopy(ref);ranked,_=random_indices(ref,'image-sha',590,101)
        obs['input']['selected_indices']=ranked
        compact={v:i for i,v in enumerate(sorted(ranked))}
        for b in obs['basis_slot_to_source']:b['compact_output_position']=compact[b['assignment_slot']]
        self.assertEqual(assert_selector_identity(ref,obs,ranked)['status'],'passed')
        obs['basis_slot_to_source'][0]['assignment_slot']+=1
        with self.assertRaises(ValueError):assert_selector_identity(ref,obs,ranked)

    def test_summaries_average_seed_effects_without_pooling_question_denominators(self):
        ids=load_profile()['case_ids'][:2];rows=[]
        for name in CONDITIONS:
            values={'baseline576':[1,1],'E-merge':[1,0],'E-no-merge':[1,1],
                'random-101-merge':[0,0],'random-101-no-merge':[1,0],
                'random-202-merge':[1,0],'random-202-no-merge':[1,1],
                'random-303-merge':[1,1],'random-303-no-merge':[1,1]}[name]
            for cid,correct in zip(ids,values):rows.append({'case_id':cid,'condition':name,'scoring':{'correct':bool(correct)}})
        s=summarize(rows)['gqa'];self.assertEqual(s['unique_questions'],2)
        effects=s['effects']['accuracy'];self.assertEqual(effects['random_merge_seed_values'],[0,.5,1])
        self.assertAlmostEqual(effects['E_minus_random_no_merge'],1-5/6)
        self.assertAlmostEqual(effects['interaction'],-.5-(-1/3))
        with self.assertRaises(ValueError):summarize(rows+rows[:1])
        with self.assertRaises(ValueError):summarize(rows[:-1])


if __name__=='__main__':unittest.main()
