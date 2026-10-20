"""CPU scientific-design checks for fixed-selection routing/protection."""
import copy
from collections import Counter
from pathlib import Path
import random
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO/'scripts'))
from rq2_routing_protocol import (CONDITIONS, RANDOM_SEEDS, PROTECTED_COUNT, load_profile,
    routing_plan, assert_routing_identity, summarize)


def reference():
    targets = list(range(278)); sources = list(range(288,576)); destinations = [i % 278 for i in range(288)]
    return {'source_clip':'clip', 'source_patches':'patches', 'pre_merge':'basis-replaced',
        'input':{'errors':[float(576-i) for i in range(576)], 'basis_indices':list(range(10)),
                 'selected_indices':list(range(288))}, 'basis_slot_to_source':[],
        'merge_argument_indices':[list(range(288))], 'merge_basis_tokens':10, 'merge_scaling':1,
        'encoded':{'mask':'same-mask'}, 'input_merge':{'target_indices':targets, 'dropped_indices':sources,
        'assigned_target_indices':destinations, 'merge_counts':[Counter(destinations)[i] for i in targets],
        'basis_slots':list(range(278,288)), 'basis_assignment_slots':list(range(278,288)), 'scaling':1}}


class RoutingProtocolTests(unittest.TestCase):
    def test_profile_fixes_cohort_budget_conditions_and_protection_count(self):
        p = load_profile()
        self.assertEqual(len(CONDITIONS),10); self.assertEqual(len(p['case_ids']),30)
        self.assertEqual(len(p['calibration_case_ids']),7); self.assertEqual(p['protected_count'],28)
        self.assertEqual(p['decoder_layers'],[])

    def test_shuffling_preserves_loads_is_reproducible_and_rng_isolated(self):
        ref = reference(); state = random.getstate(); plans = []
        for seed in RANDOM_SEEDS:
            name = f'shuffled-{seed}'; plan = routing_plan(ref,'image',590,name)
            self.assertEqual(plan,routing_plan(ref,'image',590,name)); plans.append(plan)
            self.assertEqual(Counter(plan['assigned_target_indices']),Counter(ref['input_merge']['assigned_target_indices']))
            self.assertEqual(plan['protected_targets'],[])
        self.assertNotEqual(plans[0]['assigned_target_indices'],plans[1]['assigned_target_indices'])
        self.assertNotEqual(plans[0],routing_plan(ref,'other-image',590,'shuffled-101'))
        self.assertEqual(state,random.getstate())

    def test_protection_is_answer_independent_matched_size_with_index_ties(self):
        ref = reference(); plan = routing_plan(ref,'image',590,'protect-high-E')
        self.assertEqual(plan['protected_targets'],list(range(28)))
        ref['input']['errors'] = [1.]*576
        self.assertEqual(routing_plan(ref,'image',590,'protect-high-E')['protected_targets'],list(range(28)))
        ref['gold_answer'] = 'cannot affect sampling'
        for seed in RANDOM_SEEDS:
            p = routing_plan(ref,'image',590,f'protect-random-{seed}')
            self.assertEqual(len(set(p['protected_targets'])),PROTECTED_COUNT)
            self.assertTrue(set(p['protected_targets']) <= set(ref['input_merge']['target_indices']))
            self.assertFalse(p['uses_gold_or_predictions'])

    def test_controls_reject_count_selection_and_protected_feature_changes(self):
        ref=reference(); plan=routing_plan(ref,'image',590,'shuffled-101'); obs=copy.deepcopy(ref)
        obs['input_merge']['assigned_target_indices']=plan['assigned_target_indices']
        self.assertEqual(assert_routing_identity(ref,obs,plan)['status'],'passed')
        obs['input_merge']['merge_counts'][0]+=1
        with self.assertRaises(ValueError):assert_routing_identity(ref,obs,plan)
        plan=routing_plan(ref,'image',590,'protect-high-E');obs=copy.deepcopy(ref)
        assigned=[28 if t in plan['protected_targets'] else t for t in obs['input_merge']['assigned_target_indices']]
        obs['input_merge']['assigned_target_indices']=assigned
        obs['input_merge']['merge_counts']=[Counter(assigned)[i] for i in obs['input_merge']['target_indices']]
        self.assertEqual(assert_routing_identity(ref,obs,plan,True)['status'],'passed')
        with self.assertRaises(ValueError):assert_routing_identity(ref,obs,plan,False)
        obs['encoded']['mask']='changed'
        with self.assertRaises(ValueError):assert_routing_identity(ref,obs,plan,True)

    def test_invalid_graph_and_nonfinite_errors_fail_before_planning(self):
        for damage in ('counts','errors','sources'):
            ref=reference()
            if damage=='counts':ref['input_merge']['merge_counts'][0]+=1
            elif damage=='errors':ref['input']['errors'][0]=float('nan')
            else:ref['input_merge']['dropped_indices'].reverse()
            with self.assertRaises(ValueError):routing_plan(ref,'image',590,'shuffled-101')

    def test_summary_keeps_unique_question_denominators_and_paired_outcomes(self):
        ids=load_profile()['case_ids'][:2];rows=[]
        for name in CONDITIONS:
            values=[True,True] if name in ('baseline576','no-merge','protect-high-E') else [True,False]
            for cid,correct in zip(ids,values):
                rows.append({'case_id':cid,'condition':name,'response':str(correct),'scoring':{'correct':correct}})
        s=summarize(rows)['gqa'];self.assertEqual(s['unique_questions'],2)
        self.assertEqual(s['effects']['accuracy']['high_E_minus_random_protection_mean'],.5)
        self.assertEqual(s['paired_vs_native']['protect-high-E']['rescues_vs_native'],ids[1:])
        with self.assertRaises(ValueError):summarize(rows+rows[:1])
        with self.assertRaises(ValueError):summarize(rows[:-1])


if __name__=='__main__':unittest.main()
