"""CPU-only checks of the versioned execution policy; no private data needed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import rq2_frozen_protocol as protocol
from rq2_protocol import load_profile as v1_profile


class FrozenProtocolTests(unittest.TestCase):
    def test_v2_preserves_all_accepted_identities_and_has_distinct_control_policy(self):
        old, new = v1_profile(), protocol.load_profile()
        for field in ('case_ids', 'cases', 'arms', 'generation', 'evaluation', 'lock_sha256'):
            self.assertEqual(old[field], new[field])
        self.assertNotEqual(old['version'], new['version'])
        self.assertEqual(len(new['calibration']['case_ids']), 7)
        self.assertEqual(set(new['controls']['fresh_native_repeats']), set(protocol.REPEAT_CASE_IDS))
        self.assertIn('First unconditional', new['frozen_realization']['primary_capture'])

    def test_tampered_selection_policy_cohort_or_removed_controls_are_rejected(self):
        original = protocol.load_profile()
        def variants():
            changed = copy.deepcopy(original); changed['arms']['native-input288']['input_tokens'] = 96; yield changed
            changed = copy.deepcopy(original); changed['case_ids'].reverse(); yield changed
            changed = copy.deepcopy(original); changed['frozen_realization']['primary_capture'] = 'Choose best answer'; yield changed
            changed = copy.deepcopy(original); changed['controls']['fresh_native_repeats'] = []; yield changed
            changed = copy.deepcopy(original); changed['calibration']['full_run_requires_passed_calibration'] = False; yield changed
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.json'
            for changed in variants():
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): protocol.load_profile(path)

    def test_calibration_includes_same_image_pair_all_datasets_and_repeat_controls(self):
        profile = protocol.load_profile()
        ids = profile['calibration']['case_ids']
        cases = {c['case_id']: c for c in profile['cases']}
        self.assertEqual(cases[ids[0]]['image_sha256'], cases[ids[1]]['image_sha256'])
        self.assertEqual({cid.split(':')[0] for cid in ids}, {'gqa', 'mmstar', 'textvqa'})
        self.assertTrue(set(protocol.REPEAT_CASE_IDS) <= set(ids))
        with self.assertRaises(ValueError): protocol.choose_panel([], calibration='yes')


if __name__ == '__main__': unittest.main()
