"""Geometry/marker safeguards for review figures; no plotting packages needed."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from render_rq1_gallery import marker_choices, patch_regions, photo_bounds


class GalleryTests(unittest.TestCase):
    def test_landscape_padding_and_patch_locations(self):
        bounds=photo_bounds((640,480))
        self.assertEqual(bounds,[0,42,336,294])
        regions=patch_regions(bounds)
        self.assertEqual(sum(r['region']=='padding' for r in regions.values()),144)
        self.assertEqual(sum(r['region']=='photo' for r in regions.values()),432)
        self.assertEqual(regions[0]['region'],'padding')
        self.assertEqual(regions[72]['region'],'photo')
        self.assertEqual(regions[335]['region'],'photo')

    def test_portrait_mixed_edges_and_conserved_area(self):
        regions=patch_regions(photo_bounds((300,500)))
        counts={name:sum(r['region']==name for r in regions.values()) for name in ['photo','mixed','padding']}
        self.assertEqual(counts,{'photo':336,'mixed':48,'padding':192})
        self.assertAlmostEqual(sum(r['photo_fraction'] for r in regions.values()),576*.6)
        self.assertAlmostEqual(regions[4]['photo_fraction'],.2)

    def test_square_and_integer_centering_follow_native_padding(self):
        self.assertEqual(photo_bounds((4,3)),[0,0,336,252])
        self.assertTrue(all(r['region']=='photo' for r in patch_regions(photo_bounds((500,500))).values()))
        with self.assertRaises(ValueError):photo_bounds((0,300))

    def test_markers_separate_padding_and_exclude_basis_from_high_E(self):
        regions=patch_regions(photo_bounds((640,480)))
        tokens=[{'token_index':i,'reconstruction_error':0,'intervention_score':-1} for i in range(576)]
        for i,error,effect in [(72,999,.1),(73,50,.2),(74,99,.3),(75,100,.01),(0,0,1000),(1,0,999)]:
            tokens[i]['reconstruction_error']=error;tokens[i]['intervention_score']=effect
        selected=marker_choices(tokens,regions,{72})
        self.assertEqual(selected,{0:['I'],73:['I'],74:['E','I'],75:['E']})
        self.assertNotIn(1,selected)
        self.assertNotIn(72,selected)


if __name__=='__main__':unittest.main()
