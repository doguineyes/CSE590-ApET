"""Geometry, matching ambiguity, lineage, and restricted tensor-loader checks."""
import io
from pathlib import Path
import pickle
import sys
import unittest
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from textvqa_patch_properties import canvas_box,patch_overlap,padding_fraction,detail_energy,source_paths,spearman
from read_cpu_tensor_archive import rebuild,read_tensors
from analyze_textvqa_evidence import match_answer

class EvidenceTests(unittest.TestCase):
    def test_landscape_and_odd_padding(self):
        b=dict(top_left_x=0,top_left_y=0,width=1,height=1)
        self.assertEqual(canvas_box(b,672,336),[0,84,336,252])
        self.assertAlmostEqual(float(padding_fraction(672,336).mean()),.5)
        self.assertTrue(np.allclose(padding_fraction(336,336),0))
        self.assertAlmostEqual(canvas_box(b,101,100)[1],0)
        self.assertAlmostEqual(float(padding_fraction(101,100).mean()),1/101)

    def test_patch_boundary_overlap(self):
        a=patch_overlap([0,0,14,14]);self.assertEqual(np.flatnonzero(a).tolist(),[0]);self.assertEqual(a[0],1)
        a=patch_overlap([7,0,21,14]);self.assertEqual(a[:2].tolist(),[.5,.5])

    def test_detail_ignores_patch_border_edges(self):
        p=np.zeros((1,3,336,336));p[:,:,:,14:]=1
        self.assertTrue(np.all(detail_energy(p)==0))
        p[:,:,:,7:14]=1
        self.assertTrue(detail_energy(p)[0]>0)

    def test_matching_is_conservative(self):
        def word(s,x):return dict(word=s,bounding_box=dict(top_left_x=x,top_left_y=.2,width=.1,height=.1))
        c=dict(gold_answers=['brand']*10)
        m,_=match_answer(c,dict(ocr_info=[word('brand',.1)]),str.lower,(336,336))
        self.assertEqual(m['status'],'unique_candidate')
        m,_=match_answer(c,dict(ocr_info=[word('brand',.1),word('brand',.5)]),str.lower,(336,336))
        self.assertEqual(m['status'],'ambiguous_multiple_matches')
        m,_=match_answer(c,dict(ocr_info=[word('brond',.1)]),str.lower,(336,336))
        self.assertEqual(m['status'],'unmatched')
        c['gold_answers']=['new york']*10
        m,_=match_answer(c,dict(ocr_info=[word('new',.1),word('york',.2)]),str.lower,(336,336))
        self.assertEqual(m['status'],'multiword_review')

    def test_basis_replacement_and_merge_paths(self):
        d,a=source_paths([0,2],[(2,3)],dict(dropped_indices=[1,3],assigned_target_indices=[0,0]),[{i} for i in range(4)])
        self.assertEqual(d,{0:{0},2:{3}});self.assertEqual(a,{0:{0,1,3},2:{3}})
        self.assertNotIn(2,set().union(*a.values()))

    def test_rank_ties_and_constant(self):
        self.assertEqual(spearman([1,1,2],[1,1,2]),1)
        self.assertIsNone(spearman([1,1],[1,2]))

    def test_strided_tensor_and_bounds(self):
        a=np.arange(8,dtype='<f2')
        np.testing.assert_array_equal(rebuild(a,1,(2,2),(4,1),False,{}),[[1,2],[5,6]])
        with self.assertRaises(ValueError):rebuild(a,3,(2,2),(4,1),False,{})

    def test_rejects_executable_pickle(self):
        import os
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            z.writestr('t/byteorder',b'little');z.writestr('t/data.pkl',pickle.dumps(os.system))
        with self.assertRaisesRegex(ValueError,'Forbidden pickle global'):read_tensors(b.getvalue())

if __name__=='__main__':unittest.main()
