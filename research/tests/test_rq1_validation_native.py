"""Kaggle prerequisites: real Torch replacements and tiny native text-only path."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from rq1_validation_native import perturb_patch,generate_text_only


class NativeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Torch checks require the original Kaggle runtime')
        os.environ['MPLBACKEND']='Agg';cls.torch=torch

    def test_replacements_read_original_neighbors_preserve_context_and_shams(self):
        torch=self.torch
        values=torch.arange(580*3,dtype=torch.float32).reshape(1,580,3).half()
        original=values.clone();start=2;i=23
        for method in ['zero','half','neighbor_mean']:
            changed=perturb_patch(values,start,i,method)
            self.assertTrue(torch.equal(values,original))
            self.assertTrue(torch.equal(changed[:,:start+i],original[:,:start+i]))
            self.assertTrue(torch.equal(changed[:,start+i+1:],original[:,start+i+1:]))
            self.assertTrue(torch.equal(perturb_patch(values,start,i,method,strength=0),original))
        expected=values[:,[start+47,start+22],:].float().mean(1).half()
        self.assertTrue(torch.equal(perturb_patch(values,start,i,'neighbor_mean')[:,start+i],expected))
        self.assertTrue(torch.equal(perturb_patch(values,start,i,'half')[:,start+i],values[:,start+i]*.5))
        self.assertTrue((perturb_patch(values,start,i,'zero')[:,start+i]==0).all())

    def test_tiny_native_text_only_generate_has_no_image_positions(self):
        torch=self.torch
        from llava.model.language_model.llava_llama_x import LlavaConfig,LlavaLlamaForCausalLM_X
        config=LlavaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=128,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)
        config._attn_implementation='sdpa'
        model=LlavaLlamaForCausalLM_X(config,visual_token_num=576,selected_indices=[]).eval()
        class Tokenizer:
            bos_token_id=1
            def __call__(self,text):return SimpleNamespace(input_ids=[1,3,4])
            def batch_decode(self,ids,skip_special_tokens=True):return ['A']
        with patch('rq1_validation_native.synchronize'):
            record=generate_text_only(Tokenizer(),model,'Pick A.',590,max_new_tokens=2)
        self.assertEqual(record['visual_tokens'],0)
        self.assertEqual(record['image_placeholders'],0)
        self.assertNotIn('<image>',record['prompt'])
        self.assertEqual(model.model.image_tokens,[0])
        self.assertTrue(1<=record['generated_tokens']<=2)


if __name__=='__main__':unittest.main()
