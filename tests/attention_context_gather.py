"""Joint temporal/spatial gathering preserves token ordering and duplicates."""
import unittest
import torch
from uav.propainter.video.model.modules.sparse_transformer import SparseWindowAttention,context_tokens,selected_window_tokens,window_partition


class ContextGatherTests(unittest.TestCase):
    @torch.inference_mode()
    def test_geometry_cache_retains_more_than_four_shapes(self):
        module=SparseWindowAttention(8,2,(5,9)).eval()
        x=torch.zeros(1,1,5,9,8)
        context_tokens(module,x)
        first=next(iter(module._context_index_cache.values()))
        for height in [10,15,20,25,30,35]:context_tokens(module,torch.zeros(1,1,height,9,8))
        context_tokens(module,x)
        self.assertIs(next(iter(module._context_index_cache.values())),first)
        self.assertEqual(len(module._context_index_cache),7)

    @torch.inference_mode()
    def test_prepared_indices_match_independent_padded_attention(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            module=SparseWindowAttention(32,4,(5,9)).to(device).eval()
            x=torch.randn(1,5,11,19,32,device=device)
            for density in [0.,.1,1.]:
                mask=(torch.rand(1,3,11,19,1,device=device)<density).float()
                prepared=module.prepare_indices(mask,(11,19))
                expected=module(x,mask)
                actual=module(x,mask,prepared_indices=prepared)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    @torch.inference_mode()
    def test_selected_local_windows_match_full_partition(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            packed=torch.arange(2*7*10*18*96,device=device).reshape(2,7,10,18,96).float()
            for x in packed.chunk(3,dim=-1):
                for ids in [[],[0,1,2,3],[3,0,3]]:
                    selected=torch.tensor(ids,device=device,dtype=torch.long)
                    expected=window_partition(x,(5,9),4).view(2,4,4,7,45,8).index_select(1,selected)
                    actual=selected_window_tokens(x,(5,9),4,selected)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    @torch.inference_mode()
    def test_joint_gather_matches_two_stage_on_packed_views(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            module=SparseWindowAttention(32,4,(5,9)).to(device).eval()
            packed=torch.arange(2*7*10*18*96,device=device).reshape(2,7,10,18,96).float()
            for expansion in [(2,4),(0,0)]:
                module.expand_size=expansion
                for x in packed.chunk(3,dim=-1):
                    for selected in [None,torch.tensor([3,0,3],device=device)]:
                        for times in [torch.tensor([0,2,4,6],device=device),torch.tensor([5,1,5],device=device)]:
                            expected=context_tokens(module,x[:,times],selected)
                            actual=context_tokens(module,x,selected,times)
                            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
