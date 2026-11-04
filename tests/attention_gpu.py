from uav.config import UAVConfig,use_config,get_config
from dataclasses import replace
"""Compare dense and sparse attention for padding, batch, mask and context edges."""
import os
import torch
from uav.propainter.video.model.modules.sparse_transformer import SparseWindowAttention

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(25)
    count=0
    for shape in [(1,5,10,18,32),(2,7,11,19,32)]:
        for expansion in [(4,4),(0,0)]:
            for density in [0.,.02,1.]:
                for sdpa in ['0','1']:
                    module=SparseWindowAttention(32,4,(5,9),pool_size=(4,4)).cuda().half().eval()
                    if expansion==(0,0):module.expand_size=(0,0)
                    x=torch.randn(shape,device='cuda',dtype=torch.float16)
                    mask=(torch.rand(*shape[:4],1,device='cuda')<density).half()
                    for indices in [None,torch.arange(0,shape[1],2,device='cuda')]:
                        scope=use_config(replace(get_config(),propainter_sparse_context=False,propainter_sdpa=sdpa=='1'));scope.__enter__()
                        expected=module(x,mask,indices)
                        scope.__exit__(None,None,None)
                        scope=use_config(replace(get_config(),propainter_sparse_context=True,propainter_sdpa=sdpa=='1'));scope.__enter__()
                        actual=module(x,mask,indices)
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                        count+=1
                        scope.__exit__(None,None,None)
    print(f'{count} dense/sparse attention cases bitexact',flush=True)

if __name__=='__main__':main()
