from uav.config import UAVConfig,use_config,get_config
from dataclasses import replace
"""Check frame-independent FFN specialization on synthetic inputs only."""
import os
import torch
from uav.propainter.video.model.modules.sparse_transformer import FusionFeedForward

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(43)
    params=dict(kernel_size=(7,7),padding=(3,3),stride=(3,3))
    model=FusionFeedForward(32,hidden_dim=98,t2t_params=params).cuda().half().eval()
    checks=0
    for height,width in [(16,24),(24,24)]:
        tokens=((height-1)//3+1)*((width-1)//3+1)
        for batch,frames in [(1,8),(1,11),(2,13)]:
            x=torch.randn(batch,frames*tokens,32,device='cuda',dtype=torch.float16)
            scope=use_config(replace(get_config(),propainter_compile_ffn=False));scope.__enter__()
            expected=model(x,(height,width))
            scope.__exit__(None,None,None)
            actual=model(x,(height,width))
            assert actual.shape==expected.shape
            torch.testing.assert_close(actual,expected,rtol=0,atol=.002)
            checks+=1
    print(f'{checks} compiled FFN shape/batch checks passed',flush=True)

if __name__=='__main__':main()
