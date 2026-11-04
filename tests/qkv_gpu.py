from uav.config import UAVConfig,use_config,get_config
from dataclasses import replace
import torch,os,json
from pathlib import Path
from uav.propainter.video.model.modules.sparse_transformer import SparseWindowAttention
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.manual_seed(15);rows=[]
 for dim in [32,512]:
  for bias in [True,False]:
   m=SparseWindowAttention(dim,4,(5,9),qkv_bias=bias).cuda().half().eval()
   x=torch.randn(1,13,20,36,dim,device='cuda',dtype=torch.float16)
   for pooled in [False,True]:

    with use_config(replace(get_config(),propainter_pack_qkv=False)):a=m.project(x,pooled)
    b=m.project(x,pooled)
    error=max(float((aa-bb).abs().max()) for aa,bb in zip(a,b));assert error<=.004,error
    row=dict(dim=dim,bias=bias,pooled=pooled,max_error=error)
    for name,enabled in [('separate','0'),('packed','1')]:
     scope=use_config(replace(get_config(),propainter_pack_qkv=enabled=='1'));scope.__enter__()
     for _ in range(3):m.project(x,pooled)
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
     for _ in range(30):m.project(x,pooled)
     end.record();end.synchronize();row[name+'_ms']=start.elapsed_time(end)/30
     scope.__exit__(None,None,None)
    rows.append(row)
   previous=m.project(x)[0].clone();state=m.state_dict();state['query.weight']=state['query.weight']+.1;m.load_state_dict(state);assert not torch.equal(previous,m.project(x)[0])
 print(json.dumps(rows),flush=True);None
if __name__=="__main__":main()
