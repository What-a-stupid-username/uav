from uav.config import UAVConfig,use_config,get_config
from dataclasses import replace
import os,torch,json
from pathlib import Path
from uav.propainter.video.model.propainter import InpaintGenerator
from uav.cuda_graph import graph_propagation
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.manual_seed(47)
 m=InpaintGenerator(model_path='/models/propainter/ProPainter.pth').cuda().half().eval().feat_prop_module
 x=torch.randn(1,6,128,32,48,device='cuda',dtype=torch.float16)
 f=torch.randn(1,5,2,32,48,device='cuda',dtype=torch.float16)*.5
 mask=torch.rand(1,6,2,32,48,device='cuda',dtype=torch.float16)
 args=(x,f,-f,mask)

 with use_config(replace(get_config(),propainter_stream_deform=False)):original=m(*args)
 eager=m(*args)
 for a,b in zip(original,eager):
  if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
 row=dict(original_max_error=max(float((a-b).abs().max()) for a,b in zip(original,eager) if torch.is_tensor(a)))
 graph=graph_propagation(m)
 for i in range(3):
  x.normal_();f.normal_(std=.5);mask.uniform_()
  expected=m(*args);actual=graph(*args)
  for a,b in zip(expected,actual):
   if torch.is_tensor(b):torch.testing.assert_close(a,b,rtol=0,atol=0)
 row['graph_bitexact']=True
 for name,fn in [('eager',m.forward),('graph',graph)]:
  for _ in range(2):fn(*args)
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(5):fn(*args)
  b.record();b.synchronize();row[name+'_ms']=a.elapsed_time(b)/5
 print(json.dumps(row),flush=True);None
if __name__=="__main__":main()
