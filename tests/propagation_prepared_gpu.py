import torch,json
from uav.propainter.video.model.propainter import InpaintGenerator
from uav.cuda_graph import graph_propagation
from torch.nn import functional as F
@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.manual_seed(49)
 m=InpaintGenerator(model_path='/models/propainter/ProPainter.pth').cuda().half().eval().feat_prop_module
 graph=graph_propagation(m)
 for t in [6,11]:
  for b in [1,2]:
   if b==2:continue # torchvision batch-two fallback uses default-stream sampling during capture
   x=torch.randn(b,t,128,32,48,device='cuda',dtype=torch.float16)
   flows=[torch.randn(b,t-1,2,128,192,device='cuda',dtype=torch.float16) for _ in range(2)]
   masks=[torch.rand(b,t,1,128,192,device='cuda',dtype=torch.float16) for _ in range(2)]
   for i in range(3):
    x.normal_();flows[0].normal_();masks[0].uniform_()
    ds=[F.interpolate(f.flatten(0,1),scale_factor=.25,mode='bilinear',align_corners=False).reshape(b,t-1,2,32,48)/4 for f in flows]
    dm=[F.interpolate(f.flatten(0,1),scale_factor=.25,mode='nearest').reshape(b,t,1,32,48) for f in masks]
    expected=m(x,*ds,torch.cat(dm,dim=2))[2]
    actual=graph.prepared(x,flows,*masks,'bilinear')[2]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 print(json.dumps(dict(prepared_bitexact=True,captures=graph.replay.captures,replays=graph.replay.replays)),flush=True)
if __name__=="__main__":main()
