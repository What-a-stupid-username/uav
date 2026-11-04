import torch,json
from torchvision.ops import deform_conv2d as original
from uav.deform_conv import deform_conv2d
from pathlib import Path
@torch.inference_mode()
def main():
 torch.manual_seed(4);torch.set_num_threads(4);rows=[]
 for h,w in [(16,24),(64,112)]:
  x=torch.randn(1,128,h,w,device='cuda',dtype=torch.float16)
  o=torch.randn(1,288,h,w,device='cuda',dtype=torch.float16)
  weight=torch.randn(128,128,3,3,device='cuda',dtype=torch.float16)*.01
  bias=torch.randn(128,device='cuda',dtype=torch.float16)*.01
  m=torch.rand(1,144,h,w,device='cuda',dtype=torch.float16)
  a=original(x,o,weight,bias,padding=(1,1),mask=m)
  b=deform_conv2d(x,o,weight,bias,mask=m)
  torch.testing.assert_close(a,b,rtol=0,atol=0)
  error=float((a-b).abs().max());print('eager error',error,flush=True)
  from uav.cuda_graph import GraphReplay
  graph=GraphReplay(lambda x,o,m:deform_conv2d(x,o,weight,bias,mask=m))
  for i in range(3):
   x.normal_();o.normal_();m.uniform_()
   a=deform_conv2d(x,o,weight,bias,mask=m);b=graph(x,o,m)
   torch.testing.assert_close(a,b,rtol=0,atol=0)
  rows.append(dict(size=[h,w],torchvision_max_error=error,graph_bitexact=True))
 print(json.dumps(rows),flush=True);None
if __name__=="__main__":main()
