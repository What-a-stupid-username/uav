"""Check source-pixel units, conservative small masks, and exact known flow."""
import torch
from uav.flow_completion_resolution import complete_flow
class Completion:
 def forward_bidirect_flow(self,flows,masks):
  self.size=flows[0].shape[-2:];self.masks=masks
  h,w=self.size
  prediction=[]
  for flow in flows:
   value=torch.ones_like(flow);value[:,:,0]*=w/16;value[:,:,1]*=-h/16
   prediction.append(value)
  return prediction,None
 def combine_flow(self,flows,prediction,masks):
  return tuple(p*m+f*(1-m) for f,p,m in zip(flows,prediction,(masks[:,:-1],masks[:,1:])))
for device in ('cpu','cuda'):
 model=Completion();flows=tuple(torch.randn(1,3,2,256,384,device=device) for _ in range(2))
 masks=torch.zeros(1,4,1,256,384,device=device);masks[:,:,:,61:115,91:180]=1;masks[:,:,:,201,301]=1
 for scale in (1,.5):
  result=complete_flow(model,flows,masks,scale)
  assert model.size==((256,384) if scale==1 else (128,192))
  assert model.masks[0,0,0,100 if scale==.5 else 201,150 if scale==.5 else 301]==1
  for flow,out,m in zip(flows,result,(masks[:,:-1],masks[:,1:])):
   known=(m==0).expand_as(flow);assert torch.equal(flow[known],out[known])
   for c,v in ((0,24),(1,-16)):
    torch.testing.assert_close(out[:,:,c][m[:,:,0].bool()],torch.full_like(out[:,:,c][m[:,:,0].bool()],v))
print('PASS: CPU/GPU flow units, conservative masks, exact known pixels and baseline path')
