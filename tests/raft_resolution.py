"""Synthetic flow units, unchanged default, and cross-scale cache isolation."""
from types import SimpleNamespace
import torch
from uav.raft_resolution import flow_size,estimate_flow
from uav.propainter_flow_cache import FlowCache
class Raft:
 def __init__(self):
  self.calls=[];self.fix_raft=SimpleNamespace(args=SimpleNamespace(mixed_precision=True))
 def __call__(self,frames,iters):
  self.calls.append(tuple(frames.shape));b,t,_,h,w=frames.shape
  flow=frames.new_ones((b,t-1,2,h,w));flow[:,:,0]*=w/16;flow[:,:,1]*=-h/16
  self.last=(flow,-flow);return self.last
r=Raft();x=torch.zeros(1,4,3,256,384)
assert flow_size(256,384,.5)==(128,192)
assert flow_size(128,256,.5)==(128,128)
assert flow_size(256,384,1)==(256,384)
a=estimate_flow(r,x,8);assert a is r.last
for scale in (.5,.75):
 a,b=estimate_flow(r,x,8,scale)
 assert a.shape==(1,3,2,256,384)
 torch.testing.assert_close(a[:,:,0],torch.full_like(a[:,:,0],24))
 torch.testing.assert_close(a[:,:,1],torch.full_like(a[:,:,1],-16))
 assert torch.equal(a,-b)
cache=FlowCache();before=len(r.calls)
a=cache.estimate(r,x,0,(0,0,384,256),8,.5)
assert len(r.calls)==before+1
b=cache.estimate(r,x,0,(0,0,384,256),8,.5)
assert len(r.calls)==before+1 and torch.equal(a[0],b[0])
cache.estimate(r,x,0,(0,0,384,256),8,1)
assert len(r.calls)==before+2
for scale in (0,-1,1.1,float('nan')):
 try:flow_size(256,384,scale)
 except ValueError:pass
 else:raise AssertionError(scale)
print('PASS: unchanged default, non-square vector units, scale-specific cache and invalid values')
