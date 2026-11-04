"""GPU overlap blending: predictions farther from crop edges weigh more."""
import torch
from functools import lru_cache


def edge_weights(height,width,device):
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    return _edge_weights(height,width,device)


@lru_cache(maxsize=32)
def _edge_weights(height,width,device):
    # Immutable geometry only; callers multiply it into their own weights.
    y=torch.arange(height,device=device,dtype=torch.float32)+.5
    x=torch.arange(width,device=device,dtype=torch.float32)+.5
    return torch.minimum(torch.minimum(y,height-y)[:,None],torch.minimum(x,width-x)[None,:])[None]


class RepairBlend:
    def __init__(self,frames,bounds=None):
        sample=next(iter(frames.values()))
        h,w=sample.shape[-2:]
        self.bounds=(0,0,w,h) if bounds is None else bounds
        l,t,r,b=self.bounds
        if not 0<=l<r<=w or not 0<=t<b<=h:
            raise ValueError('blend bounds must lie inside the original frame')
        self.sums={i:torch.zeros((f.shape[0],b-t,r-l),device=f.device,dtype=torch.float32) for i,f in frames.items()}
        self.weights={i:torch.zeros((1,b-t,r-l),device=sample.device,dtype=torch.float32) for i in frames}

    def add(self,index,patch,mask,crop,weight):
        l,t,r,b=crop
        x,y,z,v=self.bounds
        if not x<=l<r<=z or not y<=t<b<=v:
            raise ValueError('crop must lie inside blend bounds')
        masked_weight=weight*mask
        self.sums[index][:,t-y:b-y,l-x:r-x].add_(patch.float()*masked_weight)
        self.weights[index][:,t-y:b-y,l-x:r-x].add_(masked_weight)

    def finish(self,index,original):
        weight=self.weights.pop(index);summed=self.sums.pop(index)
        result=(summed/weight.clamp_min(.5)).round().clamp_(0,255).to(torch.uint8)
        l,t,r,b=self.bounds
        patch=torch.where(weight>0,result,original[:,t:b,l:r])
        if (l,t,r,b)==(0,0,original.shape[-1],original.shape[-2]):return patch
        output=original.clone()
        output[:,t:b,l:r]=patch
        return output
