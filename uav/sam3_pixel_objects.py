"""Build disjoint tracking objects from mask support, independent of SAM boxes."""
import cv2
import time
import torch
import torch.nn.functional as F


def pixel_objects(mask,scale=4,radius=2,profile=None):
    """Cluster coarse occupancy; retain every original foreground pixel exactly.

    CUDA masks use GPU pooling, dilation, connected components and bounds.
    Only component ids and box coordinates cross to CPU. Dilated pixels
    determine membership only and never enter output masks or their bounds.
    """
    if mask.ndim!=2 or mask.dtype!=torch.bool:raise ValueError('expected 2D bool mask')
    if scale<1 or radius<0:raise ValueError('invalid clustering scale/radius')
    h,w=mask.shape
    marks=[];host=[];begin=time.monotonic()
    def mark():
        host.append(time.monotonic())
        if profile is not None and mask.is_cuda:
            event=torch.cuda.Event(enable_timing=True);event.record();marks.append(event)
    mark()
    pooled=F.max_pool2d(mask[None,None].float(),scale,stride=scale,ceil_mode=True)
    joined=F.max_pool2d(pooled,2*radius+1,stride=1,padding=radius)
    mark()
    if mask.is_cuda:
        from .gpu_components import components
        device_labels=components(joined[0,0]>0)
        mark()
        active=pooled[0,0]>0
        ids=torch.unique(device_labels[active])
        gh,gw=device_labels.shape
        yy=torch.arange(gh,device=mask.device)[:,None].expand(gh,gw)
        xx=torch.arange(gw,device=mask.device)[None,:].expand(gh,gw)
        statistics=[]
        for label in ids.unbind():
            member=(device_labels==label)&active
            statistics.append(torch.stack([label,torch.where(member,xx,gw).amin(),
                torch.where(member,yy,gh).amin(),torch.where(member,xx+1,0).amax(),
                torch.where(member,yy+1,0).amax()]))
        regions=[] if not statistics else torch.stack(statistics).cpu().tolist()
        mark()
    else:
        count,labels,stats,_=cv2.connectedComponentsWithStats(joined[0,0].to(torch.uint8).numpy(),8)
        device_labels=torch.as_tensor(labels)
        regions=[[label,int(x),int(y),int(x+cw),int(y+ch)] for label,(x,y,cw,ch,_) in enumerate(stats) if label]
    bounds_list=[];offsets=[];objects={}
    for label,x,y,x1,y1 in regions:
        cw,ch=x1-x,y1-y
        l,t=x*scale,y*scale;r,b=min(w,(x+cw)*scale),min(h,(y+ch)*scale)
        membership=(device_labels[y:y+ch,x:x+cw]==label).repeat_interleave(scale,0).repeat_interleave(scale,1)
        local=mask[t:b,l:r] & membership[:b-t,:r-l]
        rows=local.any(1);cols=local.any(0)
        ys=torch.arange(b-t,device=mask.device);xs=torch.arange(r-l,device=mask.device)
        bounds_list.append(torch.stack([torch.where(cols,xs,r-l).amin(),torch.where(rows,ys,b-t).amin(),
                            torch.where(cols,xs+1,0).amax(),torch.where(rows,ys+1,0).amax()]))
        offsets.append((l,t))
        full=torch.zeros_like(mask);full[t:b,l:r]=local
        objects[len(objects)]=full
    mark()
    boxes=[]
    if bounds_list:
        for (x0,y0,x1,y1),(l,t) in zip(torch.stack(bounds_list).cpu().tolist(),offsets):
            boxes.append([l+x0,t+y0,l+x1,t+y1])
    mark()
    if profile is not None:
        profile.append(dict(events=marks,host=host,seconds=time.monotonic()-begin,
                            objects=len(objects),shape=(h,w)))
    return boxes,objects
