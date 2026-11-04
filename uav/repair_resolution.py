"""Bounded scaled-input preparation and original-resolution repair compositing."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .repair_prefetch import repair_boxes


def repair_size(width, height, scale):
    if not 0 < scale <= 1:
        raise ValueError('repair scale must be in (0, 1]')
    if scale == 1:
        return width, height
    return tuple(min(extent, max(8, round(extent*scale/8)*8)) for extent in (width,height))


def restore_rgb(rgb, original, mask):
    """Use global half-pixel coordinates; original mask alone owns output pixels."""
    restored=F.interpolate(rgb.permute(2,0,1)[None].float(),size=original.shape[:2],
                           mode='bilinear',align_corners=False)[0].permute(1,2,0)
    restored=restored.round().clamp(0,255).to(torch.uint8)
    return torch.where(mask[:,:,None],restored,original)


@lru_cache(maxsize=1)
def _compiled_restore():
    return torch.compile(restore_rgb,fullgraph=True,dynamic=False,
                         options={'emulate_precision_casts':True})


class RepairInputs:
    """One bounded CPU producer; mask callbacks are consumed in timeline order."""
    def __init__(self,frames,mask_at,size,capacity=8):
        if capacity < 1:
            raise ValueError('preparation capacity must be positive')
        self.frames,self.mask_at,self.size=frames,mask_at,size
        self.height,self.width=frames(0).shape[:2]
        self.capacity=capacity
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='uav-repair-input')
        self.pending=deque()

    def _prepare(self,index):
        image=self.frames(index)
        mask=self.mask_at(index)
        if mask.dtype != np.bool_ or mask.shape != (self.height,self.width):
            raise ValueError('repair requires a final boolean mask at original resolution')
        original=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(original,self.size,interpolation=cv2.INTER_AREA)
        # Area support, rather than nearest neighbor, preserves thin strokes.
        local=cv2.resize(mask.astype(np.float32),self.size,interpolation=cv2.INTER_AREA)>0
        boxes=repair_boxes(local)
        return index,rgb,local,boxes,original,mask

    def __iter__(self):
        next_frame=0
        try:
            for _ in range(min(self.capacity,self.frames.count)):
                self.pending.append(self.executor.submit(self._prepare,next_frame))
                next_frame+=1
            while self.pending:
                value=self.pending.popleft().result()
                if next_frame<self.frames.count:
                    self.pending.append(self.executor.submit(self._prepare,next_frame))
                    next_frame+=1
                yield value
        finally:
            self.close()

    def close(self):
        for future in self.pending:future.cancel()
        self.pending.clear()
        self.executor.shutdown(wait=True,cancel_futures=True)


class ResolutionWriter:
    """Restore and composite before the shared segment blender or final encoder."""
    def __init__(self,writer):
        self.writer=writer
        self.originals={}
        self.index=0

    def remember(self,index,image,mask):
        self.originals[index]=(image,mask)

    def save(self,rgb):
        image,mask=self.originals.pop(self.index)
        self.index+=1
        original=torch.from_numpy(image).to(rgb.device)
        if mask.any():
            support=torch.from_numpy(mask).to(rgb.device)
            operation=_compiled_restore() if rgb.is_cuda else restore_rgb
            result=operation(rgb,original,support)
        else:
            result=original
        self.writer.save(result)

    def close(self):
        try:self.writer.close()
        finally:self.originals.clear()

    def profile(self):
        return self.writer.profile()
