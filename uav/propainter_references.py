"""Bounded, deterministic reference features for rolling video repair."""
from collections import OrderedDict
import random
import torch
import torch.nn.functional as F


class ReferenceBank:
    def __init__(self, fps, count=6, seconds=3., capacity_bytes=128*2**20, seed=20261019,
                 host_capacity_bytes=4096*2**20):
        if fps<=0 or count<0 or seconds<0 or capacity_bytes<0 or host_capacity_bytes<0:
            raise ValueError('invalid reference configuration')
        self.radius=round(fps*seconds);self.count=count;self.capacity=capacity_bytes;self.seed=seed
        self.frames={};self.masks={};self.features=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.peak=0;self.local_fallbacks=0
        self.host=OrderedDict();self.host_capacity=host_capacity_bytes
        self.host_ready={}
        self.host_bytes=0;self.host_peak=0;self.host_hits=0

    def _retain_host(self,key,entry):
        size=entry.numel()*entry.element_size()
        if key in self.host or size>self.host_capacity:return
        while self.host and self.host_bytes+size>self.host_capacity:
            old_key,old=self.host.popitem(last=False)
            self.host_ready.pop(old_key,None)
            self.host_bytes-=old.numel()*old.element_size()
        # Only evicted immutable encodings move to RAM. No evidence is persisted.
        if entry.is_cuda:
            host=torch.empty(entry.shape,dtype=entry.dtype,pin_memory=True)
            host.copy_(entry,non_blocking=True)
            ready=torch.cuda.Event()
            ready.record(torch.cuda.current_stream(entry.device))
            self.host_ready[key]=ready
        else:host=entry
        self.host[key]=host;self.host_bytes+=size
        self.host_peak=max(self.host_peak,self.host_bytes)

    def add(self, index, rgb, mask):
        self.frames[index]=rgb;self.masks[index]=mask

    def select(self, start, end, total):
        low=max(0,start-self.radius);high=min(total,end+self.radius)
        candidates=list(range(low,start))+list(range(end,high))
        return sorted(random.Random(self.seed+start).sample(candidates,min(self.count,len(candidates))))

    def prune(self, first, required=None):
        retain=lambda i:i>=first or (required is not None and i in required)
        self.frames={i:x for i,x in self.frames.items() if retain(i)}
        self.masks={i:x for i,x in self.masks.items() if retain(i)}
        for key in list(self.features):
            if not retain(key[1]):
                entry=self.features.pop(key);self.bytes-=entry.numel()*entry.element_size()
        for key in list(self.host):
            if not retain(key[1]):
                entry=self.host.pop(key);self.host_bytes-=entry.numel()*entry.element_size()
                self.host_ready.pop(key,None)

    def _encode(self,backend,ids):
        """Encode independent references in bounded TRT batches."""
        l,t,r,b,ph,pw=backend.flow_crop
        images=[];masks=[]
        for idx in ids:
            crop=self.frames[idx][:,t:b,l:r];mask=self.masks[idx][t:b,l:r]
            if ph or pw:
                crop=F.pad(crop,(0,pw,0,ph),mode='replicate')
                mask=F.pad(mask,(0,pw,0,ph))
            mask=mask[None,None,None].half()
            images.append((crop[None,None].half()/127.5-1)*(1-mask));masks.append(mask)
        image=images[0] if len(images)==1 else torch.cat(images,dim=1)
        mask=masks[0] if len(masks)==1 else torch.cat(masks,dim=1)
        encoded=backend.model.encode_frames(image,mask,mask)
        # Own each result separately: caching one view must not retain a batch.
        return {idx:encoded[:,i:i+1].clone() if len(ids)>1 else encoded for i,idx in enumerate(ids)}

    @torch.inference_mode()
    def get(self, backend, start, end, total):
        ids=self.select(start,end,total)
        if not ids:
            self.local_fallbacks+=1
            return None
        missing=[idx for idx in ids if (backend.flow_crop,idx) not in self.features
                 and (backend.flow_crop,idx) not in self.host]
        l,t,r,b,ph,pw=backend.flow_crop
        use_trt=(hasattr(getattr(getattr(backend,'model',None),'encoder',None),'trt_backend')
                 and self.frames[ids[0]].is_cuda
                 and all(8<=size<=520 for size in (b-t+ph,r-l+pw)))
        batch=4 if use_trt else 1
        precomputed={}
        for offset in range(0,len(missing),batch):
            precomputed.update(self._encode(backend,missing[offset:offset+batch]))
        entries=[]
        for idx in ids:
            key=(backend.flow_crop,idx)
            if key in self.features:
                entry=self.features.pop(key);self.features[key]=entry;self.hits+=1
            else:
                self.misses+=1
                if key in self.host:
                    entry=self.host.pop(key);self.host[key]=entry;self.host_hits+=1
                    device=self.frames[idx].device
                    ready=self.host_ready.get(key)
                    if ready is not None:
                        if device.type=='cuda':torch.cuda.current_stream(device).wait_event(ready)
                        else:ready.synchronize()
                    entry=entry.to(device,non_blocking=True)
                else:
                    entry=precomputed.pop(idx,None)
                    # A small/disabled cache can evict an earlier hit while
                    # consuming this request. Re-encode only that reference.
                    if entry is None:entry=self._encode(backend,[idx])[idx]
                size=entry.numel()*entry.element_size()
                while self.features and self.bytes+size>self.capacity:
                    old_key,old=self.features.popitem(last=False)
                    self._retain_host(old_key,old)
                    self.bytes-=old.numel()*old.element_size()
                if size<=self.capacity:
                    self.features[key]=entry;self.bytes+=size;self.peak=max(self.peak,self.bytes)
            entries.append(entry)
        return torch.cat(entries,dim=1)

    def clear(self):
        self.frames.clear();self.masks.clear();self.features.clear();self.bytes=0
        self.host.clear();self.host_bytes=0
        self.host_ready.clear()

    def profile(self):
        return dict(count=self.count,radius_frames=self.radius,seed=self.seed,hits=self.hits,
                    misses=self.misses,peak_bytes=self.peak,capacity_bytes=self.capacity,
                    local_fallbacks=self.local_fallbacks,host_hits=self.host_hits,
                    host_peak_bytes=self.host_peak,host_capacity_bytes=self.host_capacity)
