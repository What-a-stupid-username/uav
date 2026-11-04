"""GPU bidirectional raw-flow cache for exactly matching crop/frame pairs."""
class FlowCache:
    def __init__(self,capacity_bytes=256*2**20):
        self.items={};self.capacity=capacity_bytes;self.bytes=0;self.hits=0;self.misses=0;self.peak=0;self.keep_from=0

    def estimate(self,raft,frames,start,crop,iters,scale=1.):
        import torch
        from .raft_resolution import estimate_flow,flow_size
        size=flow_size(*frames.shape[-2:],scale)
        keys=[(crop,start+i,iters,raft.fix_raft.args.mixed_precision,getattr(raft,'reuse_features',False),size) for i in range(frames.shape[1]-1)]
        output={};missing=[]
        for i,key in enumerate(keys):
            if key in self.items:output[i]=self.items[key];self.hits+=1
            else:missing.append(i);self.misses+=1
        # Keep consecutive missing pairs batched; never span a cached gap.
        cursor=0
        while cursor<len(missing):
            lo=missing[cursor];hi=lo+1;cursor+=1
            while cursor<len(missing) and missing[cursor]==hi:hi+=1;cursor+=1
            f,b=estimate_flow(raft,frames[:,lo:hi+1],iters,scale)
            for j in range(lo,hi):
                pair=(f[:,j-lo],b[:,j-lo])
                size=sum(x.numel()*x.element_size() for x in pair)
                if keys[j][1]>=self.keep_from and self.bytes+size<=self.capacity:
                    pair=tuple(x.clone() for x in pair)
                    self.items[keys[j]]=pair;self.bytes+=size;self.peak=max(self.peak,self.bytes)
                output[j]=pair
            if lo == 0 and hi == len(keys):
                return f,b
        return tuple(torch.stack([output[i][direction] for i in range(len(keys))],dim=1) for direction in (0,1))

    def prune(self,first):
        for key in list(self.items):
            if key[1]<first:
                self.bytes-=sum(x.numel()*x.element_size() for x in self.items.pop(key))

    def profile(self):
        return dict(hits=self.hits,misses=self.misses,peak_bytes=self.peak,capacity_bytes=self.capacity)
