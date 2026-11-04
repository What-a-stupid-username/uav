"""Box-only repair planning with fixed size classes and persistent crop positions."""
import math

SIZES=tuple(range(192,513,32))


def context_margin(width,height):
    from .config import get_config
    # Leave at least 16 pixels of writable core within a 512px model input.
    return min(248,round(min(width,height)*get_config().repair_context_ratio))


def union(a,b):
    return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]


def intersects(a,b):
    return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[2]>=b[2] and a[3]>=b[3]


def dimensions(core,width,height):
    margin=context_margin(width,height)
    required=[max(0,core[0]-margin),max(0,core[1]-margin),min(width,core[2]+margin),min(height,core[3]+margin)]
    dims=[]
    for need,extent in ((required[2]-required[0],width),(required[3]-required[1],height)):
        size=next((min(s,extent) for s in SIZES if min(s,extent)>=need),None)
        if size is None:return None
        dims.append(size)
    return dims,required


def cost(core,width,height):
    dims=dimensions(core,width,height)
    return float('inf') if dims is None else dims[0][0]*dims[0][1]+128**2


def plan_regions(boxes,width,height,previous=(),guard=0):
    """Plan fixed-size crops with background proportional to the video short side.

    Full crops are writable under the final mask; overlapping results blend.
    Optional guard is retained for offline geometry comparisons only.
    """
    cores=[]
    for box in boxes:
        if len(box)!=4 or not all(math.isfinite(x) for x in box):raise ValueError('invalid repair box')
        l,t,r,b=box
        core=[max(0,math.floor(l-guard)),max(0,math.floor(t-guard)),min(width,math.ceil(r+guard)),min(height,math.ceil(b+guard))]
        if core[2]>core[0] and core[3]>core[1]:cores.append(core)
    # Many frames repeat the same proposal (including both anchor envelopes).
    # Remove redundant supports before the cubic greedy search, then close
    # overlapping cores in a cheap pass. Both operations preserve their union.
    unique=sorted(set(map(tuple,cores)),key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),b))
    cores=[]
    for box in unique:
        if not any(contains(other,box) for other in cores):cores.append(list(box))
    while True:
        joined=False
        for i,a in enumerate(cores):
            for j in range(i+1,len(cores)):
                if intersects(a,cores[j]):
                    cores[i]=union(a,cores[j]);del cores[j];joined=True;break
            if joined:break
        if not joined:break
    # First close overlapping output cores: a pixel must have one repair owner.
    # Then greedily merge the best area/call-cost saving, even across small gaps.
    while True:
        candidate=None
        for i,a in enumerate(cores):
            for j in range(i+1,len(cores)):
                b=cores[j];merged=union(a,b)
                saving=cost(a,width,height)+cost(b,width,height)-cost(merged,width,height)
                overlap=intersects(a,b)
                if overlap:
                    score=(1,0 if not math.isfinite(saving) else saving,-i,-j)
                elif math.isfinite(saving) and saving>0:score=(0,saving,-i,-j)
                else:continue
                if candidate is None or score>candidate[0]:candidate=(score,i,j,merged)
        if candidate is None:break
        _,i,j,merged=candidate;cores[i]=merged;del cores[j]
    # Very broad marks retain bounded model inputs instead of growing VRAM.
    split=[]
    core_size=min(384,512-2*context_margin(width,height))
    for l,t,r,b in cores:
        for y in range(t,b,core_size):
            for x in range(l,r,core_size):split.append([x,y,min(r,x+core_size),min(b,y+core_size)])
    regions=[]
    for core in sorted(split,key=lambda b:(b[1],b[0],b[3],b[2])):
        info=dimensions(core,width,height)
        if info is None:raise ValueError('repair core exceeds supported size')
        (cw,ch),required=info
        reusable=[list(p) for p in previous if p[2]-p[0]==cw and p[3]-p[1]==ch and contains(p,required)]
        if reusable:crop=min(reusable,key=lambda p:(p[1],p[0]))
        else:
            # Snap positions where possible, while preserving every context pixel.
            pos=[]
            for lo,hi,size,extent in ((required[0],required[2],cw,width),(required[1],required[3],ch,height)):
                low=max(0,hi-size);high=min(lo,extent-size)
                ideal=(lo+hi-size)/2
                value=max(low,min(high,round(ideal/16)*16))
                pos.append(int(value))
            x,y=pos;crop=[x,y,x+cw,y+ch]
        regions.append(dict(core=core,crop=crop))
    return regions
