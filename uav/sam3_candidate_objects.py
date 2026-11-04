"""Group pixel fragments using substantial, competitive candidate coverage."""
import torch

def candidate_objects(boxes,objects,candidates):
    if len(boxes)<2 or not candidates:return boxes,objects
    parents=list(range(len(boxes)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    checks=[];pairs=[]
    # Count intersections rather than accepting one stray foreground pixel.
    # Areas are reduced once per component, on the same device as the masks.
    areas=torch.stack([objects[i][t:b,l:r].sum()
                       for i,(l,t,r,b) in enumerate(boxes)])
    for cid,(crop,mask) in enumerate(candidates):
        l,t,r,b=crop
        for oid,(x,y,z,v) in enumerate(boxes):
            x0,y0,x1,y1=max(l,x),max(t,y),min(r,z),min(b,v)
            if x1<=x0 or y1<=y0:continue
            checks.append((objects[oid][y0:y1,x0:x1]&mask[y0-t:y1-t,x0-l:x1-l]).sum())
            pairs.append((cid,oid))
    if not checks:return boxes,objects
    coverage=torch.zeros((len(candidates),len(boxes)),device=areas.device,dtype=torch.float32)
    ci,oi=zip(*pairs)
    coverage[list(ci),list(oi)]=torch.stack(checks).float()/areas[list(oi)].clamp_min(1)
    # A candidate must explain a majority of each fragment, and compete with
    # that fragment's best candidate. Whole-word candidates can join distant
    # strokes; a candidate mostly describing another object cannot annex them.
    eligible=(coverage>=.6)&(coverage>=.8*coverage.amax(0,keepdim=True))
    evidence=eligible.cpu().tolist()
    members={i:{i} for i in range(len(boxes))}
    for row in sorted(evidence,key=lambda row:-sum(row)):
        supported={i for i,hit in enumerate(row) if hit}
        roots={root(i) for i in supported}
        if len(roots)<2:continue
        whole=set().union(*(members[i] for i in roots))
        # Require one candidate to support the whole proposed group: A-B and
        # B-C alone do not prove A-B-C is one object (union-find chaining).
        if not whole<=supported:continue
        owner=min(roots)
        for other in roots:
            if other!=owner:parents[other]=owner;members.pop(other)
        members[owner]=whole
    groups={}
    for i in range(len(boxes)):groups.setdefault(root(i),[]).append(i)
    result_boxes=[];result_objects={}
    for oid,members in enumerate(groups.values()):
        result_boxes.append([min(boxes[i][0] for i in members),min(boxes[i][1] for i in members),max(boxes[i][2] for i in members),max(boxes[i][3] for i in members)])
        mask=objects[members[0]]
        if len(members)>1:
            mask=mask.clone()
            for i in members[1:]:mask|=objects[i]
        result_objects[oid]=mask
    return result_boxes,result_objects
