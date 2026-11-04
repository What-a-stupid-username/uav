import torch
from uav.sam3_pixel_objects import pixel_objects
from uav.sam3_candidate_objects import candidate_objects
for device in ('cpu','cuda'):
 mask=torch.zeros((100,240),dtype=torch.bool,device=device)
 mask[10:25,10:30]=True;mask[65:85,10:30]=True;mask[10:30,180:200]=True
 boxes,objects=pixel_objects(mask);assert len(boxes)==3
 candidate=torch.zeros_like(mask);candidate[:,0:40]=mask[:,0:40]
 result,merged=candidate_objects(boxes,objects,[((0,0,240,100),candidate)])
 assert len(result)==2 and [10,10,30,85] in result
 assert torch.equal(torch.stack(list(merged.values())).any(0),mask)
 assert sum(int(m.sum()) for m in merged.values())==int(mask.sum())
 # A broad bounding rectangle with zero candidate pixels in the other object
 # must not connect that object. Crop-local coordinates are handled exactly.
 local=candidate[0:100,0:40]
 assert len(candidate_objects(boxes,objects,[((0,0,40,100),local)])[0])==2
 assert len(candidate_objects(boxes,objects,[])[0])==3
 # Candidate IDs are scoped per invocation, never carried across frames.
 singles=[((0,0,240,100),m) for m in objects.values()]
 assert len(candidate_objects(boxes,objects,singles)[0])==3
 # Stray pixels cannot override complete independent candidates.
 mixed=objects[0].clone();other=objects[1].nonzero()[:5]
 mixed[other[:,0],other[:,1]]=True
 assert len(candidate_objects(boxes,objects,singles+[((0,0,240,100),mixed)])[0])==3
 # Partial A-B/B-C evidence must not transitively swallow all three.
 ab=objects[0]|objects[1];bc=objects[1]|objects[2]
 result,merged=candidate_objects(boxes,objects,[((0,0,240,100),ab),((0,0,240,100),bc)])
 assert len(result)==2
 assert torch.equal(torch.stack(list(merged.values())).any(0),mask)
 # One complete candidate may still join all distant fragments, even when
 # duplicate per-fragment proposals also exist.
 result,merged=candidate_objects(boxes,objects,singles+[((0,0,240,100),mask)])
 assert len(result)==1 and torch.equal(merged[0],mask)
print('PASS CPU/GPU: same SAM candidate connects fragments, mask support not boxes, exact disjoint union, crop offsets and independent candidates')
