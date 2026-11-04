import torch
from uav.repair_blend import RepairBlend,edge_weights
frames={0:torch.full((3,12,20),77,device='cuda',dtype=torch.uint8)}
mask=torch.ones((1,12,12),device='cuda',dtype=torch.bool);mask[:,:2]=False
patches=[(torch.full((3,12,12),50,device='cuda',dtype=torch.uint8),(0,0,12,12)),(torch.full((3,12,12),150,device='cuda',dtype=torch.uint8),(8,0,20,12))]
results=[]
for order in [patches,patches[::-1]]:
 blend=RepairBlend(frames)
 for patch,crop in order:blend.add(0,patch,mask,crop,edge_weights(12,12,'cuda'))
 results.append(blend.finish(0,frames[0]))
assert torch.equal(*results)
r=results[0];assert (r[:,:2]==77).all()
assert (r[:,2:,0:8]==50).all() and (r[:,2:,12:]==150).all()
assert r[0,6,8]<r[0,6,11]
# Sole source preserves all values exactly, including outermost pixels.
b=RepairBlend(frames);b.add(0,frames[0],torch.ones(1,12,20,device='cuda'),(0,0,20,12),edge_weights(12,20,'cuda'))
assert torch.equal(b.finish(0,frames[0]),frames[0])
print('PASS: edge weights, order independence, single-source identity and mask-outside preservation')
