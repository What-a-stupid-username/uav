import torch
from uav.sam3_pixel_objects import pixel_objects

for device in ('cpu','cuda'):
    mask=torch.zeros((127,259),dtype=torch.bool,device=device)
    # Separated glyphs should form a line; a distant region remains separate.
    mask[20:35,10:20]=True;mask[20:35,27:37]=True;mask[20:35,44:54]=True
    mask[80:100,180:220]=True;mask[126,258]=True
    boxes,objects=pixel_objects(mask)
    assert len(boxes)==3,boxes
    assert [10,20,54,35] in boxes
    combined=torch.stack(list(objects.values()))
    assert torch.equal(combined.any(0),mask)
    assert int(combined.sum())==int(mask.sum()),'duplicate ownership'
    for box,obj in zip(boxes,objects.values()):
        l,t,r,b=box
        assert obj[t:b,l:r].sum()==obj.sum()
    empty=torch.zeros_like(mask)
    assert pixel_objects(empty)==([], {})
    # No detector boxes are supplied: a broad SAM box cannot bridge this gap.
    sparse=torch.zeros_like(mask);sparse[50:65,0:20]=True;sparse[50:65,150:200]=True
    assert len(pixel_objects(sparse)[0])==2
    full=torch.ones((1,1),dtype=torch.bool,device=device)
    assert pixel_objects(full)[0]==[[0,0,1,1]]
print('CPU/GPU pixel clustering: text grouping, distant separation, borders and exact disjoint union: PASS')
generator=torch.Generator().manual_seed(31)
for shape in [(33,65),(127,259)]:
    fixture=torch.rand(shape,generator=generator)>.99
    cpu_boxes,cpu_objects=pixel_objects(fixture)
    gpu_boxes,gpu_objects=pixel_objects(fixture.cuda())
    assert sorted(cpu_boxes)==sorted(gpu_boxes)
    cpu_pairs={tuple(b):cpu_objects[i] for i,b in enumerate(cpu_boxes)}
    for i,b in enumerate(gpu_boxes):assert torch.equal(cpu_pairs[tuple(b)],gpu_objects[i].cpu())
print('CPU/GPU clustering partitions and original-pixel boxes agree: PASS')
