from uav.repair_regions import plan_regions,contains,intersects,SIZES,context_margin
for boxes,w,h in [([],720,1316),([[350,350,410,410]],720,1316),([[0,0,20,20],[690,1290,720,1316]],720,1316),([[0,0,720,1316]],720,1316),([[10,10,80,40],[70,20,160,60],[170,20,230,50]],720,1316),([[5,5,10,10]],120,80)]:
 regions=plan_regions(boxes,w,h)
 for i,a in enumerate(regions):
  assert contains(a['crop'],a['core'])
  assert all(not intersects(a['core'],b['core']) for b in regions[i+1:])
  assert a['crop'][2]-a['crop'][0] in tuple(min(size,w) for size in SIZES)
  assert a['crop'][3]-a['crop'][1] in tuple(min(size,h) for size in SIZES)
  margin=context_margin(w,h)
  assert contains(a['crop'],[max(0,a['core'][0]-margin),max(0,a['core'][1]-margin),min(w,a['core'][2]+margin),min(h,a['core'][3]+margin)])
 for l,t,r,b in boxes:
  for y in range(t,b):
   for x in range(l,r):assert any(a['core'][0]<=x<a['core'][2] and a['core'][1]<=y<a['core'][3] for a in regions)
 assert plan_regions(boxes,w,h,[a['crop'] for a in regions])==regions
# A grid corner should need only one crop, not four fixed tiles.
assert len(plan_regions([[370,370,398,398]],720,1316))==1
assert context_margin(720,1280)==64
assert context_margin(504,896)==45
assert context_margin(1920,1080)==96
for region in plan_regions([[0,0,1920,1080]],1920,1080):
 assert region['crop'][2]-region['crop'][0]<=512
 assert region['crop'][3]-region['crop'][1]<=512
print('PASS: disjoint ownership, coverage, edges, size classes, stable crops and grid-boundary case')
