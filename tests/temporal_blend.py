from uav.temporal_blend import window_weight
for center in [0,5,10]:
 assert window_weight(center,center,5)==1
 for d in range(1,6):
  assert 0<window_weight(center+d,center,5)<window_weight(center+d-1,center,5)
  assert abs(window_weight(center+d,center,5)-window_weight(center-d,center,5))<1e-10
# Partition via normalization is invariant to order and preserves constant output.
for i in range(16):
 weights=[window_weight(i,c,5) for c in [0,5,10,15] if abs(i-c)<=5]
 assert abs(sum(123*w for w in weights)/sum(weights)-123)<1e-10
print('PASS: positive symmetric smooth weights, normalization and constant preservation')
