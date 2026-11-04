import numpy as np
from uav.review import draw_box
im=np.zeros((40,60,3),dtype=np.uint8)
draw_box(im,[2,3,50,30],(255,50,50))
assert tuple(im[3,2])==(255,50,50)
assert not im[4:29,3:49].any()
im[:]=0
draw_box(im,[2,3,50,30],(40,255,80),dashed=True)
assert im[3,2].any() and not im[3,10].any() and im[3,12].any()
draw_box(im,[-20,-10,100,100],(1,2,3))
print('PASS: thin solid/dashed box colors, gaps, clipping and unchanged interior')
