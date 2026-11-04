import time
import cv2
import numpy as np
import torch
from uav.gpu_components import components

fixtures=[]
rng=np.random.default_rng(71)
for shape in [(1,1),(7,33),(65,97),(320,180)]:
    for density in [0.,.1,.5,1.]:fixtures.append(rng.random(shape)<density)
snake=np.zeros((65,97),np.uint8);snake[::2]=1
for y in range(1,65,2):snake[y,-1 if (y//2)%2==0 else 0]=1
fixtures.append(snake)
fixtures.append(np.eye(65,dtype=np.uint8))
for sample in fixtures:
    mask=torch.as_tensor(sample.astype(bool),device='cuda')
    n,ref=cv2.connectedComponents(sample.astype(np.uint8),connectivity=8)
    for _ in range(3):
        actual=components(mask).cpu().numpy()
        assert (actual<0).tolist()==(ref==0).tolist()
        assert len(np.unique(actual[actual>=0]))==n-1
        for label in range(1,n):assert len(np.unique(actual[ref==label]))==1
mask=torch.zeros((320,180),dtype=torch.bool,device='cuda')
mask[20:35,10:100]=True;mask[130:145,20:130]=True;mask[290:300,50:170]=True
for _ in range(5):components(mask)
torch.cuda.synchronize();start=time.perf_counter()
for _ in range(100):components(mask)
torch.cuda.synchronize()
print('GPU CCL matches CPU partitions for random/dense/empty/diagonal/snake masks: PASS')
print('warm GPU CCL milliseconds:',(time.perf_counter()-start)*10)
