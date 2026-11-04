"""Packed binary supports preserve nearest warp samples and extent bounds."""
import cv2,numpy as np
rng=np.random.default_rng(92)
for shape in ((17,81),(100,31),(64,64)):
 for angle in (0,17,-65,179):
  for scale in (.4,1.,1.8):
   mask=(rng.random(shape)>.65).astype(np.uint8)
   extent=(rng.random(shape)>.85).astype(np.uint8)
   matrix=cv2.getRotationMatrix2D((shape[1]/2,shape[0]/2),angle,scale).astype(np.float32)
   matrix[:,2]+=[-8.34,12.55]
   for size in ((97,79),(13,17)):
    a=cv2.warpAffine(mask,matrix,size,flags=cv2.INTER_NEAREST)
    b=cv2.warpAffine(extent,matrix,size,flags=cv2.INTER_NEAREST)
    packed=cv2.warpAffine(mask|(extent<<1),matrix,size,flags=cv2.INTER_NEAREST)
    assert np.array_equal(a,packed&1)
    assert np.array_equal(b,(packed&2)>>1)
    assert cv2.boundingRect(b)==cv2.boundingRect(packed&2)
print('PASS packed nearest projection: rotation, scaling, translation, clipping and exact extent bounds')
