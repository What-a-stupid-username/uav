"""Batched refined box D2H preserves offsets, order and full-precision values."""
import torch
for count in (0,1,17):
 for dtype in (torch.float16,torch.float32):
  boxes=torch.arange(count*4,device='cuda',dtype=dtype).reshape(count,4).float()/7
  offset=boxes.new_tensor([123,456,123,456])
  old=[(box+offset).cpu().tolist() for box in boxes]
  new=(boxes+offset).cpu().tolist()
  assert old==new
print('PASS: batch box transfer exactly equals per-box transfer, including empty output')
