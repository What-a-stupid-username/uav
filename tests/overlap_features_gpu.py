"""Synthetic spatial alignment and weighted feature fusion checks."""
import torch
from uav.sam3_repair_sink import FeatureCanvas
from uav.repair_blend import edge_weights

canvas=FeatureCanvas()
canvas.add(torch.ones((128,8,8),device='cuda',dtype=torch.float16),(0,0,32,32),.25)
canvas.add(torch.full((128,8,8),3.,device='cuda',dtype=torch.float16),(16,0,48,32),.75)
normalized=canvas.sums/canvas.weights
assert torch.allclose(normalized[:,:,:4],torch.ones_like(normalized[:,:,:4]))
w=edge_weights(8,8,'cuda')
expected=(w[:,:,4:]*.25+w[:,:,:4]*2.25)/(w[:,:,4:]*.25+w[:,:,:4]*.75)
assert torch.allclose(normalized[:,:,4:8],expected.expand(128,-1,-1))
assert torch.allclose(normalized[:,:,8:],torch.full_like(normalized[:,:,8:],3.))
assert canvas.bounds==(0,0,12,8)
host=FeatureCanvas('cpu')
host.add(torch.ones((128,8,8),device='cuda',dtype=torch.float16),(0,0,32,32),.25)
host.add(torch.full((128,8,8),3.,device='cuda',dtype=torch.float16),(16,0,48,32),.75)
assert host.sums.device.type=='cpu'
assert torch.allclose(host.sums,canvas.sums.cpu(),rtol=1e-6,atol=1e-7)
assert torch.allclose(host.weights,canvas.weights.cpu(),rtol=1e-6,atol=1e-7)
print('feature coordinates and normalized overlap: PASS')
