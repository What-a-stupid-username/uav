"""Verify fast SAM loading against the upstream builder; no media/inference."""
import torch
from sam3.model_builder import build_sam3_image_model
from uav.settings import sam3_checkpoint
from uav.sam3_model import build

torch.set_num_threads(4)
reference = build_sam3_image_model(checkpoint_path=str(sam3_checkpoint()),
    load_from_HF=False, device='cpu').state_dict()
optimized = build().state_dict()
assert reference.keys() == optimized.keys()
for name, value in reference.items():
    assert torch.equal(value, optimized[name].cpu()), name
print('PASS: all', len(reference), 'SAM state tensors are exactly equal')
