"""Presence is applied exactly once for joint and separate scoring models."""
from types import SimpleNamespace
import torch
from sam3.model.sam3_image_processor import Sam3Processor
for joint in [True,False]:
    presence=torch.tensor([[.62646484375]])
    raw=torch.tensor([[[.7880859375],[.76416015625],[.685546875],[.681640625],[.60888671875],[.5380859375]]])
    probabilities=raw*presence.unsqueeze(1) if joint else raw
    model=SimpleNamespace(supervise_joint_box_scores=joint, _get_dummy_prompt=lambda **kw: None)
    model.forward_grounding=lambda **kw:dict(pred_boxes=torch.full((1,6,4),.5),pred_logits=probabilities.logit(),presence_logit_dec=presence.logit())
    processor=Sam3Processor(model,device='cpu',confidence_threshold=.3)
    state=processor._forward_grounding(dict(backbone_out={},geometric_prompt=None,original_width=720,original_height=1316,boxes_only=True))
    assert len(state['boxes'])==6
    torch.testing.assert_close(state['scores'],(raw*presence.unsqueeze(1)).flatten())
print('PASS: joint/separate presence scoring retains the same six candidates')
