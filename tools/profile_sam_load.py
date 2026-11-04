"""Profile SAM startup without decoding media or running inference."""
import json
import time
started = time.perf_counter()
import torch
print(json.dumps(dict(phase='import_torch',seconds=time.perf_counter()-started)),flush=True)
started = time.perf_counter()
import sam3.model_builder as builder
print(json.dumps(dict(phase='import_sam_builder',seconds=time.perf_counter()-started)),flush=True)

def timed(obj, name):
    original=getattr(obj,name)
    def call(*args,**kwargs):
        tick=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:print(json.dumps(dict(phase=name,seconds=time.perf_counter()-tick)),flush=True)
    setattr(obj,name,call)

for name in ('_create_vision_backbone','_create_text_encoder','_create_sam3_transformer',
             '_create_segmentation_head','_load_checkpoint'):
    timed(builder,name)
timed(torch,'load')
timed(torch.nn.Module,'load_state_dict')
timed(torch.nn.Module,'cuda')
from uav.sam3_model import build
from uav.inference_weights import compact_weights
torch.set_num_threads(4)
started=time.perf_counter()
model=build()
torch.cuda.synchronize()
print(json.dumps(dict(phase='build_total',seconds=time.perf_counter()-started)),flush=True)
started=time.perf_counter()
saved=compact_weights(model.backbone)
torch.cuda.synchronize()
print(json.dumps(dict(phase='compact_weights',seconds=time.perf_counter()-started,saved_bytes=saved)),flush=True)
