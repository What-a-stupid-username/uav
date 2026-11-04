"""GPU detector-only parity on in-memory synthetic input; no media access."""
import torch,time,json
from PIL import Image
import numpy as np
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from sam3.model.sam3_image_processor import Sam3Processor
from uav.settings import sam3_checkpoint
from uav.sam3_resolution import set_resolution
from uav.inference_weights import compact_weights
from uav.sam3_crop_groups import set_detection_images,segment_group
@torch.inference_mode()
def main():
 torch.set_num_threads(4)
 with torch.autocast('cuda',dtype=torch.float16):
  video=Sam3VideoPredictor(checkpoint_path=str(sam3_checkpoint()));model=video.model.detector;compact_weights(model.backbone);set_resolution(model,1008)
  p=Sam3Processor(model,resolution=1008,confidence_threshold=.3);p.generate_binary_masks=False
  text=model.backbone.forward_text(['watermark'],device='cuda')
  image=Image.fromarray(np.random.default_rng(27).integers(0,256,(231,357,3),dtype=np.uint8))
  old=p.set_image(image);new=dict(original_height=231,original_width=357,backbone_out=set_detection_images(p,[image]))
  for key in ('vision_features','vision_pos_enc','backbone_fpn'):
   a,b=old['backbone_out'][key],new['backbone_out'][key]
   if isinstance(a,list):assert all(torch.equal(x,y) for x,y in zip(a,b)),key
   else:assert torch.equal(a,b),key
  a=segment_group(p,image,dict(crop=[0,0,357,231]),text,state=old)
  b=segment_group(p,image,dict(crop=[0,0,357,231]),text,state=new)
  for key in ('boxes','scores','masks_logits'):assert torch.equal(a[key],b[key]),key
  print('PASS 1008 detector features, boxes, scores and mask probabilities exactly equal; tracker neck omitted',flush=True)
if __name__ == "__main__":main()
