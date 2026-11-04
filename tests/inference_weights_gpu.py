import torch,json,time
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from uav.settings import sam3_checkpoint
from uav.inference_weights import compact_weights
@torch.inference_mode()
@torch.autocast('cuda',dtype=torch.float16)
def main():
 torch.set_num_threads(4);torch.manual_seed(221)
 video=Sam3VideoPredictor(checkpoint_path=str(sam3_checkpoint()))
 model=video.model.detector
 x=torch.randn(1,3,336,336,device='cuda')
 from uav.sam3_resolution import set_resolution
 set_resolution(model,336)
 def call():return model.backbone.forward_image(x)
 old=call();text=model.backbone.forward_text(['watermark'],device='cuda')
 from sam3.model.sam3_image_processor import Sam3Processor
 from PIL import Image
 processor=Sam3Processor(model,resolution=336,confidence_threshold=.0)
 def ground():
  state=processor.set_image(Image.new('RGB',(640,480),color=(90,130,180)))
  state['backbone_out'].update(model.backbone.forward_text(['watermark'],device='cuda'))
  state['geometric_prompt']=model._get_dummy_prompt()
  return processor._forward_grounding(state)
 oldground=ground()
 torch.cuda.synchronize();tick=time.monotonic()
 for _ in range(3):call()
 torch.cuda.synchronize();before=(time.monotonic()-tick)/3
 saved=compact_weights(model.backbone)
 new=call();newtext=model.backbone.forward_text(['watermark'],device='cuda');newground=ground()
 for key in ('boxes','scores','masks_logits'):
  torch.testing.assert_close(oldground[key],newground[key],rtol=0,atol=0)
 from torch.utils._pytree import tree_flatten
 diffs=[]
 for a,b in zip(tree_flatten((old,text))[0],tree_flatten((new,newtext))[0]):
  if isinstance(a,torch.Tensor) and a.is_floating_point():
   torch.testing.assert_close(a,b,rtol=0,atol=0);diffs.append(float((a-b).abs().max()))
 torch.cuda.synchronize();tick=time.monotonic()
 for _ in range(3):call()
 torch.cuda.synchronize();after=(time.monotonic()-tick)/3
 print(json.dumps(dict(saved_mib=saved/2**20,bitexact=True,before_seconds=before,after_seconds=after)))
main()
