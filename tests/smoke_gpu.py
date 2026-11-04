"""GPU smoke test using generated tensors only; never opens media files."""
import json
import torch
from uav.settings import validate_models, propainter_models, sam3_checkpoint
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from sam3.model.sam3_image_processor import Sam3Processor
from uav.sam3_resolution import set_resolution
from uav.propainter.propainter_inpaint import PropainterInpaint

@torch.inference_mode()
def main():
    validate_models()
    torch.set_num_threads(4)
    with torch.autocast('cuda', dtype=torch.float16):
        video = Sam3VideoPredictor(checkpoint_path=str(sam3_checkpoint()))
        model = video.model.detector
        processor = Sam3Processor(model, resolution=504, confidence_threshold=.3)
        set_resolution(model, 504)
        # Purely synthetic input held in memory.
        image = torch.zeros((3, 128, 128), dtype=torch.uint8, device='cuda')
        from torchvision.transforms.v2.functional import to_image
        state = processor.set_image(to_image(image))
        state = processor.set_text_prompt('watermark', state)
        assert 'boxes' in state and 'scores' in state
        print(json.dumps({'sam3_detection': 'passed', 'boxes': len(state['boxes'])}), flush=True)
    repair = PropainterInpaint(torch.device('cuda'), str(propainter_models()),
                              sub_video_length=24, flow_mask_dilation=4, mask_dilation=0)
    repair.raft_iter = 8
    repair.fix_raft.fix_raft.args.mixed_precision = True
    repair.fix_raft.fix_raft.final_upsample_only = True
    frames = torch.full((4, 3, 128, 128), 128, dtype=torch.uint8, device='cuda')
    masks = torch.zeros((4, 128, 128), dtype=torch.bool, device='cuda')
    masks[:, 48:72, 48:72] = True
    output = repair.inpaint(frames, masks, output_range=(0,4))
    assert len(output) == 4
    for i in range(4):
        assert output[i].shape == frames[i].shape
        assert output[i].dtype == torch.uint8
        assert torch.equal(output[i][:, ~masks[i]], frames[i][:, ~masks[i]])
    torch.cuda.synchronize()
    print(json.dumps({'propainter': 'passed', 'frames': 4,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}), flush=True)

if __name__ == '__main__':
    main()
