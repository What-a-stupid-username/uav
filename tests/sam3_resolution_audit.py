"""SAM3 spatial configuration/parity audit on synthetic RAM-only input.

No image/video/cache files are decoded. The only external model input is the
existing checkpoint. JSON reports contain shapes, errors and counts only.
"""
import argparse
import gc
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from uav.inference_weights import compact_weights
from uav.sam3_resolution import set_resolution
from uav.sam3_crop_groups import set_detection_images, segment_group


def independent_model(size):
    """Construct upstream components at the requested size, without adapter."""
    import sam3.model_builder as builder
    from uav.sam3_model import build
    original_vit, original_decoder = builder.ViT, builder.TransformerDecoder
    original_load = torch.nn.Module.load_state_dict

    def vit(*args, **kwargs):
        kwargs['img_size'] = size
        return original_vit(*args, **kwargs)

    def decoder(*args, **kwargs):
        kwargs['resolution'] = size
        return original_decoder(*args, **kwargs)

    replaced = []

    def load(module, state_dict, *args, **kwargs):
        # The checkpoint includes deterministic 72x72 RoPE buffers. Preserve
        # independently constructed small-grid buffers on size mismatch, while
        # loading EVERY learned parameter unchanged.
        state_dict = dict(state_dict)
        for key, value in module.named_buffers():
            if key.endswith('.freqs_cis') and key in state_dict and state_dict[key].shape != value.shape:
                state_dict[key] = value
                replaced.append(key)
        return original_load(module, state_dict, *args, **kwargs)

    with patch.object(builder, 'ViT', vit), patch.object(builder, 'TransformerDecoder', decoder), \
         patch.object(torch.nn.Module, 'load_state_dict', load):
        model = build()
    assert len(replaced) == 4, replaced
    compact_weights(model.backbone)
    return model


def synthetic_image():
    rng = np.random.default_rng(792)
    image = rng.integers(0, 256, (231, 357, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (15, 15), 3)
    cv2.putText(image, 'WATERMARK', (15, 112), cv2.FONT_HERSHEY_SIMPLEX, .95,
                (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(image, 'example 2026', (65, 145), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (225, 225, 225), 1, cv2.LINE_AA)
    return Image.fromarray(image)


def compare(a, b):
    assert a.keys() == b.keys()
    report = {}
    for key in a:
        assert a[key].shape == b[key].shape, (key, a[key].shape, b[key].shape)
        delta = (a[key].float()-b[key].float()).abs()
        report[key] = dict(shape=list(a[key].shape), exact=torch.equal(a[key], b[key]),
                           max_error=float(delta.max()) if delta.numel() else 0.)
    return report


def model_configuration(model, size):
    grid = size // 14
    trunk = model.backbone.vision_backbone.trunk
    rows = []
    for index, block in enumerate(trunk.blocks):
        attn = block.attn
        actual = block.window_size or grid
        assert attn.input_size == (actual, actual)
        assert attn.freqs_cis.shape[0] == actual * actual
        rows.append(dict(block=index, window=block.window_size, input_size=list(attn.input_size),
                         rope_base=list(attn.rope_pt_size), rope_scale=attn.rope_pt_size[0]/actual,
                         rope_tokens=attn.freqs_cis.shape[0]))
    decoder = model.transformer.decoder
    assert decoder.compilable_stored_size == (grid, grid)
    coords = decoder.compilable_cord_cache
    for coord in coords:
        expected = torch.arange(grid,device=coord.device,dtype=coord.dtype)/grid
        assert torch.equal(coord,expected)
    return dict(size=size,patch_size=list(trunk.patch_embed.proj.kernel_size),
                position_embedding_shape=list(trunk.pos_embed.shape),tile_abs_pos=trunk.tile_abs_pos,
                blocks=rows,decoder_grid=list(decoder.compilable_stored_size))


@torch.inference_mode()
def forward(model, size, image):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    processor = Sam3Processor(model,resolution=size,confidence_threshold=0.)
    processor.generate_binary_masks=False
    text=model.backbone.forward_text(['watermark'],device='cuda')
    full=processor.set_image(image)['backbone_out']
    fast=set_detection_images(processor,[image])
    feature_shapes={}
    # Detector-only optimized neck must match the full upstream neck, at every
    # resolution (not just the previously checked 1008).
    for key in ('vision_features','vision_pos_enc','backbone_fpn'):
        aa,bb=full[key],fast[key]
        if isinstance(aa,list):
            assert len(aa)==len(bb)
            assert all(torch.equal(a,b) for a,b in zip(aa,bb)),key
            feature_shapes[key]=[list(x.shape) for x in aa]
        else:
            assert torch.equal(aa,bb),key
            feature_shapes[key]=list(aa.shape)
    fast.update(text)
    raw=model.forward_grounding(backbone_out=fast,find_input=processor.find_stage,
        geometric_prompt=model._get_dummy_prompt(),find_target=None,boxes_only=False)
    result={k:raw[k].float().cpu() for k in
            ('pred_boxes','pred_logits','pred_masks','presence_logit_dec')}
    scores=raw['pred_logits'].sigmoid()
    if not model.supervise_joint_box_scores:
        scores*=raw['presence_logit_dec'].sigmoid().unsqueeze(1)
    scores=scores.squeeze(-1)
    keep=scores>0.
    expected_p=F.interpolate(raw['pred_masks'][keep,None],(image.height,image.width),
                            mode='bilinear',align_corners=False).sigmoid()
    expected_boxes=box_cxcywh_to_xyxy(raw['pred_boxes'][keep])*torch.tensor(
        [image.width,image.height,image.width,image.height],device='cuda')
    state=dict(original_height=image.height,original_width=image.width,backbone_out=full)
    out=segment_group(processor,image,dict(crop=[0,0,image.width,image.height]),text,state)
    assert torch.equal(out['boxes'],expected_boxes)
    assert torch.equal(out['scores'],scores[keep])
    assert torch.equal(out['masks_logits'],expected_p)
    # Confirm that dynamic regeneration matches upstream's mathematical rule.
    from sam3.model.vitdet import compute_axial_cis
    for block in model.backbone.vision_backbone.trunk.blocks:
        attn=block.attn
        grid=attn.input_size[0]
        reference=compute_axial_cis(attn.head_dim,grid,grid,theta=attn.rope_theta,
                                   scale_pos=attn.rope_pt_size[0]/grid).to('cuda')
        assert torch.equal(attn.freqs_cis,reference)
    return result,dict(feature_shapes=feature_shapes,full_neck_exact=True,
                       processor_boxes_scores_probabilities_exact=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    started=time.monotonic()
    image=synthetic_image()
    from uav.sam3_model import build
    with torch.inference_mode():
        model=build()
        compact_weights(model.backbone)
    results={}; configurations={};observations={}
    # Exercise both directions and repeated return to the same grid.
    sequence=(1008,336,672,336,1008,672)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for size in sequence:
            set_resolution(model,size)
            configurations[str(size)]=model_configuration(model,size)
            output,details=forward(model,size,image)
            if size in results:
                repeat=compare(results[size],output)
                assert all(v['exact'] for v in repeat.values()),repeat
                observations.setdefault(str(size),{})['switch_round_trip']=repeat
            else:
                results[size]=output
            observations.setdefault(str(size),{}).update(details)
            print(json.dumps(dict(event='dynamic_resolution_checked',size=size)),flush=True)
    del model
    gc.collect();torch.cuda.empty_cache()
    independent={}
    for size in (336,672):
        with torch.inference_mode():
            model=independent_model(size)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            config=model_configuration(model,size)
            assert config==configurations[str(size)]
            output,details=forward(model,size,image)
            parity=compare(results[size],output)
            assert all(v['exact'] for v in parity.values()),parity
            independent[str(size)]=parity
        del model
        gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(event='independent_small_model_exact',size=size)),flush=True)
    report=dict(passed=True,seconds=time.monotonic()-started,synthetic_only=True,
                media_files_decoded=0,sequence=list(sequence),configurations=configurations,
                checks=observations,independent_model_parity=independent)
    Path(args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(event='resolution_audit_passed',seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
