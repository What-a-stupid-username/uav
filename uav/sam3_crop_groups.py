"""Detection-only crop encoding and grounding for shared-template refinement."""
from .sam3_gpu_image import image_tensor,processor_image


def set_detection_images(processor, images):
    """Encode crops without the unused tracker neck; leave full-frame encoding intact."""
    import torch
    backbone = processor.model.backbone
    neck = backbone.vision_backbone
    if not hasattr(neck, 'sam2_convs'):
        raise TypeError('detection-only crops require the dual SAM3 neck')
    tensors = [processor.transform(image_tensor(image,processor.device)) for image in images]
    samples = tensors[0].unsqueeze(0) if len(tensors) == 1 else torch.stack(tensors)
    x = neck.trunk(samples)[-1]
    # Keep the same SAM3 operations/order as the dual neck, including scalp.
    features = []
    positions = []
    # simpleFPN branches are independent. Scalp discards trailing branches,
    # so do not execute their convolutions/position encodings in this path.
    convolutions = neck.convs[:-backbone.scalp] if backbone.scalp > 0 else neck.convs
    for conv in convolutions:
        feature = conv(x)
        features.append(feature)
        positions.append(neck.position_encoding(feature).to(feature.dtype))
    return dict(vision_features=features[-1],vision_pos_enc=positions,
                backbone_fpn=features,sam2_backbone_out=None)


def segment_group(processor, image, group, text, state=None):
    # Only current or motion-verified boxes may become positive prompts.
    if state is None:
        state = processor.set_image(processor_image(image.crop(group['crop'])))
    state['backbone_out'].update(text)
    state['geometric_prompt'] = processor.model._get_dummy_prompt()
    if group.get('prompts'):
        import torch
        prompts=torch.tensor(group['prompts'],device=processor.device,dtype=torch.float32).unsqueeze(1)
        state['geometric_prompt'].append_boxes(prompts,torch.ones(prompts.shape[:2],device=processor.device,dtype=torch.bool))
    return processor._forward_grounding(state)
