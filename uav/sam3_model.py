from sam3.model_builder import build_sam3_image_model
from .settings import sam3_checkpoint
from .checkpoint_initialization import checkpoint_initialization

def build():
    import sam3.model.vitdet as vitdet
    import sam3.model_builder as builder
    with checkpoint_initialization(extra_initializers=[(vitdet, 'trunc_normal_')],
            factories=[(builder, '_create_vision_backbone'), (builder, '_create_text_encoder')]):
        model = build_sam3_image_model(checkpoint_path=str(sam3_checkpoint()), load_from_HF=False, device="cpu")
    return model.cuda().eval()
