"""CPU-only SAM construction profiling, without reading media or weights."""
import cProfile
import pstats
import torch
from sam3.model_builder import build_sam3_image_model
torch.set_num_threads(4)
profile=cProfile.Profile()
profile.enable()
model=build_sam3_image_model(checkpoint_path=None,load_from_HF=False,device='cpu')
profile.disable()
pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(22)
