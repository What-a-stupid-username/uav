"""External model paths; weights are never bundled or downloaded implicitly."""
import os
from pathlib import Path

def model_root():
    return Path(os.environ.get('UAV_MODEL_DIR', str(Path(__file__).resolve().parents[1]/'models')))

def sam3_checkpoint():
    return Path(os.environ.get('SAM3_CHECKPOINT', str(model_root()/'sam3/sam3.pt')))

def propainter_models():
    return Path(os.environ.get('PROPAINTER_MODEL_DIR', str(model_root()/'propainter')))

def validate_models(*, repair=True):
    required = [sam3_checkpoint()]
    if repair:
        required += [propainter_models()/n for n in
                     ('raft-things.pth','recurrent_flow_completion.pth','ProPainter.pth')]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError('Missing model weights: '+', '.join(missing))
