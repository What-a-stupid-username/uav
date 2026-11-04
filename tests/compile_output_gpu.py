"""Compile and async output checks using only in-memory generated tensors."""
import json
import tempfile
from pathlib import Path
import torch
from uav.async_video_writer import AsyncVideoWriter

@torch.inference_mode()
def main():
    def operation(x):return x.sin() * 2 + x
    x=torch.randn(16,16,device='cuda')
    actual=torch.compile(operation,fullgraph=True)(x)
    torch.testing.assert_close(actual,operation(x))
    with tempfile.TemporaryDirectory() as tmp:
        writer=AsyncVideoWriter(Path(tmp)/'synthetic.mp4',128,128,30)
        for _ in range(6):writer.save(torch.zeros((128,128,3),dtype=torch.uint8,device='cuda'))
        writer.close()
        assert writer.frames==6
        # Generated media is never reopened or inspected.
    print(json.dumps({'torch_compile':'passed','async_encoding':'passed'}))
if __name__=='__main__':main()
