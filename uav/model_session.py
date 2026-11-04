"""Lazy GPU-resident models shared across sequential segments and videos."""
import gc
import time
import torch


class ModelSession:
    def __init__(self):
        self.sam_model = self.repair_model = None
        self.events = []
        self.detector = None
        self.phase = None

    def _record(self, phase, started, loaded):
        self.events.append(dict(phase=phase,loaded=loaded,seconds=time.monotonic()-started,
                                allocated_bytes=torch.cuda.memory_allocated()))

    @torch.inference_mode()
    def sam(self):
        started=time.monotonic()
        loaded=self.sam_model is None
        if loaded:
            from .sam3_model import build
            from .inference_weights import compact_weights
            torch.set_num_threads(4)
            self.sam_model=build()
            compact_weights(self.sam_model.backbone)
        if self.phase == 'repair':
            self._clear_repair_graphs()
        self.phase='sam'
        self._record('sam',started,loaded)
        return self.sam_model

    def _clear_repair_graphs(self):
        for name in ('model','fix_flow_complete','fix_raft'):
            for module in getattr(self.repair_model,name).modules():
                clear=getattr(module.forward,'clear_templates',None)
                if clear is not None:clear()
                replay=getattr(module.forward,'replay',None)
                if replay is not None:replay.clear()

    @torch.inference_mode()
    def repair(self):
        started=time.monotonic()
        loaded=self.repair_model is None
        if self.phase == 'sam':
            replay=getattr(self.sam_model.forward_grounding,'replay',None)
            if replay is not None:replay.clear()
        if loaded:
            from .repair_model import build_repair_model
            self.repair_model=build_repair_model()
        self.phase='repair'
        self._record('repair',started,loaded)
        return self.repair_model

    def detection(self, cache, config):
        """Keep text/stream/processor storage alive for captured SAM graphs."""
        from .template_mask.sam import SamDetector
        model=self.sam_model if self.sam_model is not None else self.sam()
        if self.detector is None:
            self.detector=SamDetector(model,cache,config)
        else:
            self.detector.cache=cache
            self.detector.config=config
            self.detector.calls=self.detector.cache_hits=0
            self.detector.task_calls=dict(global_coarse=0,local=0,refinement=0,boundary_full=0)
        return self.detector

    def close(self):
        # Drop the entire session only after the video/batch completes. CUDA
        # weights retain their addresses throughout ordinary reuse.
        self.detector=None
        self.sam_model=self.repair_model=None
        gc.collect()
        torch.cuda.empty_cache()
