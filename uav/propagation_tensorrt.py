"""FP16 convolution islands captured by ProPainter's feature-propagation graph."""
from pathlib import Path
import torch
from .raft_tensorrt import cached_engine


class ConvolutionEngine:
    """Keep context state per shape: captured graphs retain context-owned pointers."""
    def __init__(self, path, channels, max_batch, capacity=8):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot load propagation engine: {path}')
        self.channels, self.max_batch, self.capacity = channels, max_batch, capacity
        self.contexts = {}
        self.workspace = None

    def supports(self, image):
        return (image.is_cuda and image.dtype == torch.float16 and image.ndim == 4
                and image.shape[1] == self.channels and 1 <= image.shape[0] <= self.max_batch
                and all(8 <= size <= 130 for size in image.shape[2:])
                and (tuple(image.shape) in self.contexts or len(self.contexts) < self.capacity))

    def __call__(self, image):
        import tensorrt as trt
        image = image.contiguous()
        key = tuple(image.shape)
        if key not in self.contexts:
            context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            if context is None:
                raise RuntimeError('Cannot create propagation execution context')
            if not context.set_input_shape('x0', key):
                raise ValueError(f'Unsupported propagation shape: {key}')
            self.contexts[key] = context
        context = self.contexts[key]
        # Storage is allocated once before any graph capture, and never resized.
        context.set_device_memory(self.workspace.data_ptr(), self.workspace.numel())
        output = torch.empty(tuple(context.get_tensor_shape('y0')), device=image.device, dtype=image.dtype)
        context.set_tensor_address('x0', image.data_ptr())
        context.set_tensor_address('y0', output.data_ptr())
        stream = torch.cuda.current_stream(image.device)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Propagation TensorRT execution failed')
        image.record_stream(stream)
        return output


@torch.inference_mode()
def install_propagation(model, checkpoint):
    """Install before graph warmup; all islands run sequentially on the caller stream."""
    propagation = model.feat_prop_module
    targets = [(f'offset-{name}', alignment.conv_offset, 261, 1)
               for name, alignment in propagation.deform_align.items()]
    targets += [(f'backbone-{name}', module, 258, 1)
                for name, module in propagation.backbone.items()]
    targets.append(('fuse', propagation.fuse, 258, 10))
    backends = {}
    for name, module, channels, batch in targets:
        shapes = [((1,channels,8,8), (batch,channels,128,128), (batch,channels,130,130))]
        path = cached_engine(name, module, (channels,), checkpoint, shape_ranges=shapes,
                             cache_group='propagation', source_files=(__file__,))
        backend = ConvolutionEngine(path, channels, batch)
        original = module.forward
        def forward(image, backend=backend, original=original):
            return backend(image) if backend.supports(image) else original(image)
        module.forward = forward
        backends[name] = backend
    # These modules share one owner stream, including inside CUDA Graph replay.
    # Separate from RAFT/SAM workspaces, whose lifetimes can overlap graph storage.
    size = max(backend.engine.device_memory_size_v2 for backend in backends.values())
    workspace = torch.empty(size, device=next(propagation.parameters()).device, dtype=torch.uint8)
    for backend in backends.values():
        backend.workspace = workspace
    model.propagation_trt_backends = backends
