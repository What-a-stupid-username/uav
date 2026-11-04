"""Optional cached TensorRT visual encoder; other resolutions retain PyTorch."""
from pathlib import Path

import torch


class TensorRTEncoder:
    def __init__(self, path):
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError('TensorRT backend requires the tensorrt-cu12 package') from error
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot deserialize SAM TensorRT engine: {path}')
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        outputs = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        if inputs != ['image'] or outputs != ['features']:
            raise ValueError('Expected a SAM encoder engine with image input and features output')
        self.shape = tuple(self.engine.get_tensor_shape('image'))
        output_shape = tuple(self.engine.get_tensor_shape('features'))
        if self.shape != (1, 3, 1008, 1008) or output_shape != (1, 1024, 72, 72):
            raise ValueError('Expected a static 1008 SAM encoder engine')
        if self.engine.get_tensor_dtype('image') != trt.float32 or self.engine.get_tensor_dtype('features') != trt.float32:
            raise ValueError('Expected FP32 encoder input/output bindings')
        if any(self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR for name in names):
            raise ValueError('Expected linear TensorRT input/output bindings')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot allocate TensorRT execution context')
        self.device = torch.cuda.current_device()
        self.output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address('features', self.output.data_ptr())

    def __call__(self, image):
        if not image.is_cuda or image.device.index != self.device or image.dtype != torch.float32:
            raise ValueError('TensorRT encoder requires FP32 input on its CUDA device')
        if tuple(image.shape) != self.shape:
            raise ValueError('Unsupported TensorRT encoder input shape')
        image = image.contiguous()
        stream = torch.cuda.current_stream(image.device)
        self.context.set_tensor_address('image', image.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('SAM TensorRT encoder execution failed')
        # The temporary NCHW copy may be released before the asynchronous engine
        # completes. Record its use on the caller's stream without a host wait.
        image.record_stream(stream)
        return [self.output]


def install_encoder(model, path):
    """Keep the original trunk/parameters for resolution changes and fallback."""
    trunk = model.backbone.vision_backbone.trunk
    if getattr(trunk, '_uav_trt_path', None) is not None:
        if trunk._uav_trt_path != str(Path(path).resolve()):
            raise ValueError('Create a new model session to change TensorRT engine')
        return
    backend = TensorRTEncoder(path)
    original = trunk.forward
    def forward(image):
        if isinstance(image, torch.Tensor) and tuple(image.shape) == backend.shape:
            return backend(image)
        return original(image)
    trunk.forward = forward
    trunk._uav_trt_path = str(Path(path).resolve())
    trunk._uav_trt_encoder = backend
