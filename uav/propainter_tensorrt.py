"""Cached FP16 ProPainter encoder; model evidence never enters the cache."""
import hashlib
import json
import os
from pathlib import Path

import torch


class ExportEncoder(torch.nn.Module):
    """Explicit channels avoid ambiguous TRT grouped-reshape inference."""
    def __init__(self, encoder):
        super().__init__()
        self.layers = encoder.layers

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                skip = out
            if i > 8 and i % 2 == 0:
                groups = (1, 2, 4, 8, 1)[(i-8)//2]
                left = skip.reshape(skip.shape[0], groups, 256//groups, *skip.shape[2:])
                right = out.reshape(out.shape[0], groups, (layer.in_channels-256)//groups, *out.shape[2:])
                out = torch.cat((left, right), 2).reshape(out.shape[0], layer.in_channels, *out.shape[2:])
            out = layer(out)
        return out


def cached_engine(encoder, checkpoint, trt):
    """Build once per model/runtime/GPU, with atomic publication and a lock."""
    from filelock import FileLock
    stat = Path(checkpoint).stat()
    identity = dict(version=1, source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weights=(stat.st_size, stat.st_mtime_ns), trt=trt.__version__,
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(__file__).resolve().parents[1]/'tmp/tensorrt/propainter'/key
    directory.mkdir(parents=True, exist_ok=True)
    engine = directory/'encoder.engine'
    with FileLock(str(directory/'build.lock')):
        if engine.exists():
            return engine
        onnx = directory/'encoder.onnx'
        sample = torch.zeros((1,5,256,256), device=next(encoder.parameters()).device, dtype=torch.float16)
        with torch.inference_mode():
            torch.onnx.export(ExportEncoder(encoder), sample, str(onnx),
                input_names=['image'], output_names=['features'], opset_version=17, dynamo=False,
                dynamic_axes={'image':{0:'batch',2:'height',3:'width'},
                              'features':{0:'batch',2:'feature_height',3:'feature_width'}})
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError('ProPainter encoder export failed: '+
                '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<30)
        profile = builder.create_optimization_profile()
        profile.set_shape('image', (1,5,8,8), (4,5,264,264), (4,5,520,520))
        config.add_optimization_profile(profile)
        timing = directory/'timing.cache'
        config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), False)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('ProPainter TensorRT encoder build failed')
        temporary = directory/'encoder.engine.tmp'
        temporary.write_bytes(serialized)
        os.replace(temporary, engine)
        timing.write_bytes(config.get_timing_cache().serialize())
        (directory/'identity.json').write_text(json.dumps(identity, indent=2))
    return engine


class TensorRTEncoder:
    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot load ProPainter encoder: {path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot create ProPainter encoder context')

    def __call__(self, image):
        image = image.contiguous()
        if not self.context.set_input_shape('image', tuple(image.shape)):
            raise ValueError(f'Unsupported ProPainter encoder shape: {tuple(image.shape)}')
        output = torch.empty(tuple(self.context.get_tensor_shape('features')),
                             dtype=torch.float16, device=image.device)
        self.context.set_tensor_address('image', image.data_ptr())
        self.context.set_tensor_address('features', output.data_ptr())
        stream = torch.cuda.current_stream(image.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('ProPainter TensorRT encoder execution failed')
        image.record_stream(stream)
        return output


def install_encoder(model, checkpoint):
    import tensorrt as trt
    encoder = model.encoder
    backend = TensorRTEncoder(cached_engine(encoder, checkpoint, trt))
    original = encoder.forward
    def forward(image):
        if (image.is_cuda and image.dtype == torch.float16 and image.ndim == 4
                and image.shape[1] == 5 and 1 <= image.shape[0] <= 4
                and all(8 <= size <= 520 for size in image.shape[2:])):
            return backend(image)
        return original(image)
    encoder.forward = forward
    encoder.trt_backend = backend
