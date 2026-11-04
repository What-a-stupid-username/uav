"""FP16 RAFT subnetworks with persistent engines and bounded shared workspace."""
import copy
import hashlib
import json
import os
from pathlib import Path

import torch


class ExplicitInstanceNorm(torch.nn.Module):
    """FP32 statistics avoid TensorRT's faulty dynamic InstanceNorm tactic."""
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        value = x.float()
        centered = value - value.mean((2, 3), keepdim=True)
        variance = centered.square().mean((2, 3), keepdim=True)
        return (centered * torch.rsqrt(variance + self.epsilon)).to(x.dtype)


def explicit_norms(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.InstanceNorm2d):
            if child.affine or child.track_running_stats:
                raise ValueError('RAFT export requires stateless non-affine InstanceNorm')
            setattr(module, name, ExplicitInstanceNorm(child.eps))
        else:
            explicit_norms(child)
    return module


class ExportUpdate(torch.nn.Module):
    def __init__(self, module, upsample):
        super().__init__()
        self.module = module
        self.upsample = upsample

    def forward(self, net, inp, corr, flow):
        result = self.module(net, inp, corr, flow, upsample=self.upsample)
        return result if self.upsample else (result[0], result[2])


def cached_engine(name, module, channels, checkpoint, *, shape_ranges=None,
                  cache_group='raft', source_files=()):
    import tensorrt as trt
    from filelock import FileLock
    stat = Path(checkpoint).stat()
    identity = dict(version=1, source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    weights=(stat.st_size, stat.st_mtime_ns), trt=trt.__version__,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    if shape_ranges is not None:
        identity.update(shapes=shape_ranges, sources=[hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                    for p in source_files])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(__file__).resolve().parents[1]/'tmp/tensorrt'/cache_group/key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{name}.engine'
    with FileLock(str(directory/'build.lock')):
        if path.exists():
            return path
        recurrent = len(channels) > 1
        side = 32 if recurrent else 256
        ranges = shape_ranges or [((1,c,16,16), (4,c,side,side), (24,c,33,33))
                                  if recurrent else ((1,c,128,128), (4,c,side,side), (24,c,264,264))
                                  for c in channels]
        sample = tuple(torch.zeros(optimum, device='cuda', dtype=torch.float16)
                       for _, optimum, _ in ranges)
        outputs = module(*sample)
        count = len(outputs) if isinstance(outputs, tuple) else 1
        axes = {f'x{i}': {0: 'batch', 2: 'height', 3: 'width'} for i in range(len(sample))}
        axes.update({f'y{i}': {0: 'batch', 2: 'out_height', 3: 'out_width'} for i in range(count)})
        onnx = directory/f'{name}.onnx'
        with torch.inference_mode():
            torch.onnx.export(module, sample, str(onnx), opset_version=17, dynamo=False,
                              do_constant_folding=False, dynamic_axes=axes,
                              input_names=[f'x{i}' for i in range(len(sample))],
                              output_names=[f'y{i}' for i in range(count)])
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError('RAFT export failed: '+ '; '.join(
                str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        profile = builder.create_optimization_profile()
        for i, (minimum, optimum, maximum) in enumerate(ranges):
            profile.set_shape(f'x{i}', minimum, optimum, maximum)
        config.add_optimization_profile(profile)
        timing = directory/'timing.cache'
        config.set_timing_cache(config.create_timing_cache(timing.read_bytes() if timing.exists() else b''), False)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f'RAFT TensorRT build failed: {name}')
        temporary = path.with_suffix('.engine.tmp')
        temporary.write_bytes(serialized)
        os.replace(temporary, path)
        timing.write_bytes(config.get_timing_cache().serialize())
        (directory/'identity.json').write_text(json.dumps(identity, indent=2))
    return path


class Workspace:
    """One sequential caller stream owns all RAFT contexts and their scratch memory."""
    def __init__(self):
        self.storage = None
        self.stream = None
        self.event = None

    def bind(self, context, engine, stream):
        if self.stream is not None and self.stream != stream.cuda_stream:
            stream.wait_event(self.event)
        size = engine.device_memory_size_v2
        if self.storage is None or self.storage.numel() < size:
            if self.event is not None:
                self.event.synchronize()
            self.storage = torch.empty(size, device=stream.device, dtype=torch.uint8)
        self.storage.record_stream(stream)
        context.set_device_memory(self.storage.data_ptr(), self.storage.numel())
        self.stream = stream.cuda_stream

    def finish(self, stream):
        if self.event is None:
            self.event = torch.cuda.Event()
        self.event.record(stream)


class Engine:
    def __init__(self, path, workspace):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Cannot load RAFT engine: {path}')
        self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if self.context is None:
            raise RuntimeError('Cannot create RAFT execution context')
        self.outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                        if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        self.workspace = workspace
        self.calls = 0

    def __call__(self, *inputs):
        stream = torch.cuda.current_stream(inputs[0].device)
        # A context must not be reshaped while a previous stream still uses it.
        if self.workspace.stream is not None and self.workspace.stream != stream.cuda_stream:
            self.workspace.event.synchronize()
        inputs = tuple(x.half().contiguous() for x in inputs)
        for i, x in enumerate(inputs):
            if not self.context.set_input_shape(f'x{i}', tuple(x.shape)):
                raise ValueError(f'Unsupported RAFT input: {tuple(x.shape)}')
            self.context.set_tensor_address(f'x{i}', x.data_ptr())
        outputs = [torch.empty(tuple(self.context.get_tensor_shape(name)), device=inputs[0].device,
                               dtype=torch.float16) for name in self.outputs]
        for name, output in zip(self.outputs, outputs):
            self.context.set_tensor_address(name, output.data_ptr())
        self.workspace.bind(self.context, self.engine, stream)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('RAFT TensorRT execution failed')
        self.workspace.finish(stream)
        for x in inputs:
            x.record_stream(stream)
        self.calls += 1
        return tuple(outputs)


def supported(inputs, channels, minimum, maximum):
    return all(x.is_cuda and x.ndim == 4 and x.dtype in (torch.float16, torch.float32)
               and x.shape[1] == c and 1 <= x.shape[0] <= 24
               and all(minimum <= size <= maximum for size in x.shape[2:])
               for x, c in zip(inputs, channels)) and len(inputs) == len(channels)


@torch.inference_mode()
def install_raft(raft, checkpoint):
    workspace = Workspace()
    engines = {}
    for name in ('fnet', 'cnet'):
        module = getattr(raft, name)
        export = copy.deepcopy(module).half()
        if name == 'fnet':
            explicit_norms(export)
        engine = Engine(cached_engine(name, export, (3,), checkpoint), workspace)
        engines[name] = engine
        original = module.forward
        def forward(x, engine=engine, original=original):
            split = isinstance(x, (list, tuple))
            image = torch.cat(x) if split else x
            if not supported((image,), (3,), 128, 264):
                return original(x)
            out = engine(image)[0]
            return out.split([value.shape[0] for value in x]) if split else out
        module.forward = forward
    module = raft.update_block
    original = module.forward
    export = copy.deepcopy(module).half()
    # Export the module method, never the torch.compile closure.
    export.forward = type(module).forward.__get__(export, type(module))
    export.channels_last_inputs = False
    for upsample in (False, True):
        name = 'update-mask' if upsample else 'update'
        engines[name] = Engine(cached_engine(name, ExportUpdate(export, upsample),
                                            (128, 128, 324, 2), checkpoint), workspace)
    def update(net, inp, corr, flow, upsample=True):
        inputs = (net, inp, corr, flow)
        if not supported(inputs, (128, 128, 324, 2), 16, 33):
            return original(*inputs, upsample=upsample)
        result = engines['update-mask' if upsample else 'update'](*inputs)
        return result if upsample else (result[0], None, result[1])
    module.forward = update
    raft.trt_backends = engines
