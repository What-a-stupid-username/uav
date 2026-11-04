"""Bounded CUDA graph replay for fixed tensor signatures during inference."""
from collections import OrderedDict
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

# Each replay owner has its own allocator pool. SAM and ProPainter alternate
# across segments and must never alias each other's static graph storage.


class GraphReplay:
    def __init__(self, function, capacity=8, context=None):
        self.function = function
        self.capacity = capacity
        self.context = context or (lambda: None)
        self.cache = OrderedDict()
        self.pool = None
        self.captures = self.replays = self.fallbacks = 0

    def clear(self):
        """Release captures and allocator ownership together at a phase boundary."""
        torch.cuda.synchronize()
        self.cache.clear()
        self.last_static=None
        self.pool=None

    def __call__(self, *args, **kwargs):
        return self.run(args,kwargs)

    def run(self,args,kwargs,prepare=None):
        leaves, spec = tree_flatten((args, kwargs))
        if torch.is_grad_enabled():
            return self.function(*args, **kwargs)
        key = (str(spec), self.context(), torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'), tuple(
            ('tensor', tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
            if isinstance(x, torch.Tensor) else ('value', type(x), x) for x in leaves))
        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                # Keep memory bounded without repeatedly evicting hot signatures.
                self.fallbacks += 1
                self.last_static=None
                if prepare is not None:
                    prepare(leaves)
                return self.function(*args, **kwargs)
            static = [(torch.empty_like(x,memory_format=torch.preserve_format) if prepare is not None
                       else x.clone(memory_format=torch.preserve_format))
                      if isinstance(x,torch.Tensor) else x for x in leaves]
            if prepare is not None:
                prepare(static)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call_args, call_kwargs = tree_unflatten(static, spec)
                    output = self.function(*call_args, **call_kwargs)
                    del output
            torch.cuda.current_stream().wait_stream(stream)
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                call_args, call_kwargs = tree_unflatten(static, spec)
                output = self.function(*call_args, **call_kwargs)
            self.cache[key] = (static, graph, output)
            self.captures += 1
        static, graph, output = self.cache[key]
        self.last_static=static
        if prepare is not None:
            prepare(static)
        else:
            for target, source in zip(static, leaves):
                if isinstance(source, torch.Tensor):
                    target.copy_(source)
        graph.replay()
        self.replays += 1
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output)


def graph_grounding(model, *, eager_boxes_only=False):
    """Capture grounding only; visual features and variable-size postprocess stay outside."""
    from sam3.model.geometry_encoders import Prompt
    from sam3.model.data_misc import FindStage
    original = model.forward_grounding
    decoder = model.transformer.decoder
    from .sam3_resolution import use_configured_rpb_size
    use_configured_rpb_size(model)

    def operation(backbone, find_fields, prompt_fields, boxes_only):
        prompt = Prompt.__new__(Prompt)
        prompt.__dict__.update(prompt_fields)
        find = FindStage(**find_fields)
        result = original(backbone_out=backbone, find_input=find, geometric_prompt=prompt, find_target=None, boxes_only=boxes_only)
        return {key: result[key] for key in ('pred_boxes', 'pred_logits', 'pred_masks', 'presence_logit_dec') if key in result}

    replay = GraphReplay(operation, capacity=8, context=lambda: decoder.compilable_stored_size)

    def forward(backbone_out, find_input, find_target, geometric_prompt, boxes_only=False, **kwargs):
        # Switching captured segmentation/boxes-only signatures corrupts cached
        # GPU state in this runtime. Keep presence-only calls eager; full
        # segmentation retains graph replay and its original storage lifetime.
        if (eager_boxes_only and boxes_only) or model.training or find_target is not None or kwargs:
            return original(backbone_out, find_input, find_target, geometric_prompt, boxes_only=boxes_only, **kwargs)
        backbone = {key: backbone_out[key] for key in
                    ('backbone_fpn', 'vision_pos_enc', 'language_features', 'language_mask')}
        return replay(backbone, vars(find_input), vars(geometric_prompt), boxes_only)
    forward.replay = replay
    return forward


def graph_flow_completion(module):
    """Replay inference flow predictions; training still computes edge outputs."""
    original = module.forward
    replay = GraphReplay(lambda flows, masks: original(flows, masks)[0], capacity=8)
    def forward(flows, masks):
        if (module.training or torch.is_grad_enabled() or not flows.is_cuda
                or flows.dtype != torch.float16 or flows.shape[0] != 1):
            return original(flows, masks)
        return replay(flows, masks), None
    forward.replay = replay
    return forward


def graph_propagation(module):
    """The repair generator consumes only the fused local feature output."""
    original=module.forward
    def fused(*args,**kwargs):
        return original(*args,**kwargs,fused_only=True)[2]
    replay=GraphReplay(fused,capacity=8)
    def forward(*args,**kwargs):
        if (module.training or torch.is_grad_enabled() or args[0].shape[0]!=1
                or args[0].dtype!=torch.float16):
            return original(*args,**kwargs)
        return None,None,replay(*args,**kwargs),None
    templates=OrderedDict()
    def prepared(local,flows,masks,updated,interpolation):
        b,t,c,h,w=local.shape
        # Metadata-only views select a graph signature; producer fills graph-owned
        # buffers directly, rather than allocating resize/cat outputs then copying.
        key=(b,t,c,h,w,local.dtype,local.device,tuple(local.stride()),interpolation)
        if key in templates:
            flow,backward,mask=templates[key]
        else:
            flow=local.new_empty(b,t-1,2,h,w)
            backward=torch.empty_like(flow)
            mask=local.new_empty(b,t,2,h,w)
        args=(local,flow,backward,mask,interpolation)
        def produce(targets):
            targets[0].copy_(local)
            for dst,src in zip(targets[1:3],flows):
                torch.ops.aten.upsample_bilinear2d.out(src.reshape(-1,2,*src.shape[-2:]),
                    [h,w],False,scales_h=.25,scales_w=.25,out=dst.view(-1,2,h,w))
                dst.div_(4.)
            for index,src in enumerate((masks,updated)):
                torch.ops.aten.upsample_nearest2d.out(src[:,:t].reshape(-1,1,*src.shape[-2:]),
                    [h,w],scales_h=.25,scales_w=.25,
                    out=targets[3][:,:,index:index+1].reshape(b*t,1,h,w))
        if b!=1 or local.dtype!=torch.float16:
            produce(list(args))
            return original(*args,fused_only=True)
        result=replay.run(args,{},prepare=produce)
        # Static input storage belongs to the graph cache. Reuse it for signature
        # selection too; do not allocate scratch flow/mask tensors on every call.
        if key not in templates and replay.cache:
            static=getattr(replay,'last_static',None)
            if static is not None and static[1].data_ptr()!=flow.data_ptr():
                if tuple(static[0].shape)==tuple(local.shape) and static[1].shape==flow.shape:
                    templates[key]=tuple(static[1:4])
        return None,None,result,None
    forward.clear_templates=templates.clear
    forward.prepared=prepared
    forward.replay=replay
    return forward
