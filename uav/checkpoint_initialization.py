"""Skip random initialization only when checkpoint loading covers every tensor.

Construction is synchronous, before inference threads start. Constant and
derived positional initialization remains intact. Patches are always restored.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch


@contextmanager
def checkpoint_initialization(*, extra_initializers=(), factories=()):
    skipped = {}
    loaded = set()
    load_state_dict = torch.nn.Module.load_state_dict
    active = not factories

    def scoped(factory):
        def construct(*args, **kwargs):
            nonlocal active
            previous, active = active, True
            try:
                return factory(*args, **kwargs)
            finally:
                active = previous
        return construct

    def initializer(original):
        def initialize(tensor, *args, **kwargs):
            if not active:
                return original(tensor, *args, **kwargs)
            skipped[id(tensor)] = tensor
            return tensor
        return initialize

    def load(module, state_dict, *args, **kwargs):
        result = load_state_dict(module, state_dict, *args, **kwargs)
        missing = set(result.missing_keys)
        for name, tensor in module.state_dict(keep_vars=True).items():
            if name in state_dict and name not in missing:
                loaded.add(id(tensor))
        return result

    with ExitStack() as stack:
        for name in ('uniform_', 'normal_', 'trunc_normal_', 'kaiming_uniform_',
                     'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_', 'orthogonal_'):
            stack.enter_context(patch.object(torch.nn.init, name, initializer(getattr(torch.nn.init, name))))
        for module, name in extra_initializers:
            stack.enter_context(patch.object(module, name, initializer(getattr(module, name))))
        for module, name in factories:
            stack.enter_context(patch.object(module, name, scoped(getattr(module, name))))
        stack.enter_context(patch.object(torch.nn.Module, 'load_state_dict', load))
        yield
        uncovered = set(skipped) - loaded
        if uncovered:
            shapes = [tuple(skipped[key].shape) for key in uncovered]
            raise RuntimeError('Checkpoint did not cover skipped random initialization: ' + str(shapes))
