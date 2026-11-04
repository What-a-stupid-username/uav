"""Flow-completion replay routing without media or CUDA execution."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from uav.cuda_graph import graph_flow_completion

class CompletionGraphTests(unittest.TestCase):
    def test_replay_and_fallback(self):
        original=Mock(return_value=('flow','edge'))
        module=SimpleNamespace(training=False,forward=original)
        replay=Mock(return_value='graph-flow')
        with patch('uav.cuda_graph.GraphReplay',return_value=replay):
            forward=graph_flow_completion(module)
        flows=SimpleNamespace(is_cuda=True,dtype=torch.float16,shape=(1,19,2,128,128))
        masks=object()
        with torch.inference_mode():
            self.assertEqual(forward(flows,masks),('graph-flow',None))
            module.training=True
            self.assertEqual(forward(flows,masks),('flow','edge'))
            module.training=False;flows.shape=(2,19,2,128,128)
            self.assertEqual(forward(flows,masks),('flow','edge'))
        self.assertIs(forward.replay,replay)
        self.assertEqual(replay.call_count,1)
