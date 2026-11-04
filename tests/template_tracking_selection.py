"""Image corroboration precedes proximity to a predicted search pose."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.config import TemplateConfig
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Instance,Template
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation

class SelectionTests(unittest.TestCase):
    def test_corroborated_fit_wins_but_still_obeys_spatial_gate(self):
        with tempfile.TemporaryDirectory() as root:
            cache=EvidenceCache();self.addCleanup(cache.close)
            e=TemplateEngine(TemplateConfig(sift_backend='cpu'),cache,None,
                WarningLog(Path(root)/'warnings.jsonl'),(400,500))
            t=Template(0,(0,0,100,40));e.bank.templates[0]=t
            e.bank.observations={r:SimpleNamespace(source_id=s,reliable=True,ambiguous=False)
                                 for r,s in (('a',1),('b',2))}
            owner=Instance(0,0,0);e.instances[0]=owner
            owner.motion=SimpleNamespace(evidence=[])
            cache.put('target',xy=np.zeros((20,2)))
            weak=Match(translation(100,100),.7,7,.01,('a',))
            for shift,expected in ((120,120),(400,100)):
                strong=Match(translation(shift,100),.7,200,.9,('a','b'))
                with (patch.object(e,'_prediction',return_value=(translation(100,100),1.,True)),
                      patch.object(e,'_matches',return_value=[weak,strong]),
                      patch.object(e.matcher,'extract',return_value='target')):
                    e._track(owner,4,10,np.zeros((400,500,3),np.uint8),image_only=True)
                self.assertAlmostEqual(owner.poses[4].matrix[0,2],expected)
                self.assertEqual(owner.poses[4].status,'measured')
