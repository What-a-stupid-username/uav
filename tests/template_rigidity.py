import unittest
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation
from uav.template_mask.rigidity import RigidityEvidence


class RigidityTests(unittest.TestCase):
    def models(self, offset=80, shared=False):
        return [Match(translation(),.7,12,.3,reference_indices=tuple(range(12))),
                Match(translation(offset,0),.7,12,.3,
                      reference_indices=tuple(range(12)) if shared else tuple(range(12,24)))]

    def test_disjoint_parts_need_repeated_originals_with_changing_layout(self):
        evidence=RigidityEvidence();box=(0,0,100,40)
        self.assertFalse(evidence.observe('original',1,self.models(),box))
        self.assertFalse(evidence.observe('original',1,self.models(180),box))
        self.assertFalse(evidence.observe('original',2,self.models(),box))
        self.assertTrue(evidence.observe('original',3,self.models(180),box))

    def test_repeated_instances_matching_same_points_are_not_composite(self):
        evidence=RigidityEvidence();box=(0,0,100,40)
        for source in range(10):
            self.assertFalse(evidence.observe('original',source,self.models(80+source*50,True),box))

    def test_rigid_fragments_and_single_visible_fragment_remain_valid(self):
        evidence=RigidityEvidence();box=(0,0,100,40)
        for source in range(10):
            self.assertFalse(evidence.observe('original',source,self.models(),box))
            self.assertFalse(evidence.observe('fragment',source,self.models()[:1],box))

    def test_reversed_candidate_order_does_not_invent_layout_change(self):
        evidence=RigidityEvidence();box=(0,0,100,40)
        self.assertFalse(evidence.observe('original',1,self.models(),box))
        self.assertFalse(evidence.observe('original',2,list(reversed(self.models())),box))


if __name__=='__main__':unittest.main()
