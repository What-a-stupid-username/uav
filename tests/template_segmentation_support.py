from types import SimpleNamespace
import unittest
from uav.template_mask.geometry import translation
from uav.template_mask.records import Instance, Pose, Template
from uav.template_mask.segmentation_support import assess
from uav.template_mask.trajectory_quality import observed_only


class SegmentationSupportTests(unittest.TestCase):
    def fixture(self):
        t=Template(0,(0,0,56,32))
        i=Instance(0,0,0,poses={f:Pose(f,translation(100,100),.7,'measured',f,(),True)
                              for f in range(57)})
        e=SimpleNamespace(shape=(400,600),bank=SimpleNamespace(templates={0:t},observations={}),
            frame_observations={},presence=SimpleNamespace(global_samples={f:(f,[]) for f in range(0,57,8)}))
        return e,i,t

    def test_continuous_sift_and_sam_disagreement_restricts_only_the_supported_interval(self):
        e,i,t=self.fixture()
        e.presence.global_samples[0]=(0,[(100,100,156,132)])
        e.presence.global_samples[48]=(48,[(100,100,156,132)])
        r=assess(e,i)
        self.assertEqual(r['intervals'][0]['first_frame'],8)
        self.assertEqual(r['intervals'][0]['last_frame'],40)
        i.segmentation_quality=r
        self.assertTrue(observed_only(i,24))
        self.assertFalse(observed_only(i,0))
        self.assertFalse(observed_only(i,48))
        self.assertEqual(len(i.poses),57)

    def test_template_status_and_refinement_do_not_replace_current_semantic_evidence(self):
        e,i,t=self.fixture()
        t.provisional=False;t.refinement_sources={0,8}
        self.assertEqual(assess(e,i)['action'],'restrict_intervals')

    def test_missing_sift_clipping_crossing_sparse_or_duplicate_sources_cannot_reject(self):
        for kind in ('missing','edge','crossing','few','same_source','ambiguous'):
            e,i,t=self.fixture()
            if kind=='missing':
                for f in range(1,57,8):i.poses[f].status='uncertain'
            elif kind=='edge':
                for p in i.poses.values():p.matrix=translation(100,-10)
            elif kind=='crossing':
                e.presence.global_samples={f:(f,[(150,110,200,130)]) for f in range(0,57,8)}
            elif kind=='few':e.presence.global_samples={0:(0,[]),8:(8,[])}
            elif kind=='same_source':e.presence.global_samples={f:(0,[]) for f in range(0,57,8)}
            else:i.ambiguous_frames.update(i.poses)
            self.assertEqual(assess(e,i)['action'],'retain',kind)

    def test_occasional_misses_reset_when_sam_detects_again(self):
        e,i,t=self.fixture()
        for f in (16,40):e.presence.global_samples[f]=(f,[(100,100,156,132)])
        self.assertEqual(assess(e,i)['action'],'retain')


if __name__=='__main__':unittest.main()
