"""Debug trajectories on synthetic in-memory canvases and pose records."""
from types import SimpleNamespace
import unittest
import numpy as np
from trajectory_reference import draw_trajectory
from uav.template_mask.geometry import translation
from uav.template_mask.records import Instance, Pose, Template
from uav.template_mask.review import trajectory_segments


class TrajectoryReviewTests(unittest.TestCase):
    def test_dashes_continue_across_dense_sample_boundaries(self):
        sparse = np.zeros((40, 150, 3), np.uint8)
        dense = sparse.copy()
        draw_trajectory(sparse, [(0, 10, 20), (120, 130, 20)], -1, (255, 255, 255), full_path_dashed=True)
        draw_trajectory(dense, [(f, 10+f, 20) for f in range(121)], -1, (255, 255, 255), full_path_dashed=True)
        # Interior dashes and gaps survive 1-pixel source sampling.
        self.assertTrue(sparse[20, 15].any()); self.assertFalse(sparse[20, 24].any())
        # Antialiasing may differ at intermediate polyline vertices, while
        # the full-strength row must have the same dash/gap pattern.
        np.testing.assert_array_equal(sparse[20] > 10, dense[20] > 10)

    def test_recent_tail_remains_solid_and_fades_by_elapsed_frames(self):
        image = np.zeros((40, 150, 3), np.uint8)
        draw_trajectory(image, [(0, 10, 20), (120, 130, 20)], 50, (255, 255, 255), full_path_dashed=True)
        # x=42 and x=24 are full-path gaps; only the recent tail fills them.
        self.assertGreater(image[20, 42, 0], image[20, 24, 0])
        self.assertGreater(image[20, 24, 0], 0)
        self.assertFalse(image[20, 78].any())  # Future path gap is still blank.

    def test_final_display_anchor_follows_extent_without_changing_tracking_anchor(self):
        template = Template(0, (0, 0, 20, 10), anchor=(4, 3))
        instance = Instance(0, 0, 0, poses={f: Pose(f, translation(f, 10), .5, 'measured') for f in range(3)})
        engine = SimpleNamespace(instances={0: instance}, bank=SimpleNamespace(templates={0: template},
                                 members=lambda tid:{0:np.eye(3)}),
                                 _renders_at=lambda i, f: True,associated_observations=lambda i,f:[])
        before = trajectory_segments(engine)
        template.box = (-100, -50, 200, 90)
        self.assertEqual(before[0][0], [(0, 10., 15.), (1, 11., 15.), (2, 12., 15.)])
        self.assertEqual(trajectory_segments(engine)[0][0],
                         [(0,50.,30.),(1,51.,30.),(2,52.,30.)])
        self.assertEqual(template.anchor,(4,3))

    def test_absence_and_hidden_frames_split_paths(self):
        template = Template(0, (0, 0, 20, 10), anchor=(4, 3))
        instance = Instance(0, 0, 0, poses={f: Pose(f, translation(f, 10), .5, 'measured') for f in range(8)})
        instance.poses[2].status = 'absent'
        del instance.poses[5]
        engine = SimpleNamespace(instances={0: instance}, bank=SimpleNamespace(templates={0: template},
                                 members=lambda tid:{0:np.eye(3)}),
                                 _renders_at=lambda i, f: f != 6,associated_observations=lambda i,f:[])
        segments = trajectory_segments(engine)[0]
        self.assertEqual([[s[0] for s in segment] for segment in segments], [[0, 1], [3, 4], [7]])


class DebugOutputTests(unittest.TestCase):
    def test_shrunken_template_display_anchor_agrees_across_member_coordinates(self):
        from uav.template_mask.review import display_anchor, final_marks
        from uav.template_mask.geometry import points
        e=self.make_engine()
        templates={0:Template(0,(100,20,120,30),anchor=(20,80)),
                   1:Template(1,(0,0,20,10),anchor=(80,80))}
        e.bank=SimpleNamespace(templates=templates,members=lambda tid:
            {0:translation(),1:translation(100,20)} if tid==0 else
            {0:translation(-100,-20),1:translation()})
        np.testing.assert_allclose(display_anchor(e,0),[110,25])
        np.testing.assert_allclose(display_anchor(e,1),[10,5])
        owner=Instance(0,1,0,poses={0:Pose(0,translation(30,40),.7,'measured')})
        e.instances={0:owner};e._renders_at=lambda i,f:True
        mark=final_marks(e,0)[0]
        np.testing.assert_allclose(mark['anchor'],[40,45])
        self.assertEqual(trajectory_segments(e)[0][0],[(0,40.,45.)])
        self.assertEqual(templates[1].anchor,(80,80))

    def test_review_box_and_path_use_the_same_observation_pose_as_mask(self):
        from uav.template_mask.review import final_marks
        e=self.make_engine()
        template=Template(0,(0,0,20,10),anchor=(4,3))
        owner=Instance(0,0,0,poses={0:Pose(0,translation(10,20),.7,'measured')})
        obs=SimpleNamespace(pose=translation(30,40),reliable=True,trajectory_evidence=None)
        e.instances={0:owner}
        e.bank=SimpleNamespace(templates={0:template},members=lambda tid:{0:np.eye(3)})
        e._renders_at=lambda i,f:True
        e.associated_observations=lambda i,f:[obs]
        np.testing.assert_allclose(final_marks(e,0)[0]['box'],[30,40,50,50])
        self.assertEqual(trajectory_segments(e)[0][0],[(0,40.,45.)])
        self.assertEqual(final_marks(e,0)[0]['anchor'],[40.,45.])

    def make_engine(self):
        from uav.template_mask.cache import EvidenceCache
        from uav.template_mask.config import TemplateConfig
        cache = EvidenceCache()
        return SimpleNamespace(debug=True, shape=(100, 160),
            config=TemplateConfig(sift_backend='cpu'),
            cache=cache, source_ids={0:100, 8:110}, discovery_groups={},
            first_pass_observations={}, first_pass_marks={}, instances={},
            associated_observations=lambda i,f:[])

    def test_sam_result_holds_eight_frames_and_clears_at_negative_sample(self):
        from uav.template_mask.review import stage1_mask
        e = self.make_engine()
        e.discovery_groups = {100:dict(groups=[dict(sources=['raw'])]),110:dict(groups=[])}
        e.cache.put(e.cache.key('raw_discovery',100,'raw'),
            probability=np.full((4,6),.8,np.float32),valid=np.ones((4,6),bool),
            origin=np.array([30,40]))
        baseline = stage1_mask(e,0)
        self.assertEqual(baseline.sum(),24)
        for frame in range(1,8):
            np.testing.assert_array_equal(stage1_mask(e,frame),baseline)
        for frame in range(8,16):
            self.assertFalse(stage1_mask(e,frame).any())

    def test_snapshot_keeps_original_labels_and_geometry_after_stage2_changes(self):
        from uav.template_mask.review import capture_first_frame, final_marks
        e = self.make_engine()
        template = Template(2,(0,0,20,10),anchor=(4,3))
        owner = Instance(5,2,0,poses={0:Pose(0,translation(30,40),.5,'measured')})
        e.instances={5:owner}
        e.bank=SimpleNamespace(templates={2:template},members=lambda tid:{2:np.eye(3)})
        e.frame_observations={0:['coarse']}
        e._renders_at=lambda i,f:True
        e.presence=SimpleNamespace(output_margin=lambda i,f:False)
        capture_first_frame(e,0)
        snapshot=dict(e.first_pass_marks[0][0])
        owner.poses[0].matrix=translation(50,60)
        template.box=(0,0,40,20)
        np.testing.assert_array_equal(e.first_pass_marks[0][0]['box'],snapshot['box'])
        self.assertEqual(e.first_pass_marks[0][0]['anchor'],[34.,43.])
        np.testing.assert_array_equal(final_marks(e,0)[0]['box'],[50,60,90,80])

    def test_final_labels_resolve_alias_and_merged_template_geometry(self):
        from uav.template_mask.review import final_marks
        e = self.make_engine()
        canonical=Instance(3,0,0,continuation_of=1)
        alias=Instance(7,2,0,alias_of=3,poses={4:Pose(4,translation(30,40),.5,'measured')})
        e.instances={3:canonical,7:alias}
        e.bank=SimpleNamespace(templates={0:Template(0,(0,0,10,10)),2:Template(2,(0,0,20,10))},
            members=lambda tid:{0:translation(-10,0),2:np.eye(3)})
        e._renders_at=lambda i,f:True
        marks=final_marks(e,4)
        self.assertEqual((marks[0]['instance'],marks[0]['template']),(1,0))
        np.testing.assert_array_equal(marks[0]['box'],[20,40,50,50])

    def test_template_overlay_uses_link_transform_and_keeps_original_background(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from uav.template_mask.review import write_template_overlays, write_page, review_layout
        e=self.make_engine()
        mean=np.zeros((10,20),np.float32);mean[2:5,3:8]=.8
        data=dict(mean=mean,origin=np.zeros(2),sources=np.array([100]),refined=np.array(True))
        obs=SimpleNamespace(id='fine',score=.9,frame=0,source_id=100,resolution=1008,
                            pose=translation(50,40))
        e.bank=SimpleNamespace(templates={0:Template(0,(0,0,20,10)),2:Template(2,(0,0,20,10))},
            members=lambda tid:{0:np.eye(3),2:translation(10,0)},
            output_fusion=lambda tid:(data,False),
            selected_views=lambda tid:{100:[(obs,translation(10,0))]})
        original=np.full((100,160,3),90,np.uint8)
        with tempfile.TemporaryDirectory() as root, patch('uav.template_mask.review.cv2.imwrite',return_value=True) as save:
            rows=write_template_overlays(e,lambda frame:original,Path(root))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['members'],[0,2])
            self.assertEqual(rows[0]['mask_pixels'],15)
            crop=rows[0]['crop'];overlay=save.call_args.args[1]
            expected=np.full_like(overlay,90)
            l,t,_,_=crop
            expected[42-t:45-t,43-l:48-l]=np.rint(np.array([90,90,90])*.55+np.array([60,220,70])*.45)
            np.testing.assert_array_equal(overlay,expected)
            write_page(Path(root),review_layout(e.shape),rows,False)
            page=(Path(root)/'index.html').read_text()
            self.assertIn('src="mask-review.mp4"',page)
            self.assertEqual(page.count('<video '),1)
            self.assertIn('src="template-0.png"',page)
            self.assertNotIn('同步播放',page)
            self.assertIn('T0, T2',page)


if __name__ == '__main__':
    unittest.main()
