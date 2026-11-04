"""Synthetic arrays only: no image/video files are read, including generated media."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import SiftMatcher
from uav.template_mask.geometry import translation, discrepancy, points
from uav.template_mask.motion import MotionPrior, fit_component
from uav.template_mask.gaps import trajectory_bridge
from uav.template_mask.policy import WarningLog, SkipVideo, uncertain_mask
from uav.template_mask.records import Observation, Candidate, Instance, Pose
from uav.template_mask.templates import TemplateBank


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.cfg = TemplateConfig(sift_backend='cpu')
        self.bank = TemplateBank(self.cache, self.cfg)

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def observation(self, oid, source, p, origin=(0, 0), pose=None, tid=None, valid=None):
        pk, fk = 'p'+oid, 'f'+oid
        self.cache.put(pk, probability=p, valid=np.ones(p.shape, bool) if valid is None else valid)
        self.cache.put(fk, xy=np.empty((0, 2)), desc=np.empty((0, 128)), size=np.empty(0))
        box = (*origin, origin[0]+p.shape[1], origin[1]+p.shape[0])
        return Observation(oid, source, source, oid, box, origin, pk, fk, .9, 336,
                           template_id=tid, pose=pose, reliable=True)

    def test_complementary_support_unknown_holes_and_source_dedup(self):
        a = np.zeros((8, 12), np.float32); a[1:4, 1:4] = .7; a[5, 10] = .9
        template = self.bank.create(self.observation('a', 1, a))
        self.bank.add(self.observation('repeat', 1, a, tid=template.id, pose=np.eye(3)))
        self.assertEqual(int(self.bank.fuse(template.id)['counts'].max()), 1)
        b = np.zeros_like(a); b[1:4, 1:4] = .6; b[1:4, 7:10] = .8
        self.bank.add(self.observation('b', 2, b, tid=0, pose=np.eye(3)))
        c = np.zeros_like(a); c[1:4, 7:10] = .5
        self.bank.add(self.observation('c', 3, c, tid=0, pose=np.eye(3), valid=c > 0))
        fused = self.bank.fuse(0)
        # Template bounds follow actual support, independently of storage padding.
        from uav.template_mask.geometry import warp
        stable = warp(fused['mean'], translation(*fused['origin']), a.shape)
        self.assertTrue(np.all(stable[1:4, 1:4] > .23))
        self.assertTrue(np.all(stable[1:4, 7:10] > .23))
        self.assertAlmostEqual(float(stable[5, 10]), .45, places=6)  # third view is unknown here
        self.assertEqual(stable[2, 5], 0)   # internal hole is preserved

    def test_expansion_does_not_move_anchor_or_poses(self):
        p = np.ones((5, 5), np.float32)
        seed = self.observation('a', 1, p, (40, 30))
        template = self.bank.create(seed)
        pose, anchor = seed.pose.copy(), template.anchor
        self.bank.add(self.observation('b', 2, p, (36, 30), tid=0, pose=pose))
        self.assertEqual(template.anchor, anchor)
        np.testing.assert_equal(seed.pose, pose)
        self.assertLess(template.box[0], 0)

    def test_blank_probability_storage_and_proposal_jitter_do_not_expand_template(self):
        seed = np.zeros((40, 80), np.float32)
        seed[12:22, 20:50] = .8
        original = self.observation('seed', 1, seed, (80, 50))
        template = self.bank.create(original)
        anchor, pose, box = template.anchor, original.pose.copy(), template.box
        for i, padding in enumerate((40, 120, 300), 2):
            p = np.full((40+2*padding, 80+2*padding), .12, np.float32)
            p[padding+12:padding+22, padding+20:padding+50] = .8
            # A high-probability invalid pixel also supplies no geometry.
            p[0, 0] = 1.
            valid = np.ones(p.shape, bool); valid[0, 0] = False
            obs = self.observation(str(i), i, p, (80-padding, 50-padding),
                                   tid=template.id, pose=pose.copy(), valid=valid)
            self.bank.add(obs)
            self.assertEqual(template.box, box)
            self.assertEqual(template.anchor, anchor)
            np.testing.assert_equal(obs.pose, pose)
        fused = self.bank.fuse(template.id)
        self.assertEqual(fused['mean'].shape, (14, 34))
        self.assertEqual(int((fused['mean'] > .23).sum()), 300)

    def test_bridge_requires_independent_sources_and_can_revoke(self):
        p = np.ones((4, 4), np.float32)
        self.bank.create(self.observation('a', 1, p))
        self.bank.create(self.observation('b', 2, p, (10, 0)))
        transform = translation(10, 0)
        self.bank.propose_link(0, 1, transform, 5, 'bridge')
        self.bank.propose_link(0, 1, transform, 5, 'bridge')
        self.assertEqual(set(self.bank.members(0)), {0})
        self.bank.propose_link(0, 1, transform, 6, 'bridge')
        self.assertEqual(set(self.bank.members(0)), {0, 1})
        self.bank.propose_link(0, 1, translation(30, 0), 7, 'bridge')
        self.assertEqual(set(self.bank.members(0)), {0})
        self.assertTrue(self.bank.links[(0, 1)].revoked)
        self.assertEqual(len(self.bank.observations), 2)

    def test_cache_retains_large_arrays_and_immutable_values(self):
        values = np.arange(4096, dtype=np.float32)
        self.cache.put('large', values=values)
        loaded = self.cache.get('large')['values']
        np.testing.assert_equal(loaded, values)
        self.assertGreaterEqual(self.cache.bytes, values.nbytes)
        with self.assertRaises(ValueError):
            loaded[0] = 8

    def test_roi_dilation_is_exact_at_frame_edges_and_inside_holes(self):
        from uav.template_mask.geometry import dilate
        for radius in (0, 1, 4, 12, 40):
            for x, y in ((0, 0), (70, 50), (145, 95)):
                mask = np.zeros((100, 150), np.uint8)
                mask[y:y+12, x:x+20] = 1
                mask[y+3:y+8, x+3:x+9] = 0
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
                expected = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)>0
                np.testing.assert_equal(dilate(mask > 0, radius), expected)

    def test_template_retains_probability_support_outside_detector_box(self):
        from uav.template_mask.sam import probability_bounds
        probability = np.zeros((40, 80), np.float32)
        probability[10:20, 12:25] = .8
        probability[10:20, 60:68] = .7
        crop = (100, 50, 180, 90)
        box = (110, 58, 130, 72)
        x, y, r, b = probability_bounds(probability, box, crop, .23)
        self.assertGreaterEqual(r, 168)
        patch = probability[y-50:b-50, x-100:r-100]
        obs = self.observation('outside', 1, patch, (x, y))
        obs.box = box
        template = self.bank.create(obs)
        data = self.bank.fuse(template.id)
        restored = points(translation(*data['origin']), np.argwhere(data['mean'] > .23)[:, ::-1])
        self.assertGreater(restored[:, 0].max(), box[2]-box[0])

    def test_sam_context_retains_background_for_small_and_thin_targets(self):
        from uav.template_mask.sam import context_crop
        for box in ((900, 500, 920, 510), (600, 500, 1200, 520),
                    (900, 200, 920, 800)):
            l, t, r, b = context_crop(box, (1080, 1920), self.cfg)
            width, height = r-l, b-t
            self.assertGreaterEqual(width, 336)
            self.assertGreaterEqual(height, 336)
            self.assertGreaterEqual(width, 3*(box[2]-box[0]))
            self.assertGreaterEqual(height, min(1080, 3*(box[3]-box[1])))
            self.assertEqual((width, height), (min(1920, max(336, 3*max(box[2]-box[0], box[3]-box[1]))),
                                               min(1080, max(336, 3*max(box[2]-box[0], box[3]-box[1])))))
            self.assertLessEqual(l, box[0]); self.assertLessEqual(t, box[1])
            self.assertGreaterEqual(r, box[2]); self.assertGreaterEqual(b, box[3])
            self.assertGreaterEqual(width*height, 9*(box[2]-box[0])*(box[3]-box[1]))

    def test_sam_context_shifts_inward_and_respects_frame_and_config(self):
        from uav.template_mask.sam import context_crop
        for box in ((0, 0, 20, 10), (1900, 1070, 1920, 1080)):
            l, t, r, b = context_crop(box, (1080, 1920), self.cfg)
            self.assertEqual((r-l, b-t), (336, 336))
            self.assertLessEqual(l, box[0]); self.assertLessEqual(t, box[1])
            self.assertGreaterEqual(r, box[2]); self.assertGreaterEqual(b, box[3])
        self.assertEqual(context_crop((0, 0, 10, 10), (100, 180), self.cfg), (0, 0, 180, 100))
        self.assertEqual(context_crop(None, (1080, 1920), self.cfg), (0, 0, 1920, 1080))
        cfg = replace(self.cfg, sam_context_scale=4., sam_context_min_side=400)
        l, t, r, b = context_crop((900, 500, 1000, 520), (1080, 1920), cfg)
        self.assertEqual((r-l, b-t), (400, 400))
        for values in (dict(sam_context_scale=1), dict(sam_context_scale=float('inf')),
                       dict(sam_context_min_side=0), dict(sam_context_min_side=336.)):
            with self.assertRaises(ValueError):
                replace(self.cfg, **values)


    def test_target_box_suppresses_neighbour_probabilities_without_clipping_strokes(self):
        from uav.template_mask.sam import target_probability
        p = np.ones((80, 180), np.float32)*.9
        original = p.copy()
        weighted = target_probability(p, (0, 0), (30, 30, 70, 50))
        np.testing.assert_equal(weighted[24:57, 24:77], p[24:57, 24:77])
        self.assertGreater(weighted[40, 73], .8)  # missing strokes just outside box
        self.assertLess(weighted[40, 115], .23)  # neighbouring watermark
        self.assertGreater(weighted[40, 80], weighted[40, 90])
        self.assertAlmostEqual(float(weighted[40, 20]), float(weighted[40, 80]))
        np.testing.assert_equal(p, original)  # cached raw probabilities immutable
        # Decay follows box dimensions, not the crop origin or absolute pixels.
        moved = target_probability(p, (100, 200), (130, 230, 170, 250))
        np.testing.assert_equal(moved, weighted)
        doubled = target_probability(np.ones((160, 360), np.float32)*.9, (0, 0), (60, 60, 140, 100))
        self.assertAlmostEqual(float(doubled[80, 200]), float(weighted[40, 100]))

    def test_partial_discovery_margin_preserves_other_line_but_suppresses_far_neighbour(self):
        from uav.template_mask.sam import target_probability
        p = np.full((220, 600), .9, np.float32)
        box = (140, 100, 425, 136)  # discovery saw only the lower line
        weighted = target_probability(p, (0, 0), box)
        np.testing.assert_equal(weighted[65:136, 140:425], p[65:136, 140:425])
        self.assertLess(float(weighted[10, 280]), .05)
        self.assertLess(float(weighted[118, 0]), .8)

    def test_reference_keypoints_are_inside_object_box_before_feature_budget(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        matcher = SiftMatcher(self.cache, self.cfg)
        def detect(gray, mask):
            self.assertEqual(gray.shape, (30, 40))
            self.assertIsNotNone(mask)
            # Fractional detector edges must not admit rounded exterior pixels.
            expected = np.zeros((30, 40), np.uint8)
            expected[1:30, 1:40] = 255
            np.testing.assert_equal(mask, expected)
            return [SimpleNamespace(pt=(4., 5.), size=2.)], np.ones((1, 128), np.float32)
        matcher.sift = SimpleNamespace(detectAndCompute=Mock(side_effect=detect))
        image = np.zeros((200, 300, 3), np.uint8)
        box = (80.25, 60.25, 120., 90.)
        result = matcher.extract(image, 1, box, feature_box=box)
        np.testing.assert_equal(self.cache.get(result)['xy'], [[84., 65.]])
        matcher.extract(image, 1, box, feature_box=box)
        self.assertEqual(matcher.sift.detectAndCompute.call_count, 1)

    def test_observation_features_use_object_box_not_sam_probability_canvas(self):
        from unittest.mock import Mock
        image = np.zeros((200, 300, 3), np.uint8)
        probability = np.full((180, 280), .5, np.float32)
        box = (80., 60., 120., 90.)
        detector = object()
        engine = TemplateEngine(self.cfg, self.cache, detector, None, image.shape[:2])
        original = engine.matcher.extract
        engine.matcher.extract = Mock(wraps=original)
        engine.observe(0, 5, image, Candidate(probability, (0, 0), box, .9, 'a'), 1008, 'local')
        engine.matcher.extract.assert_called_once_with(image, 5, box, feature_box=box)

    def test_rolling_eight_views_evict_worst_and_never_resurrect_it(self):
        p = np.zeros((10, 40), np.float32); p[3:7, 2:8] = .8
        bad = p.copy(); bad[3:7, 2:8] = .3; bad[3:7, 32:38] = .3
        template = self.bank.create(self.observation('bad', 0, bad))
        for source in range(1, 8):
            self.bank.add(self.observation(str(source), source, p, tid=0, pose=np.eye(3)))
        self.assertEqual(len(template.mask_views), 8)
        better = p.copy(); better[3:7, 2:8] = .95
        self.bank.add(self.observation('better', 8, better, tid=0, pose=np.eye(3)))
        fused = self.bank.fuse(0)
        self.assertEqual(len(fused['sources']), 8)
        self.assertNotIn(0, fused['sources'])
        self.assertIn(8, fused['sources'])
        from uav.template_mask.geometry import warp
        restored = warp(fused['mean'], translation(*fused['origin']), p.shape)
        self.assertEqual(restored[4, 35], 0.)
        worse = p*.4
        self.bank.add(self.observation('worse', 9, worse, tid=0, pose=np.eye(3)))
        self.assertNotIn(9, self.bank.fuse(0)['sources'])
        self.assertNotIn(0, self.bank.fuse(0)['sources'])
        self.assertEqual(len(template.mask_views), 8)

    def test_mean_suppresses_two_background_errors_and_respects_unknown(self):
        p = np.zeros((8, 12), np.float32); p[1:4, 1:4] = .8
        for source in range(8):
            probability = p.copy()
            if source < 2:
                probability[5, 10] = .8
            obs = self.observation(str(source), source, probability, tid=0, pose=np.eye(3))
            if source == 0:
                self.bank.create(obs)
            else:
                self.bank.add(obs)
        fused = self.bank.fuse(0)
        from uav.template_mask.geometry import warp
        restored = warp(fused['mean'], translation(*fused['origin']), p.shape)
        self.assertAlmostEqual(float(restored[5, 10]), .2, places=6)
        self.assertAlmostEqual(float(restored[2, 2]), .8, places=6)

    def test_global_views_never_supply_fusion_votes_and_local_336_does(self):
        p = np.ones((8, 12), np.float32)*.9
        seed = self.observation('global', 0, p)
        seed.mask_role = 'discovery'
        self.bank.create(seed)
        for source in (1, 2):
            obs = self.observation('global'+str(source), source, p, tid=0, pose=np.eye(3))
            obs.mask_role = 'discovery'; self.bank.add(obs)
        self.assertEqual(len(self.bank.fuse(0)['sources']), 0)
        self.assertFalse(self.bank.fuse(0)['mean'].any())
        self.assertTrue(self.bank.fuse(0, identity=True)['mean'].any())
        local = self.observation('local', 3, p*.8, tid=0, pose=np.eye(3))
        self.bank.add(local)
        self.assertEqual(self.bank.fuse(0)['sources'].tolist(), [3])
        self.assertAlmostEqual(float(self.bank.fuse(0)['mean'].max()), .72, places=6)

    def test_shared_template_mean_limits_references_and_revocation_restores_members(self):
        p = np.zeros((8, 12), np.float32); p[2:6, 2:6] = .8
        for tid, base in ((0, 0), (1, 100)):
            for i in range(8):
                obs = self.observation(str(base+i), base+i, p, tid=tid, pose=np.eye(3))
                if i == 0:
                    self.bank.create(obs)
                else:
                    self.bank.add(obs)
        self.bank.propose_link(0, 1, np.eye(3), 1000)
        self.bank.propose_link(0, 1, np.eye(3), 1001)
        self.assertEqual(len(self.bank.fuse(0)['sources']), 8)
        self.bank.revoke(0, 1, 'synthetic_layout_conflict')
        self.assertEqual(self.bank.fuse(0)['sources'].tolist(), list(range(8)))
        self.assertEqual(self.bank.fuse(1)['sources'].tolist(), list(range(100, 108)))

    def test_policy_and_warning_intervals(self):
        log = WarningLog(Path(self.temp.name)/'warnings.jsonl')
        instance = Instance(0, 0, 0)
        mask = np.zeros((100, 100), bool); mask[40:45, 40:45] = True
        expanded = uncertain_mask(mask, 1., instance, 2, self.cfg, log, 'gap')
        self.assertGreater(expanded.sum(), mask.sum())
        uncertain_mask(mask, 1., instance, 3, self.cfg, log, 'gap')
        self.assertEqual(log.rows[0]['last_frame'], 3)
        ignored = uncertain_mask(mask, 1., instance, 4, replace(self.cfg, uncertain_mask_policy='ignore'), log, 'gap')
        self.assertFalse(ignored.any())
        with self.assertRaises(SkipVideo):
            uncertain_mask(mask, 1., instance, 5, replace(self.cfg, uncertain_mask_policy='skip'), log, 'gap')
        with self.assertRaises(SkipVideo):
            uncertain_mask(mask, float('inf'), instance, 6, self.cfg, log, 'unlocalized')
        log.flush()
        self.assertEqual(len(log.path.read_text().splitlines()), 4)


class MotionTests(unittest.TestCase):
    def test_continuity_uses_independent_originals_and_rejects_search_polyline(self):
        poses = [Pose(f, translation(f, 10), .5, 'measured', f, presence=True) for f in range(24)]
        prior = MotionPrior(poses, (0, 0, 100, 25), (50, 12))
        self.assertIsNotNone(prior.continuity(24, translation(24, 10), .5, 24))
        self.assertIsNone(prior.continuity(24, translation(80, 10), .5, 24))
        exactly_eight = MotionPrior(poses[:8], (0, 0, 100, 25), (50, 12))
        self.assertIsNone(exactly_eight.continuity(7, poses[7].matrix, .5, 7))
        # Either leg is easy to fit, but inventing a corner in a gap is not
        # agreement between independently predicted continuations.
        left = poses[:16]
        right = [Pose(f, translation(15, f-5), .5, 'measured', f, presence=True) for f in range(24, 40)]
        from uav.template_mask.records import Template
        template = Template(0, (0, 0, 100, 25), (50, 12))
        self.assertIsNone(trajectory_bridge(left, right, template, range(15, 25)))

    def test_two_sided_static_linear_and_smooth_gaps(self):
        from uav.template_mask.records import Template
        template = Template(0, (0, 0, 100, 25), (50, 12))
        for motion in (lambda t:(20, 10), lambda t:(t, .2*t),
                       lambda t:(.0001*t**3-.01*t*t+t, .003*t*t)):
            rows = [Pose(f, translation(*motion(f)), .3, 'measured', f, presence=True)
                    for f in range(88) if not 40 <= f < 48]
            bridge = trajectory_bridge(rows[:40], rows[40:], template, range(39, 49))
            self.assertIsNotNone(bridge)
            for f, (matrix, sigma) in bridge[0].items():
                self.assertLessEqual(discrepancy(matrix, translation(*motion(f)), template.box), 3*sigma)
                self.assertGreater(sigma, 0)

    def test_smooth_curve_is_strong_pose_evidence(self):
        times = np.arange(80, dtype=float)
        values = 10+.01*times**2-.0001*times**3
        model = fit_component(times, values, .1)
        self.assertEqual(model.mode, 'smooth')
        poses = [Pose(int(f), translation(x, .1*f), .2, 'measured', int(f), presence=True)
                 for f, x in zip(times, values)]
        prior = MotionPrior(poses, (0, 0, 80, 25), (40, 12))
        matrix, sigma, strong = prior.predict(40)
        self.assertTrue(strong)
        self.assertLess(abs(matrix[0, 2]-values[40]), .1)
        self.assertGreater(prior.predict(200)[1], sigma)

    def test_endpoint_clusters_do_not_certify_smooth_gap(self):
        times = np.r_[np.arange(30), np.arange(200, 230)].astype(float)
        values = .01*times**2-.00002*times**3
        self.assertIsNone(fit_component(times, values, .1))

    def test_variable_speed_curves_and_polyline_turns_are_measured(self):
        times = np.arange(80, dtype=float)
        curve = .05*times**2+3*times
        self.assertEqual(fit_component(times, curve, .1).mode, 'accelerating')
        poses = []
        originals = []
        for f in range(60):
            xy = (f*f*.03, 15*np.sin(f/12)) if f < 30 else (27+(f-30)*2, 10-(f-30)*1.5)
            pose = translation(*xy)
            angle = .02*f+.15*np.sin(f/3)
            pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            originals.append(pose.copy())
            poses.append(Pose(f, pose, .5, 'measured', f, presence=True))
        prior = MotionPrior(poses, (0, 0, 80, 25), (40, 12))
        self.assertEqual(len(prior.evidence), 60)
        for p, expected in zip(poses, originals):
            np.testing.assert_equal(p.matrix, expected)

    def test_modes_actual_times_and_prediction_not_evidence(self):
        times = np.arange(80)*3
        self.assertEqual(fit_component(times, times*.2+10, .5).mode, 'linear')
        self.assertEqual(fit_component(times, np.ones(80)*8, .5).mode, 'static')
        values = 10*np.sin(2*np.pi*3*times/times[-1])
        model = fit_component(times, values, .6)
        self.assertIsNotNone(model)
        self.assertEqual(model.mode, 'periodic')
        self.assertGreaterEqual(model.cycles, 2.5)
        self.assertIsNone(fit_component(times, np.random.default_rng(4).normal(0, 20, 80), .2))
        poses = [Pose(int(t), translation(t, 10), .8, 'measured', int(t), presence=True) for t in times[:12]]
        prior = MotionPrior(poses, (0, 0, 100, 20), (50, 10))
        prediction = prior.predict(300)
        self.assertTrue(prediction[2])
        self.assertAlmostEqual(prediction[0][0, 2], 300)
        predictions = [Pose(t, translation(t, 10), .8, 'predicted', t, presence=True) for t in range(40, 500)]
        after = MotionPrior(poses+predictions, (0, 0, 100, 20), (50, 10))
        self.assertEqual(len(after.evidence), 12)


class SiftTests(unittest.TestCase):
    def test_multiple_locations_and_unrelated_descriptors(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EvidenceCache()
            try:
                matcher = SiftMatcher(cache, TemplateConfig(sift_backend='cpu'))
                rng = np.random.default_rng(5)
                xy = rng.uniform(0, 100, (32, 2)).astype(np.float32)
                desc = rng.uniform(0, 1, (32, 128)).astype(np.float32)
                reference = dict(xy=xy, desc=desc, size=np.ones(32)*3)
                target = dict(xy=np.concatenate((xy+[50, 30], xy+[250, 50])),
                              desc=np.concatenate((desc, desc)), size=np.ones(64)*3)
                fits = matcher.fit(reference, target, 2)
                self.assertEqual(len(fits), 2)
                shifts = sorted(tuple(np.rint(m.matrix[:2, 2]).astype(int)) for m in fits)
                self.assertEqual(shifts, [(50, 30), (250, 50)])
            finally:
                cache.close()


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.cfg = TemplateConfig(sift_backend='cpu', global_stride=4, mask_reference_limit=2, dilation=0)
        self.log = WarningLog(Path(self.temp.name)/'warnings.jsonl')

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    @staticmethod
    def mark():
        image = np.zeros((64, 180, 3), np.uint8)
        cv2.putText(image, 'UAV-2026', (4, 42), cv2.FONT_HERSHEY_SIMPLEX, .85, (230, 230, 230), 2)
        return image

    def test_stage2_multi_view_then_remote_instance_shares_template(self):
        mark = self.mark()
        p = (mark.max(2) > 0).astype(np.float32)*.9
        class Detector:
            def detect(self, image, source_id, box, resolution, **kwargs):
                positions = [(20+source_id, 30)]
                if source_id >= 8:
                    positions.append((260, 100))
                return [Candidate(p, (x, y), (x, y, x+180, y+64), .9, str(i))
                        for i, (x, y) in enumerate(positions)
                        if box is None or box[0] < x+180 and box[2] > x and box[1] < y+64 and box[3] > y]
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        images = []
        for f in range(12):
            image = np.zeros((200, 500, 3), np.uint8)
            image[30:94, 20+f:200+f] = mark
            if f >= 8:
                image[100:164, 260:440] = mark
            images.append(image)
            engine.first_frame(f, f, image)
        self.assertEqual(len(engine.bank.templates), 1)
        self.assertEqual(len(engine.instances), 2)
        self.assertLessEqual(engine.stats['template_calls'], 2)
        identities = set(engine.instances)
        engine.second_pass(lambda f: images[f])
        self.assertEqual(set(engine.instances), identities)
        self.assertTrue(engine.render(10).any())
        self.assertLessEqual(engine.stats['feedback_rounds'], 1)

    def test_final_mean_does_not_add_current_raw_noise_or_global_pixels_back(self):
        class Detector:
            def detect(self, *args): return []
        cfg = replace(self.cfg, mask_reference_limit=8)
        engine = TemplateEngine(cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.zeros((20, 60), np.float32); p[4:16, 4:16] = .8
        instance = None
        for frame in range(8):
            probability = p.copy()
            if frame in (0, 7):
                probability[4:16, 40:52] = .8
            obs = engine.observe(frame, frame, image, Candidate(probability, (30, 40),
                (30, 40, 90, 60), .9, str(frame)), 336, 'mean-output')
            if instance is None:
                instance = engine.associate(obs)
            else:
                obs.pose, obs.template_id, obs.reliable = translation(30, 40), 0, True
                engine._assign(instance, obs)
        coarse = engine.observe(7, 7, image, Candidate(np.ones((12, 120), np.float32),
            (30, 40), (30, 40, 150, 52), .9, 'global'), 1008, 'global')
        coarse.pose, coarse.template_id, coarse.reliable = translation(30, 40), 0, True
        engine._assign(instance, coarse)
        self.assertTrue(engine.render(7, final=False)[48, 75])
        final = engine.render(7)
        self.assertTrue(final[48, 40])
        self.assertFalse(final[48, 75])
        self.assertFalse(final[48, 140])
        self.assertEqual(len(engine.bank.fuse(0)['sources']), 8)

    def test_only_global_pixels_use_explicit_coarse_fallback(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        obs = engine.observe(0, 0, image, Candidate(np.ones((12, 24), np.float32),
            (30, 40), (30, 40, 54, 52), .9, 'a'), 1008, 'global')
        engine.associate(obs)
        self.assertTrue(engine.render(0, final=False).any())
        self.assertTrue(engine.render(0).any())
        self.assertTrue(engine.bank.output_fusion(0)[1])

    def test_usable_refinement_replaces_coarse_336_pixels(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.zeros((20, 60), np.float32); p[4:16, 4:16] = .8
        coarse = p.copy(); coarse[4:16, 40:52] = .9
        first = engine.observe(0, 0, image, Candidate(coarse, (30, 40),
            (30, 40, 90, 60), .9, 'coarse'), 336, 'coarse')
        instance = engine.associate(first)
        refined = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 90, 60), .9, 'refined'), 336, ('refinement', 0, first.id))
        refined.pose, refined.template_id, refined.reliable = first.pose.copy(), 0, True
        engine._assign(instance, refined)
        self.assertEqual(engine.bank.fuse(0)['sources'].tolist(), [0])
        self.assertTrue(bool(engine.bank.fuse(0)['refined']))
        self.assertFalse(engine.render(0)[48, 75])
        self.assertTrue(engine.render(0)[48, 40])

    def test_crossing_marks_observations_unknown_without_destroying_ids(self):
        class Detector:
            def detect(self, *args):
                return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        obs = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'a'), 1008, 'global')
        template = engine.bank.create(obs)
        first = engine._new_instance(template, obs)
        second = Instance(1, 0, 0, poses={0: Pose(0, obs.pose.copy(), 1., 'measured', 0, presence=True)})
        engine.instances[1] = second
        engine._crossings(0)
        self.assertEqual(first.poses[0].status, 'uncertain')
        self.assertEqual(second.poses[0].status, 'uncertain')
        self.assertTrue(obs.ambiguous)
        self.assertFalse((engine.bank.fuse(0)['mean'] > .23).any())

    def test_first_frame_whole_and_fragment_preserve_original_references(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        for label, p in (('whole', np.ones((20, 60), np.float32)*.8),
                         ('fragment', np.ones((12, 24), np.float32)*.8)):
            obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
                (30, 40, 30+p.shape[1], 40+p.shape[0]), .9, label), 336, 'whole-fragment')
            engine.pending.defer(obs, 'new_template')
        engine._crossings(0)
        self.assertTrue(all(i.poses[0].status == 'measured' for i in engine.instances.values()))
        self.assertTrue(all(not o.ambiguous for o in engine.bank.observations.values()))
        self.assertEqual(engine.stats['ambiguous_frames'], 0)
        self.assertFalse(engine.bank.links)  # Pixel containment alone never merges identities.

    def test_disjoint_strokes_inside_overlapping_boxes_are_not_crossing(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.zeros((20, 60), np.float32); p[5:15, 2:12] = .8
        obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 90, 60), .9, 'a'), 336, 'disjoint')
        template = engine.bank.create(obs); first = engine._new_instance(template, obs)
        second = Instance(1, template.id, 0,
            poses={0:Pose(0, translation(50, 40), 1., 'measured', 0, presence=True)})
        engine.instances[1] = second
        engine._crossings(0)
        self.assertEqual(first.poses[0].status, 'measured')
        self.assertEqual(second.poses[0].status, 'measured')
        self.assertFalse(obs.ambiguous)

    def test_distinct_templates_with_separate_history_still_mark_crossing(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        owners = []
        for label, x in (('a', 10), ('b', 100)):
            p = np.ones((12, 24), np.float32)*.8
            obs = engine.observe(0, 0, image, Candidate(p, (x, 40),
                (x, 40, x+24, 52), .9, label), 1008, 'global')
            owner = engine.pending.defer(obs, 'local_identity_unconfirmed')
            owner.poses.update({f:Pose(f, translation(x, 40), 1., 'measured', f, presence=True)
                                for f in (0, 1)})
            owner.poses[2] = Pose(2, translation(50, 40), 1., 'measured', 2, presence=True)
            owners.append(owner)
        from unittest.mock import patch
        with patch('uav.template_mask.engine.relative_motion', return_value={'state':'independent_motion'}):
            engine._crossings(2)
        self.assertTrue(all(i.poses[2].status == 'measured' for i in owners))
        self.assertEqual(engine.stats['ambiguous_frames'], 1)

    def test_production_config_uses_shared_template_engine(self):
        from uav import UAVConfig
        from uav.cli import build_parser
        from uav.pipeline import mask_config
        config = UAVConfig()
        self.assertFalse(hasattr(config, 'two_pass'))
        self.assertIn('uncertain-mask-policy', build_parser().format_help())
        self.assertFalse(hasattr(mask_config(config), 'mask_schedule'))
        self.assertEqual(mask_config(config).sift_backend, 'cuda')


    def test_repeated_source_frame_has_one_evidence_vote(self):
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        class Detector:
            def detect(self, *args):
                return [Candidate(p, (20, 30), (20, 30, 200, 94), .9, '0')]
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        engine.first_frame(0, 0, image)
        engine.first_frame(1, 0, image)
        self.assertEqual(int(engine.bank.fuse(0, coarse=True)['counts'].max()), 1)
        self.assertEqual(engine.stats['template_calls'], 0)
        np.testing.assert_equal(engine.render(0), engine.render(1))

    def test_established_investigation_hypothesis_is_not_a_second_crossing_object(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        owners = []
        for tid in range(2):
            obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
                (30, 40, 54, 52), .9, str(tid)), 336, ('alternative', tid))
            if tid:
                # Different probability output, not an exact duplicate task.
                self.cache.put(obs.probability_key, probability=p*.9, valid=p > 0)
            owner = engine.pending.defer(obs, 'new_template' if not tid else 'local_identity_unconfirmed')
            owners.append(owner)
            engine.bank.templates[owner.template_id].provisional = False
        engine._crossings(0)
        self.assertTrue(all(i.poses[0].status == 'measured' for i in owners))

    def test_global_detection_stops_old_flash_without_spending_identity_budget(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 300))
        image = np.zeros((100, 300, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'old'), 336, 'flash')
        old = engine.pending.defer(obs, 'new_template')
        old.poses.update({f:Pose(f, translation(30, 40), .4+f,
            'uncertain', reason='no_original_image_match') for f in range(1, 13)})
        new_obs = engine.observe(8, 8, image, Candidate(p, (230, 40),
            (230, 40, 254, 52), .9, 'new'), 336, 'flash')
        new_obs.pose, new_obs.template_id, new_obs.reliable = translation(230, 40), old.template_id, True
        new = engine._new_instance(engine.bank.templates[old.template_id], new_obs)
        engine.source_ids = {f:f for f in range(13)}
        engine.presence.check(8, 8, [Candidate(p, (230, 40), (230, 40, 254, 52), .9, 'new')])
        engine.presence.resolve(lambda f:image)
        self.assertEqual(old.poses[8].status, 'absent')
        self.assertEqual(old.poses[12].status, 'absent')
        self.assertEqual(new.poses[8].status, 'measured')
        self.assertEqual(new.template_id, old.template_id)
        self.assertNotEqual(new.id, old.id)
        mask = engine.render(12)
        self.assertFalse(mask.any())

    def test_global_miss_keeps_image_match_and_two_sided_gap_repair(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'a'), 336, 'global-miss')
        owner = engine.pending.defer(obs, 'new_template')
        owner.poses = {f:Pose(f, translation(30, 40), .3, 'measured', f, presence=True)
                       for f in range(40)}
        engine.source_ids = {f:f for f in range(40)}
        engine.presence.check(8, 8, [])
        self.assertEqual(owner.poses[8].status, 'measured')
        for f in range(16, 20):
            owner.poses[f] = Pose(f, translation(30, 40), .3, 'uncertain')
        engine.presence.check(16, 16, [])
        engine.presence.resolve(lambda f:image)
        # Reverse search finds measured frame 15 and extends coverage to 16.
        self.assertNotEqual(owner.poses[16].status, 'absent')
        self.assertTrue(engine.presence.covers(owner, 16))
        engine.gaps.resolve(lambda f:image)
        self.assertEqual(owner.poses[16].status, 'predicted')
        self.assertFalse(owner.poses[16].presence)
        self.assertTrue(engine.render(16).any())

    def test_repeated_global_misses_pause_tracking_until_later_observation(self):
        from unittest.mock import patch
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'a'), 336, 'pause')
        owner = engine.pending.defer(obs, 'new_template')
        owner.poses[4] = Pose(4, obs.pose.copy(), .7, 'uncertain')
        engine.presence.check(4, 4, [])
        with patch.object(engine.matcher, 'extract', side_effect=AssertionError('unexpected SIFT')):
            for f in range(5, 16):
                engine._track(owner, f, f, image)
                if f % self.cfg.global_stride == 0:
                    engine.presence.check(f, f, [])
        self.assertTrue(all(owner.poses[f].status == 'absent' for f in range(4, 16)))
        self.assertFalse(engine.render(15).any())
        # A later original observation can resume a path; no earlier prediction
        # becomes evidence or extends this instance into a remote occurrence.
        later = engine.observe(16, 16, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'b'), 336, 'pause')
        later.pose, later.template_id, later.reliable = obs.pose.copy(), owner.template_id, True
        engine._assign(owner, later)
        engine.presence.check(16, 16, [Candidate(p, (30, 40), (30, 40, 54, 52), .9, 'b')])
        self.assertFalse(engine.presence.missing(owner, 17))
        self.assertEqual(len(owner.motion.evidence), 2)

    def test_global_sample_before_discovery_delimits_flash_appearance(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        obs = engine.observe(8, 8, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'a'), 336, 'appearance')
        owner = engine.pending.defer(obs, 'new_template')
        owner.poses.update({f:Pose(f, obs.pose.copy(), .7, 'uncertain') for f in range(4, 8)})
        owner.poses[6] = Pose(6, obs.pose.copy(), .7, 'measured', 6, presence=True)
        owner.poses[7] = Pose(7, obs.pose.copy(), .7, 'measured', 7, presence=True)
        engine.source_ids = {f:f for f in range(9)}
        engine.presence.check(4, 4, [])
        engine.presence.check(8, 8, [Candidate(p, (30, 40), (30, 40, 54, 52), .9, 'a')])
        engine.presence.resolve(lambda f:image)
        self.assertEqual(owner.birth, 5)
        self.assertFalse(engine.render(4).any())
        self.assertTrue(engine.render(5).any())
        self.assertTrue(engine.render(6).any())







    def test_partial_attaches_but_unexplained_extra_strokes_keep_a_separate_hypothesis(self):
        from unittest.mock import patch
        class Detector:
            def detect(self, *args): raise AssertionError('association must not run SAM')
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        seed = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'seed'), 1008, 'global')
        owner = engine.pending.defer(seed, 'new_template')
        owner.poses[1] = Pose(1, seed.pose.copy(), .3, 'measured', 1, (seed.id,), True)
        for label, probability, box in (('partial', p[:, :8], (30, 40, 38, 52)),):
            obs = engine.observe(1, 1, image, Candidate(probability, (30, 40), box, .9, label),
                                 1008, 'global')
            measured_pose = owner.poses[1]
            result = engine.associate(obs)
            self.assertEqual(result.id, owner.id)
            self.assertEqual(obs.association_reason, 'partial_or_complete_at_measured_pose')
            self.assertIs(owner.poses[1], measured_pose)
        engine.source_ids = {0:0, 1:1}
        engine.presence.appearance_checked.add(owner.id)
        engine.presence.global_samples[0] = (0, [seed.box])
        engine.presence.global_samples[1] = (1, [seed.box])
        engine.presence.checks[owner.id] = {0:'image_match', 1:'image_match'}
        with patch.object(engine.presence, '_probe', side_effect=AssertionError('no lifetime recheck')):
            engine.presence.online(1, lambda f:image, set())
        self.assertEqual(len(engine.instances), 1)
        self.assertEqual(engine.presence.events, [])
        box=(30,40,78,52)
        whole=engine.observe(1,1,image,Candidate(np.ones((12,48),np.float32)*.8,
            (30,40),box,.9,'whole'),336,'association')
        references=list(engine.bank.templates[owner.template_id].references)
        result=engine.associate(whole)
        self.assertNotEqual(result.id,owner.id)
        self.assertNotEqual(result.template_id,owner.template_id)
        self.assertEqual(engine.bank.templates[owner.template_id].references,references)
        self.assertNotIn(whole.id,engine.bank.templates[owner.template_id].observations)


    def test_shared_text_is_not_identity_but_partial_strokes_are_compatible(self):
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args):
                return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        rng = np.random.default_rng(72)
        image = rng.integers(0, 255, (200, 500, 3), dtype=np.uint8)
        p = np.zeros((40, 160), np.float32)
        p[5:30, 5:60:5] = 1
        p[5:30, 90:150:5] = 1
        original = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 180, 70), .9, '0'), 1008, 'global')
        template = engine.bank.create(original)
        other = p.copy(); other[:, 90:] = 0; other[5:30:4, 90:150] = 1
        obs = engine.observe(1, 1, image, Candidate(other, (20, 30), (20, 30, 180, 70), .9, '0'), 1008, 'global')
        fit = Match(translation(20, 30), .5, 20, .6)
        self.assertFalse(engine._shape_agreement(template, fit, obs))
        fragment = engine.observe(2, 2, image, Candidate(p[:, :65], (20, 30), (20, 30, 85, 70), .9, '0'), 1008, 'global')
        self.assertTrue(engine._shape_agreement(template, fit, fragment))

    def test_unassigned_pixels_obey_skip_and_ignore(self):
        class Detector:
            def detect(self, *args):
                return []
        engine = TemplateEngine(replace(self.cfg, uncertain_mask_policy='ignore'),
                                self.cache, Detector(), self.log, (100, 100))
        image = np.zeros((100, 100, 3), np.uint8)
        candidate = Candidate(np.ones((10, 10), np.float32), (40, 40), (40, 40, 50, 50), .8, '0')
        engine.observe(0, 0, image, candidate, 336, 'unassigned')
        self.assertFalse(engine.render(0).any())
        self.assertEqual(self.log.rows[-1]['action'], 'omit_uncertain')
        engine.config = replace(self.cfg, uncertain_mask_policy='skip')
        with self.assertRaises(SkipVideo):
            engine.render(0)

    def test_mixed_provisional_template_obeys_uncertainty_policy(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        obs = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'mixed'), 336, 'mixed')
        instance = engine.pending.defer(obs, 'competing_templates')
        self.assertTrue(engine.bank.templates[instance.template_id].mixed)
        self.assertTrue(engine.render(0).any())
        self.assertGreater(engine.render_metrics[0]['conservative_pixels'], 0)
        engine.config = replace(self.cfg, uncertain_mask_policy='ignore')
        self.assertFalse(engine.render(0).any())
        engine.config = replace(self.cfg, uncertain_mask_policy='skip')
        with self.assertRaises(SkipVideo):
            engine.render(0)
        self.assertTrue(any(w['reason'] == 'mixed_hypothesis_identity_unresolved' for w in self.log.rows))

    def test_mixed_hypothesis_is_actively_confirmed_without_promoting_contested_seed(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        seed = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'mixed'), 1008, 'global')
        instance = engine.pending.defer(seed, 'competing_templates')
        fit = Match(translation(30, 40), .3, 20, .9, (seed.id,))
        for frame in (1, 2):
            obs = engine.observe(frame, frame, image, Candidate(p, (30, 40),
                (30, 40, 54, 52), .9, str(frame)), 1008, 'global')
            with patch.object(engine, '_matches', return_value=[fit]):
                owner = engine.associate(obs)
            self.assertEqual(owner.id, instance.id)
            self.assertEqual(len(engine.bank.templates), 1)
        template = engine.bank.templates[0]
        self.assertFalse(template.mixed)
        self.assertFalse(template.provisional)
        self.assertTrue(seed.ambiguous)
        self.assertFalse(instance.poses[0].presence)
        self.assertEqual(instance.poses[0].status, 'uncertain')
        self.assertTrue((engine.bank.fuse(0, coarse=True)['mean'] > .23).any())
        self.assertEqual(len(template.identity_confirmations), 2)

    def test_complete_rechecks_reach_mixed_whole_before_existing_fragment(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (160, 240))
        image = np.zeros((160, 240, 3), np.uint8)
        # Probability support can extend well outside SAM's box head.
        p = np.ones((50, 90), np.float32)*.8
        fragment = engine.observe(0, 0, image, Candidate(p[:12, :24], (30, 40),
            (30, 40, 54, 52), .9, 'fragment'), 1008, 'global')
        fragment_owner = engine.pending.defer(fragment, 'new_template')
        seed = engine.observe(0, 0, image, Candidate(p, (30, 40),
            (30, 40, 90, 60), .9, 'whole'), 1008, 'global')
        whole = engine.pending.defer(seed, 'competing_templates')
        fit = Match(translation(30, 40), .3, 30, .9, (seed.id,))
        for frame, source in ((1, 1), (2, 1), (3, 2)):
            obs = engine.observe(frame, source, image, Candidate(p, (30, 40),
                (30, 40, 90, 60), .9, str(frame)), 1008, 'global')
            with patch.object(engine, '_matches', return_value=[fit]):
                owner = engine.associate(obs)
            self.assertEqual(owner.id, whole.id)
            if source == 1:
                self.assertTrue(engine.bank.templates[whole.template_id].mixed)
        template = engine.bank.templates[whole.template_id]
        self.assertFalse(template.mixed)
        self.assertEqual({engine.bank.observations[o].source_id for o in template.identity_confirmations}, {1, 2})
        self.assertEqual(len(engine.bank.templates), 2)
        self.assertEqual(fragment_owner.observations, [fragment.id])
        self.assertTrue(seed.ambiguous)

    def test_strong_motion_does_not_prove_presence(self):
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        class Detector:
            def detect(self, *args):
                return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        obs = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, '0'), 336, 'test')
        template = engine.bank.create(obs)
        instance = engine._new_instance(template, obs)
        instance.poses = {f:Pose(f, translation(20, 30), .5, 'measured', f, presence=True) for f in range(16)}
        engine._motion(instance)
        self.assertTrue(instance.motion.predict(40)[2])
        engine._track(instance, 40, 40, np.zeros_like(image))
        self.assertEqual(instance.poses[40].status, 'uncertain')
        self.assertFalse(instance.poses[40].presence)

    def test_local_complementary_strokes_survive_later_partial_view(self):
        class Detector:
            def detect(self, *args):
                return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 150))
        image = np.zeros((100, 150, 3), np.uint8)
        full = np.zeros((20, 60), np.float32)
        full[4:8, 3:9] = .9
        full[4:8, 48:54] = .9
        obs = engine.observe(0, 0, image, Candidate(full, (20, 30), (20, 30, 80, 50), .9, '0'), 1008, 'full')
        template = engine.bank.create(obs)
        instance = engine._new_instance(template, obs)
        partial = full.copy(); partial[:, 30:] = 0
        later = engine.observe(1, 1, image, Candidate(partial, (20, 30), (20, 30, 80, 50), .9, '0'), 336, 'partial')
        later.pose, later.template_id, later.reliable = obs.pose.copy(), template.id, True
        engine._assign(instance, later)
        local = engine.render(1)
        self.assertTrue(local[34:38, 68:74].all())
        other = Instance(1, template.id, 1)
        self.assertNotIn(other.id, engine.instances)

    def test_final_output_is_not_rejected_by_area_and_keeps_dilation(self):
        class Detector:
            def detect(self, *args):
                return []
        cfg = replace(self.cfg, dilation=4)
        engine = TemplateEngine(cfg, self.cache, Detector(), self.log, (100, 150))
        image = np.zeros((100, 150, 3), np.uint8)
        p = np.zeros((20, 20), np.float32); p[8:12, 8:12] = 1.
        obs = engine.observe(0, 0, image, Candidate(p, (40, 40), (40, 40, 60, 60), .9, '0'), 336, 'test')
        template = engine.bank.create(obs)
        engine._new_instance(template, obs)
        raw = engine.render(0, final=False)
        final = engine.render(0)
        diagnostic = engine.render(0, diagnostic=True)
        np.testing.assert_array_equal(final, diagnostic)
        self.assertTrue(np.all(diagnostic[raw]))
        self.assertGreater(diagnostic.sum(), raw.sum())
        self.assertEqual(engine.render_metrics[0]['dilation_radii'], [4])

    def test_pending_retries_after_future_reference_without_repeating_sam(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args):
                raise AssertionError('retained evidence must not call SAM')
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        seed = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'a'), 1008, 'global')
        template = engine.bank.create(seed); instance = engine._new_instance(template, seed)
        obs = engine.observe(1, 1, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'b'), 1008, 'global')
        with patch.object(engine, '_matches', return_value=[]):
            hypothesis = engine.pending.defer(obs, 'new_template')
        self.assertIsNotNone(hypothesis)
        self.assertIn(obs.id, engine.pending.membership)
        self.assertFalse(obs.ambiguous)
        self.assertTrue(engine.bank.templates[hypothesis.template_id].provisional)
        # A second independent original registers the hypothesis and validates
        # the coordinate relation to the existing template.
        other = engine.observe(2, 2, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'c'), 1008, 'global')
        other.pose, other.template_id, other.reliable = obs.pose.copy(), hypothesis.template_id, True
        engine._assign(hypothesis, other)
        fit = Match(seed.pose.copy(), .5, 20, .8, (seed.id,))
        with patch.object(engine, '_matches', return_value=[fit]):
            engine.pending.retry()
        self.assertIn(hypothesis.template_id, engine.bank.members(instance.template_id))
        # Original membership is immutable; the link supplies shared identity.
        self.assertEqual(obs.instance_id, hypothesis.id)
        self.assertNotIn(obs.id, engine.pending.membership)
        self.assertFalse(obs.ambiguous)
        engine.render(1)
        self.assertFalse(any(w['reason'] == 'unassigned_candidate' for w in self.log.rows))
        self.assertEqual(engine.render_metrics[1]['conservative_pixels'], 0)


    def test_unexplained_shape_exists_immediately_and_grows_progressively(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        observations = []
        for f in range(3):
            image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20+f:200+f] = mark
            obs = engine.observe(f, f, image, Candidate(p, (20+f, 30),
                (20+f, 30, 200+f, 94), .9, 'a'), 1008, 'global')
            engine.associate(obs, allow_new=False); observations.append(obs)
            if f == 0:
                self.assertEqual(len(engine.bank.templates), 1)
                self.assertTrue(engine.bank.templates[0].provisional)
                self.assertTrue(engine.render(0).any())
        engine.pending.retry()
        self.assertEqual(len(engine.bank.templates), 1)
        self.assertFalse(engine.bank.templates[0].provisional)
        self.assertTrue(all(o.instance_id == 0 and not o.ambiguous for o in observations))
        self.assertEqual(len(engine.bank.templates[0].references), 3)

    def test_two_sided_gap_predictions_do_not_become_measured_evidence(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        obs = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'a'), 336, 'seed')
        template = engine.bank.create(obs); instance = engine._new_instance(template, obs)
        instance.poses = {f:Pose(f, translation(20, 30), .5, 'measured', f, presence=True)
                          for f in range(40) if not 16 <= f < 20}
        engine.source_ids = {f:f for f in range(40)}
        before = len(instance.poses)
        engine.gaps.resolve(lambda f: np.zeros_like(image))
        for f in range(16, 20):
            self.assertEqual(instance.poses[f].status, 'predicted')
            self.assertFalse(instance.poses[f].presence)
            self.assertTrue(engine.render(f).any())
            from uav.template_mask.review import uncertainty_banner
            messages = uncertainty_banner(engine.render_metrics[f])
            self.assertIn('CONSERVATIVE COVER', messages[0])
            self.assertIn('two_sided_trajectory_presence_unconfirmed', messages)
            self.assertGreater(engine.render_metrics[f]['conservative_pixels'], 0)
        engine._motion(instance)
        self.assertEqual(len(instance.motion.evidence), before)
        self.assertEqual(engine.stats['trajectory_gap_predictions'], 4)

    def test_trajectory_support_disambiguates_without_current_source_vote(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        for number in range(2):
            seed = engine.observe(0, number, image, Candidate(p, (20, 30),
                (20, 30, 200, 94), .9, str(number)), 1008, 'global')
            template = engine.bank.create(seed); instance = engine._new_instance(template, seed)
            instance.poses = {f:Pose(f, translation(20+f, 30), .3, 'measured', f, presence=True)
                              for f in range(24 if number == 0 else 3)}
            engine._motion(instance)
        obs = engine.observe(24, 24, image, Candidate(p, (44, 30),
            (44, 30, 224, 94), .9, 'new'), 1008, 'global')
        fit = Match(translation(44, 30), .5, 20, .8)
        with patch.object(engine, '_matches', return_value=[fit]), patch.object(engine, '_shape_agreement', return_value=True):
            owner = engine.associate(obs)
        self.assertEqual(owner.id, 0)
        self.assertEqual(obs.association_reason, 'validated_trajectory_continuity')
        self.assertEqual(obs.trajectory_evidence['measured_sources'], 24)
        self.assertEqual(engine.stats['trajectory_associations'], 1)

    def test_explicit_absence_blocks_bridge_and_absent_object_is_not_crossing(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        obs = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'a'), 336, 'seed')
        template = engine.bank.create(obs); instance = engine._new_instance(template, obs)
        instance.poses = {f:Pose(f, translation(20, 30), .5, 'measured', f, presence=True)
                          for f in range(40) if not 16 <= f < 20}
        instance.poses[17] = Pose(17, translation(20, 30), .5, 'absent', reason='verified_disappearance')
        engine.source_ids = {f:f for f in range(40)}
        engine.gaps.resolve(lambda f: self.fail('explicit absence must block recheck'))
        self.assertEqual(engine.stats['trajectory_gap_predictions'], 0)
        other = Instance(1, 0, 0, poses={0:Pose(0, obs.pose.copy(), 1., 'absent')})
        engine.instances[1] = other
        engine._crossings(0)
        self.assertEqual(instance.poses[0].status, 'measured')

    def test_review_rates_and_publication_preserve_other_runs(self):
        from uav.template_mask.benchmark import processing_rates, publish
        report = dict(complete=True, skipped=False, frames=24, rendered_frames=24,
            timing=dict(first_pass_seconds=6., second_pass_seconds=2.), total_seconds=12.,
            config=TemplateConfig(sift_backend='cpu').as_dict())
        self.assertEqual(processing_rates(report), dict(algorithm_fps=3., overall_fps=2.))
        self.assertIsNone(processing_rates(dict(report, skipped=True)))
        self.assertIsNone(processing_rates(dict(report, rendered_frames=12)))
        root = Path(self.temp.name)/'review'; root.mkdir()
        (root/'index.html').write_text('<!doctype html><style></style><section>Existing</section>')
        rows = [dict(index=4, name='synthetic.mp4', start=0, status='complete', report=report)]
        for folder, filename in (('run-a/video-04', 'mask-review.mp4'),
                                 ('latest21-20261022/video-04', 'mask-passes.mp4')):
            path = root/folder/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # Existence-only fixture; never read as media.
        publish(root, 'run-a', rows); publish(root, 'run-b', rows); publish(root, 'run-a', rows)
        page = (root/'index.html').read_text()
        self.assertEqual(page.count('id="template-mask-run-a"'), 1)
        self.assertIn('template-mask-run-b', page)
        self.assertIn('<section>Existing</section>', page)
        self.assertIn('Stage 2 对共享模板', page)
        self.assertIn('本批次新实现：请审核右侧的最终 mask', page)
        self.assertIn('右半边为旧实现最终 mask（不是本次新结果）', page)
        self.assertIn('比较新实现的右侧和旧实现的右半边', page)

    def test_two_segments_keep_one_canonical_gap_coverage(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        obs = engine.observe(0, 0, image, Candidate(p, (20, 30), (20, 30, 200, 94), .9, 'a'), 336, 'seed')
        template = engine.bank.create(obs); first = engine._new_instance(template, obs)
        first.poses = {f:Pose(f, translation(20, 30), .5, 'measured', f, presence=True) for f in range(16)}
        first.ended_at = 15
        second = Instance(1, template.id, 20, poses={f:Pose(f, translation(20, 30), .5,
            'measured', f, presence=True) for f in range(20, 40)})
        for f in range(16, 20):
            second.poses[f] = Pose(f, translation(20, 30), 3., 'uncertain')
        engine.instances[1] = second
        engine.source_ids = {f:f for f in range(40)}
        engine.gaps.resolve(lambda f: np.zeros_like(image))
        self.assertEqual(second.continuation_of, first.id)
        self.assertEqual(first.render_end, 19)
        self.assertEqual(second.render_start, 20)
        self.assertTrue(all(first.poses[f].status == 'predicted' for f in range(16, 20)))
        for f in range(16, 20):
            engine.render(f)
            self.assertEqual(len(engine.render_metrics[f]['uncertainty']), 1)
            self.assertEqual(engine.render_metrics[f]['uncertainty'][0]['instance'], 0)
        self.assertEqual(second.poses[20].status, 'measured')



    def test_alias_revocation_preserves_original_poses_and_single_view_strokes(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.zeros((20, 60), np.float32); p[4:16, 4:16] = .8
        a = engine.observe(0, 0, image, Candidate(p, (30, 40), (30, 40, 90, 60), .9, 'a'), 336, 'alias')
        first = engine.pending.defer(a, 'new_template')
        more = p.copy(); more[4:16, 40:52] = .8
        b = engine.observe(1, 1, image, Candidate(more, (30, 40), (30, 40, 90, 60), .9, 'b'), 336, 'alias')
        second = engine.pending.defer(b, 'overlap_without_identity')
        for instance in (first, second):
            instance.poses = {f:Pose(f, translation(30, 40), .5, 'measured', f, presence=True)
                              for f in (0, 1)}
        second.poses[2] = Pose(2, translation(31, 40), .5, 'measured', 2, presence=True)
        for source in (0, 1):
            engine.bank.propose_link(0, 1, np.eye(3), source)
        engine.reconciler.refresh_aliases()
        self.assertEqual(second.alias_of, first.id)
        self.assertFalse(engine._renders_at(second, 1))
        self.assertTrue(engine._renders_at(second, 2))
        self.assertNotIn(2, first.poses)  # No copied measurements.
        self.assertTrue(engine.render(1)[48, 75])  # Alias's single-view stroke.
        engine.bank.revoke(0, 1, 'test_layout_conflict')
        engine.pending.retry()
        self.assertIsNone(second.alias_of)
        self.assertTrue(engine._renders_at(second, 1))
        self.assertTrue(engine.render(1)[48, 75])
        self.assertNotIn(2, first.poses)
        self.assertEqual(b.template_id, 1)
        self.assertFalse((engine.bank.fuse(0)['mean'][:, 40:52] > .23).any())

    def test_registered_alias_does_not_compete_with_canonical_trajectory(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        owners = []
        for tid in range(2):
            obs = engine.observe(tid, tid, image, Candidate(p, (30, 40),
                (30, 40, 54, 52), .9, str(tid)), 1008, 'global')
            owner = engine.pending.defer(obs, 'new_template')
            owner.poses = {f:Pose(f, translation(30, 40), .3, 'measured', f,
                (obs.id,), True) for f in range(24)}
            engine._motion(owner); owners.append(owner)
        for source in (0, 1):
            engine.bank.propose_link(0, 1, np.eye(3), source)
        engine.reconciler.refresh_aliases()
        self.assertEqual(owners[1].alias_of, 0)
        current = engine.observe(24, 24, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'current'), 1008, 'global')
        fits = Match(translation(30, 40), .3, 20, .9, (owners[0].observations[0],))
        with patch.object(engine, '_matches', return_value=[fits]):
            result = engine.associate(current)
        self.assertEqual(result.id, 0)
        self.assertEqual(len(engine.bank.templates), 2)
        self.assertNotEqual(current.association_reason, 'competing_validated_trajectories')

    def test_multiple_alias_targets_count_as_one_occurrence_and_revoke_cleanly(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 180))
        image = np.zeros((100, 180, 3), np.uint8)
        owners = []
        for tid in range(3):
            p = np.ones((12, 24), np.float32)*(.8-tid*.01)
            obs = engine.observe(tid, tid, image, Candidate(p, (30, 40),
                (30, 40, 54, 52), .9, str(tid)), 336, 'alias-chain')
            owner = engine.pending.defer(obs, 'new_template')
            owner.poses = {f:Pose(f, translation(30, 40), .3, 'measured', f, presence=True)
                           for f in range(12)}
            owners.append(owner)
        for a, b in ((0, 1), (1, 2)):
            for source in (10, 11):
                engine.bank.propose_link(a, b, np.eye(3), source)
        engine.reconciler.refresh_aliases()
        self.assertEqual([i.alias_of for i in owners], [None, 0, 0])
        self.assertEqual(owners[2].alias_templates, (0, 2))
        engine.bank.revoke(1, 2, 'independent_motion')
        engine.reconciler.refresh_aliases()
        self.assertEqual([i.alias_of for i in owners], [None, 0, None])
        self.assertTrue(all(len(i.poses) == 12 for i in owners))

    def test_linked_template_coordinates_continue_existing_instance(self):
        from unittest.mock import patch
        from uav.template_mask.features import Match
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (100, 300))
        image = np.zeros((100, 300, 3), np.uint8)
        p = np.ones((12, 24), np.float32)*.8
        owners = []
        for tid, x in enumerate((200, 30)):
            obs = engine.observe(0, 0, image, Candidate(p, (x, 40),
                (x, 40, x+24, 52), .9, str(tid)), 1008, 'global')
            owners.append(engine.pending.defer(obs, 'new_template'))
        relation = translation(3, -1)
        for source in (0, 1):
            engine.bank.propose_link(0, 1, relation, source)
        current = engine.observe(1, 1, image, Candidate(p, (30, 40),
            (30, 40, 54, 52), .9, 'current'), 1008, 'global')
        fit = Match(translation(30, 40)@np.linalg.inv(relation), .3, 20, .9,
            (owners[0].observations[0], owners[1].observations[0]))
        with patch.object(engine, '_matches', return_value=[fit]), patch.object(engine, '_shape_agreement', return_value=True):
            result = engine.associate(current)
        self.assertEqual(result.id, owners[1].id)
        self.assertEqual(current.template_id, 1)
        np.testing.assert_allclose(current.pose, translation(30, 40))
        self.assertEqual(len(engine.instances), 2)

    def test_sparse_fragment_uses_independent_measured_pose_and_trajectory(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        mark = self.mark(); p = (mark.max(2) > 0).astype(np.float32)
        image = np.zeros((200, 500, 3), np.uint8); image[30:94, 20:200] = mark
        seed = engine.observe(0, 0, image, Candidate(p, (20, 30),
            (20, 30, 200, 94), .9, 'seed'), 1008, 'global')
        template = engine.bank.create(seed); instance = engine._new_instance(template, seed)
        instance.poses = {f:Pose(f, seed.pose.copy(), .5, 'measured', f,
            (seed.id,), True) for f in range(24)}
        engine._motion(instance)
        fragment = p[:, :35].copy()
        obs = engine.observe(23, 23, image, Candidate(fragment, (20, 30),
            (20, 30, 55, 94), .9, 'fragment'), 1008, 'global')
        self.cache.put(obs.feature_key, xy=np.empty((0, 2)), desc=np.empty((0, 128)), size=np.empty(0))
        before = instance.poses[23]
        engine.associate(obs)
        self.assertEqual(obs.instance_id, instance.id)
        self.assertEqual(obs.association_reason, 'registered_fragment_and_validated_trajectory')
        self.assertIs(instance.poses[23], before)
        self.assertNotIn(obs.id, engine.pending.membership)
        self.assertEqual(obs.trajectory_evidence['measured_sources'], 23)

        # Nearby pixels outside the measured strokes remain unresolved.
        background = engine.observe(23, 23, image, Candidate(np.ones((4, 4), np.float32),
            (190, 30), (190, 30, 194, 34), .9, 'background'), 1008, 'global')
        self.assertIsNone(engine._registered_fragment(background))
        self.assertIsNone(background.instance_id)

        # Another measured occurrence explaining the same pixels blocks an
        # otherwise attractive trajectory rather than contaminating templates.
        other = Instance(1, template.id, 0, poses={23:Pose(23, seed.pose.copy(), .5,
            'measured', 23, (seed.id,), True)})
        engine.instances[1] = other
        contested = engine.observe(23, 23, image, Candidate(fragment, (20, 30),
            (20, 30, 55, 94), .9, 'contested'), 1008, 'global')
        self.assertIsNone(engine._registered_fragment(contested))
        self.assertIsNone(contested.instance_id)

    def test_strong_trajectory_supports_template_identity_without_sift_or_fake_registration(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        image = np.zeros((200, 500, 3), np.uint8)
        p = np.zeros((30, 100), np.float32); p[4:26, 4:30] = .8
        for tid, frame in enumerate((0, 26)):
            obs = engine.observe(frame, frame, image, Candidate(p, (20, 30),
                (20, 30, 120, 60), .9, str(tid)), 336, 'trajectory-identity')
            instance = engine.pending.defer(obs, 'new_template')
            instance.poses = {f:Pose(f, translation(20+f*.3, 30+.003*f*f), .3,
                'measured', f, presence=True) for f in range(frame, frame+24)}
        before = [len(i.poses) for i in engine.instances.values()]
        engine.reconciler.trajectory_identities()
        self.assertEqual(engine.bank.identity_members(0), {0, 1})
        self.assertEqual(set(engine.bank.members(0)), {0})
        self.assertEqual(engine.instances[1].continuation_of, 0)
        self.assertEqual([len(i.poses) for i in engine.instances.values()], before)
        # Explicit disappearance revokes the association on further review.
        engine.instances[0].poses[24] = Pose(24, translation(27, 32), .3,
            'absent', presence=False, reason='confirmed_disappearance')
        engine.reconciler.trajectory_identities()
        self.assertEqual(engine.bank.identity_members(0), {0})
        self.assertIsNone(engine.instances[1].continuation_of)

    def test_continuous_path_links_identity_despite_complementary_sam_dropout(self):
        class Detector:
            def detect(self, *args): return []
        engine = TemplateEngine(self.cfg, self.cache, Detector(), self.log, (200, 500))
        image = np.zeros((200, 500, 3), np.uint8)
        for tid, frame in enumerate((0, 26)):
            p = np.zeros((30, 100), np.float32)
            p[3:27, 4:20] = .8
            if tid == 0:
                p[3:27, 70:86] = .8
            else:
                p[3:9, 35:90] = .8
            obs = engine.observe(frame, frame, image, Candidate(p, (20, 30),
                (20, 30, 120, 60), .9, str(tid)), 336, 'changed-strokes')
            instance = engine.pending.defer(obs, 'new_template')
            instance.poses = {f:Pose(f, translation(20+.3*f, 30), .3,
                'measured', f, presence=True) for f in range(frame, frame+24)}
        engine.reconciler.trajectory_identities()
        self.assertEqual(engine.bank.identity_members(0), {0, 1})
        self.assertEqual(engine.instances[1].continuation_of, 0)
        # A path association does not fabricate a pixel-coordinate transform.
        self.assertEqual(set(engine.bank.members(0)), {0})


if __name__ == '__main__':
    unittest.main()
