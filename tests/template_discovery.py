"""Pre-identity discovery aggregation on in-memory synthetic evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.discovery import aggregate_candidates
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate
from uav.template_mask.records import Pose
from uav.template_mask.geometry import translation


SHAPE = (220, 360)


def candidate(name, rectangles, score=.9):
    p = np.zeros(SHAPE, np.float32)
    for l, t, r, b in rectangles:
        p[t:b, l:r] = .8
    # Deliberately identical, imprecise SAM boxes: membership uses pixels.
    return Candidate(p, (0, 0), (0, 0, 360, 220), score, name)


def expanded(c):
    p = np.zeros(SHAPE, np.float32)
    known = np.zeros(SHAPE, bool)
    x, y = c.origin
    h, w = c.probability.shape
    p[y:y+h, x:x+w] = c.probability
    known[y:y+h, x:x+w] = c.valid
    return p, known


class DiscoveryTests(unittest.TestCase):
    def fragments(self):
        a, b, small = (60, 70, 100, 80), (60, 145, 180, 155), (280, 20, 310, 35)
        return [candidate('line-a', [a]), candidate('line-b', [b]),
                candidate('small', [small]), candidate('whole', [a, b])]

    def test_whole_candidate_aggregates_fragments_before_identity_creation(self):
        raw = self.fragments()
        class Detector:
            def detect(self, *args, **kwargs):
                return raw
        with tempfile.TemporaryDirectory() as root:
            cache = EvidenceCache()
            try:
                cfg = TemplateConfig(sift_backend='cpu')
                engine = TemplateEngine(cfg, cache, Detector(), WarningLog(Path(root)/'warnings.jsonl'), SHAPE, debug=True)
                engine.first_frame(0, 29, np.zeros((*SHAPE, 3), np.uint8))
                self.assertEqual(len(engine.instances), 2)
                self.assertEqual(len(engine.bank.templates), 2)
                self.assertEqual(len(engine.bank.observations), 2)
                self.assertEqual(engine.discovery_groups[29]['raw_candidates'], 4)
                self.assertEqual(engine.discovery_groups[29]['objects'], 2)
                sources = {frozenset(o.source_candidates) for o in engine.bank.observations.values()}
                self.assertEqual(sources, {frozenset(['small']), frozenset(['line-a', 'line-b', 'whole'])})
                for c in raw:
                    stored = cache.get(cache.key('raw_discovery', 29, c.candidate_id))
                    np.testing.assert_array_equal(stored['probability'], c.probability)
            finally:
                cache.close()

    def test_exact_probability_union_without_output_dilation_or_score_weight(self):
        raw = self.fragments()
        raw[0].probability[70:80, 60:100] = .93
        raw[0].score = .31
        expected = np.maximum.reduce([c.probability for c in raw])
        actual = np.zeros(SHAPE, np.float32)
        owners = np.zeros(SHAPE, np.uint8)
        for c in aggregate_candidates(raw, SHAPE, .23):
            p, known = expanded(c)
            actual = np.maximum(actual, np.where(known, p, 0))
            owners += (p > .23) & known
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(owners, expected > .23)

    def test_pairwise_candidates_do_not_chain_three_distant_fragments(self):
        a, b, c = (30, 80, 45, 95), (140, 80, 155, 95), (260, 80, 275, 95)
        grouped = aggregate_candidates([candidate('ab', [a, b]), candidate('bc', [b, c])], SHAPE, .23)
        self.assertEqual(len(grouped), 2)
        self.assertTrue(all(g.box[2]-g.box[0] < 200 for g in grouped))

    def test_stray_pixel_cannot_annex_separate_watermark(self):
        a, b = (40, 70, 70, 100), (230, 70, 260, 100)
        almost_a = candidate('a-with-stray', [a, (231, 71, 232, 72)])
        grouped = aggregate_candidates([almost_a, candidate('b', [b])], SHAPE, .23)
        self.assertEqual(len(grouped), 2)

    def test_enclosed_other_object_is_unknown_not_background_evidence(self):
        surrounding = [(30, 30, 40, 180), (180, 30, 190, 180)]
        small = (100, 90, 115, 105)
        grouped = aggregate_candidates([candidate('big', surrounding), candidate('small', [small])], SHAPE, .23)
        self.assertEqual(len(grouped), 2)
        big = next(c for c in grouped if 'big' in c.source_candidates)
        p, known = expanded(big)
        self.assertFalse(known[90:105, 100:115].any())
        self.assertFalse(p[90:105, 100:115].any())

    def test_offset_clipping_and_unknown_pixels(self):
        p = np.full((40, 50), .8, np.float32)
        valid = np.ones_like(p, bool); valid[:10] = False
        raw = Candidate(p, (-10, -5), (-10, -5, 40, 35), .8, 'edge', valid)
        grouped = aggregate_candidates([raw], SHAPE, .23)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].box, (0, 5, 40, 35))
        p, known = expanded(grouped[0])
        self.assertEqual(int(((p > .23) & known).sum()), 40*30)

    def test_curve_motion_does_not_merge_separated_objects_by_box_overlap(self):
        for step in range(9):
            x = 100+step*10
            y = 100+round(15*np.sin(step/3))
            big = candidate('big', [(30, 30, 40, 190), (300, 30, 310, 190)])
            small = candidate('small', [(x, y, x+12, y+8)])
            grouped = aggregate_candidates([big, small], SHAPE, .23)
            self.assertEqual(len(grouped), 2)
            self.assertEqual({frozenset(c.source_candidates) for c in grouped},
                             {frozenset(['big']), frozenset(['small'])})

    def crossing_fixture(self, root):
        cache = EvidenceCache()
        cfg = TemplateConfig(sift_backend='cpu')
        engine = TemplateEngine(cfg, cache, None, WarningLog(Path(root)/'warnings.jsonl'), SHAPE)
        image = np.zeros((*SHAPE, 3), np.uint8)
        owners = []
        for i, x in enumerate((30, 260)):
            p = np.full((20, 30), .8, np.float32)
            c = Candidate(p, (x, 70), (x, 70, x+30, 90), .9, str(i))
            obs = engine.observe(0, 29, image, c, 1008, 'global')
            owner = engine._provisional(obs, 'new_template')
            owner.poses[1] = Pose(1, translation(x, 70), .5, 'measured', 31, (obs.id,), True)
            # Distinct current image fits land in adjacent parts of one SAM
            # component; they are not identities invented by its candidate id.
            owner.poses[2] = Pose(2, translation(100+i*20, 70), .5, 'measured', 34, (obs.id,), True)
            owners.append(owner)
        p = np.full((20, 50), .8, np.float32)
        mixed = Candidate(p, (100, 70), (100, 70, 150, 90), .9, 'joined',
                          source_candidates=('a', 'b', 'whole'))
        obs = engine.observe(2, 34, image, mixed, 1008, 'global')
        return cache, engine, owners, obs

    def test_mixed_discovery_keeps_two_existing_objects_without_teaching_templates(self):
        with tempfile.TemporaryDirectory() as root:
            cache, engine, owners, obs = self.crossing_fixture(root)
            try:
                versions = [t.version for t in engine.bank.templates.values()]
                references = [list(t.references) for t in engine.bank.templates.values()]
                with patch.object(engine, '_matches', side_effect=AssertionError('mixed pixels must be caught first')):
                    self.assertIsNone(engine.associate(obs))
                self.assertEqual(obs.crossing_instances, (0, 1))
                self.assertTrue(obs.ambiguous); self.assertFalse(obs.reliable)
                self.assertIsNone(obs.instance_id); self.assertIsNone(obs.template_id)
                self.assertEqual(len(engine.bank.templates), 2)
                self.assertEqual(len(engine.instances), 2)
                self.assertEqual([t.version for t in engine.bank.templates.values()], versions)
                self.assertEqual([t.references for t in engine.bank.templates.values()], references)
                self.assertEqual(len(engine.interactions.records), 1)
                self.assertTrue(engine.render(2, final=False)[70:90, 100:150].all())
                # Later separated observations can still attach to either
                # original owner: mixing has not removed or aliased them.
                self.assertEqual([i.id for i in engine.active_instances()], [0, 1])
                self.assertTrue(all(i.poses[2].status == 'measured' for i in owners))
            finally:
                cache.close()

    def test_whole_and_fragment_without_separate_history_are_not_crossing(self):
        with tempfile.TemporaryDirectory() as root:
            cache, engine, owners, obs = self.crossing_fixture(root)
            try:
                for owner in owners:
                    owner.poses[0].matrix = translation(100, 70)
                    owner.poses[1].matrix = translation(100, 70)
                self.assertFalse(engine._mixed_discovery(obs))
                self.assertFalse(obs.crossing_instances)
            finally:
                cache.close()

    def test_crossing_uses_visible_strokes_not_fraction_of_rotated_canvas_box(self):
        with tempfile.TemporaryDirectory() as root:
            cache, engine, owners, obs = self.crossing_fixture(root)
            try:
                # Rotation and earlier partial detections can leave a broad
                # template canvas. Empty canvas is not unobserved foreground.
                engine.bank.templates[owners[0].template_id].box=(-100,-100,130,120)
                engine.bank.templates[owners[0].template_id].version+=1
                engine.bank.revision+=1
                self.assertTrue(engine._mixed_discovery(obs))
                self.assertEqual(obs.crossing_instances,(0,1))
                self.assertIsNone(obs.template_id)
            finally:
                cache.close()

    def test_unbounded_prediction_cannot_claim_a_crossing(self):
        with tempfile.TemporaryDirectory() as root:
            cache, engine, owners, obs = self.crossing_fixture(root)
            try:
                owners[1].poses[2].uncertainty = 100
                owners[1].poses[2].status = 'uncertain'
                self.assertFalse(engine._mixed_discovery(obs))
            finally:
                cache.close()

    def test_short_matching_gap_still_holds_shared_pixels_out_of_templates(self):
        with tempfile.TemporaryDirectory() as root:
            cache, engine, owners, obs = self.crossing_fixture(root)
            try:
                owners[0].poses[2].status = 'uncertain'
                owners[0].poses[2].uncertainty = 8
                self.assertTrue(engine._mixed_discovery(obs))
                self.assertEqual(obs.crossing_instances, (0, 1))
                self.assertIsNone(obs.template_id)
            finally:
                cache.close()


if __name__ == '__main__':
    unittest.main()
