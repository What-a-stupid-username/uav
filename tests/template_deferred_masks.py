"""Deferred refinement contract on in-memory synthetic images only."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import Match
from uav.template_mask.geometry import dilate, translation
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Instance, Pose
from uav.template_mask.refinement import refine_templates, select_originals
from uav.template_mask.sam import context_crop, refinement_config
from uav.template_mask.review import uncertainty_banner, write_review


class Detector:
    def __init__(self, coarse, fine=()):
        self.coarse,self.fine=coarse,fine
        self.history=[]
        self.calls=0

    def detect(self,image,source_id,box,resolution,**kwargs):
        self.history.append(dict(source=source_id,box=box,resolution=resolution,**kwargs))
        self.calls+=1
        return self.coarse if box is None else self.fine


class DeferredTests(unittest.TestCase):
    def test_fusion_survives_unrelated_revision_but_not_own_validity_change(self):
        e=self.engine()
        e.first_frame(0,0,self.image)
        original=e.bank.fuse(0,coarse=True)
        e.bank.revision+=1
        with patch('uav.template_mask.templates.warp',side_effect=AssertionError('unrelated update')):
            self.assertIs(e.bank.fuse(0,coarse=True),original)
        obs=e.bank.observations[e.bank.templates[0].observations[0]]
        obs.ambiguous=True
        e.bank.revision+=1
        changed=e.bank.fuse(0,coarse=True)
        self.assertFalse(changed['sources'].size)
        self.assertFalse(changed['mask'].any())

    def test_deferred_identity_and_explicit_coarse_share_exact_fusion(self):
        e=self.engine()
        e.first_frame(0,0,self.image)
        identity=e.bank.fuse(0,identity=True)
        with patch('uav.template_mask.templates.warp',side_effect=AssertionError('duplicate fusion')):
            coarse=e.bank.fuse(0,coarse=True)
        self.assertIs(identity,coarse)
        self.assertFalse(e.bank.fuse(0)['sources'].size)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.cache=EvidenceCache()
        self.config=TemplateConfig(sift_backend='cpu')
        self.log=WarningLog(Path(self.temp.name)/'warnings.jsonl')
        self.image=np.zeros((200,500,3),np.uint8)
        cv2.putText(self.image,'UAV-2026',(24,75),cv2.FONT_HERSHEY_SIMPLEX,.9,(240,240,240),2)
        self.p=np.zeros((70,190),np.float32)
        self.p[15:65,3:165]=.8
        self.box=(20,20,210,90)
        self.coarse=Candidate(self.p,(20,20),self.box,.9,'coarse')

    def tearDown(self):
        self.cache.close();self.temp.cleanup()

    def engine(self,fine=()):
        detector=Detector([self.coarse],fine)
        return TemplateEngine(self.config,self.cache,detector,self.log,self.image.shape[:2],debug=True)

    def test_unassigned_residual_preserves_holes_and_new_strokes(self):
        from uav.template_mask.rendering import MaskRenderer
        e=self.engine();instance,obs=self.seed(e)
        obs.reliable=False;obs.ownership_confirmed=False;obs.ambiguous=True
        raw=e._observed_candidate(obs)
        yy,xx=np.nonzero(raw)
        explained=raw.copy()
        # Leave actual strokes inside an otherwise explained region, as well
        # as strokes at its edge. Subtraction must work per pixel, not per box.
        hole=(int(yy[len(yy)//2]),int(xx[len(xx)//2]))
        explained[hole]=False
        explained[:,int(np.median(xx)):]=False
        expected=raw & ~explained
        metrics=dict(dilation_radii=[],uncertainty=[])
        renderer=MaskRenderer(e)
        with patch.object(renderer,'finish') as finish:
            layers=list(renderer.unassigned(0,metrics,True,explained=explained))
            self.assertEqual(len(layers),1)
            actual=finish.call_args.args[0]
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(actual[hole])
        empty=list(MaskRenderer(e).unassigned(0,metrics,True,explained=raw))
        self.assertEqual(empty,[])

    def seed(self,e,frame=0,instance=None,source=None):
        source=frame if source is None else source
        obs=e.observe(frame,source,self.image,self.coarse,1008,'global')
        if instance is None:
            instance=e.associate(obs)
        else:
            obs.pose,obs.template_id,obs.reliable=translation(20,20),instance.template_id,True
            e._assign(instance,obs)
        e.source_ids[frame]=source
        return instance,obs

    def test_first_pass_never_calls_local_and_tracks_from_coarse_features(self):
        e=self.engine()
        with patch.object(e, 'render', side_effect=AssertionError('stage1 must not render masks')):
            for f in range(17):e.first_frame(f,f,self.image)
        self.assertEqual(len(e.first_pass_marks),17)
        self.assertFalse(any('first_mask' in key for key in e.cache.memory))
        self.assertTrue(e.detector.history)
        self.assertTrue(all(c['box'] is None for c in e.detector.history))
        self.assertEqual(e.stats['template_calls'],0)
        self.assertNotIn('pending_rechecks',e.stats)
        self.assertTrue(all(o.mask_role=='discovery' for o in e.bank.observations.values()))
        self.assertTrue(e.bank.fuse(0,identity=True)['sources'].size)
        self.assertFalse(e.bank.fuse(0)['sources'].size)
        self.assertTrue(e.render(1,final=False).any())
        self.assertEqual(e.instances[0].poses[1].status,'measured')

    def test_non_debug_does_not_keep_display_only_candidates_or_marks(self):
        e = self.engine()
        e.debug = False
        for frame in range(9):
            e.first_frame(frame, frame, self.image)
        self.assertFalse(e.first_pass_marks)
        self.assertFalse(e.first_pass_observations)
        self.assertFalse(any('raw_discovery' in key or 'first_mask' in key for key in e.cache.memory))
        self.assertTrue(e.bank.observations)
        self.assertTrue(e.bank.fuse(0, coarse=True)['sources'].size)
        self.assertTrue(e.render(0).any())
        self.assertFalse(any('conservative_mask' in key for key in e.cache.memory))

    def test_debug_snapshot_excludes_later_stage2_observations(self):
        e = self.engine()
        e.first_frame(0, 0, self.image)
        saved = e.first_pass_observations[0]
        e.observe(0, 0, self.image, self.coarse, 1008, ('refinement', 0, 'later'))
        self.assertEqual(e.first_pass_observations[0], saved)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(e.frame_observations[0]), 2)

    def test_boundary_probe_uses_sam_only_for_presence(self):
        e=self.engine();instance,_=self.seed(e)
        e.source_ids[1]=1
        instance.poses[1]=Pose(1,translation(20,20),.5,'uncertain')
        with patch.object(e,'_track'),patch.object(e,'_matches',return_value=[]):
            self.assertTrue(e.presence._probe(instance,1,lambda f:self.image))
        self.assertEqual(len(e.detector.history),1)
        self.assertIsNone(e.detector.history[0]['box'])
        self.assertEqual(len(e.bank.observations),1)
        self.assertEqual(instance.poses[1].reason,'boundary_sam_presence')

    def test_second_pass_uses_probe_then_completion_with_shared_budget(self):
        e=self.engine()
        for f in range(17):e.first_frame(f,f,self.image)
        first_calls=e.detector.calls
        first_count=len(e.detector.history)
        e.second_pass(lambda f:self.image)
        self.assertEqual(e.sam_stages['first_pass'],first_calls)
        self.assertEqual(len(e.refinement_batches),2)
        self.assertLessEqual(e.stats['template_calls'],8)
        local=[c for c in e.detector.history[first_count:] if c['box'] is not None]
        self.assertTrue(local)
        self.assertTrue(all(c.get('refinement') and c['resolution']==1008 for c in local))
        self.assertEqual(e.sam_stages['second_pass_refinement'],len(local))
        self.assertTrue(e.render(8).any())

    def test_failed_refinement_falls_back_with_exactly_double_adaptive_radius(self):
        e=self.engine();instance,_=self.seed(e)
        before=instance.poses[0].matrix.copy()
        refine_templates(e,lambda f:self.image)
        output=e.render(0)
        coarse=e.bank.fuse(instance.template_id,coarse=True)
        from uav.template_mask.geometry import warp
        raw=warp(coarse['mean']>.23,instance.poses[0].matrix@translation(*coarse['origin']),e.shape,nearest=True)>0
        metrics={ 'dilation_radii':[] }
        e.renderer.dilate(raw,metrics)
        radius=metrics['dilation_radii'][0]
        np.testing.assert_array_equal(output,dilate(raw,2*radius))
        self.assertEqual(e.render_metrics[0]['dilation_radii'],[2*radius])
        self.assertTrue(e.render_metrics[0]['coarse_fallback'])
        self.assertIn('COARSE MASK FALLBACK',uncertainty_banner(e.render_metrics[0])[0])
        self.assertEqual(self.log.rows[-1]['action'],'coarse_mask_double_dilation')
        np.testing.assert_array_equal(instance.poses[0].matrix,before)

    def test_successful_refinement_excludes_coarse_noise_and_uses_normal_dilation(self):
        p=np.zeros_like(self.p);p[20:35,10:30]=.9
        fine=Candidate(p,(20,20),self.box,.95,'fine')
        e=self.engine([fine]);instance,_=self.seed(e)
        # Isolate geometric registration from SAM: its independent support is
        # tested in the existing SIFT suite, including partial candidates.
        fit=Match(translation(20,20),.5,12,.7,tuple(instance.observations))
        with patch.object(e,'_matches',return_value=[fit]):refine_templates(e,lambda f:self.image)
        out=e.render(0)
        self.assertTrue(out[45,35]);self.assertFalse(out[60,150])
        self.assertNotIn('coarse_fallback',e.render_metrics[0])
        self.assertEqual(e.render_metrics[0]['dilation_radii'],[4])
        self.assertTrue(e.bank.fuse(0)['refined'])
        self.assertEqual(e.bank.fuse(0)['sources'].tolist(),[0])
        call=e.detector.history[0]
        self.assertEqual(call['resolution'],1008);self.assertTrue(call['refinement'])

    def test_empty_fine_mean_falls_back_without_clipping_original_coarse_strokes(self):
        fine=Candidate(np.full_like(self.p,.1),(20,20),self.box,.95,'empty')
        e=self.engine([fine]);instance,_=self.seed(e)
        fit=Match(translation(20,20),.5,12,.7,tuple(instance.observations))
        with patch.object(e,'_matches',return_value=[fit]):refine_templates(e,lambda f:self.image)
        self.assertTrue(len(e.bank.fuse(0)['sources']))
        self.assertFalse((e.bank.fuse(0)['mean']>.23).any())
        out=e.render(0)
        self.assertTrue(out[60,150])
        self.assertTrue(e.render_metrics[0]['coarse_fallback'])

    def test_coarse_canvas_retains_out_of_proposal_strokes_after_fine_extent_shrinks(self):
        e=self.engine();instance,coarse=self.seed(e)
        e.bank.templates[0].box=(0,0,20,20)
        e.bank.templates[0].version+=1
        out=e.render(0)
        self.assertTrue(out[60,150])
        self.assertTrue(e.render_metrics[0]['coarse_fallback'])

    def test_refinement_is_once_per_shared_template_with_source_dedup(self):
        e=self.engine();instance,_=self.seed(e)
        for frame in range(1,20):self.seed(e,frame,instance)
        # Another occurrence contributes the same source only once.
        self.seed(e,20,instance,source=19)
        selected=select_originals(e,e.bank.members(0))
        self.assertEqual(len(selected),8)
        self.assertEqual(len({o.source_id for o in selected}),8)
        refine_templates(e,lambda f:self.image)
        self.assertEqual(len(e.detector.history),8)
        refine_templates(e,lambda f:self.image)
        self.assertEqual(len(e.detector.history),8)
        self.assertEqual(len(e.instances),1)

    def test_registered_linked_templates_share_one_batch(self):
        e=self.engine();a,_=self.seed(e)
        bobs=e.observe(4,4,self.image,self.coarse,1008,'global')
        b=e._provisional(bobs,'new_template')
        e.bank.propose_link(a.template_id,b.template_id,np.eye(3),0)
        e.bank.propose_link(a.template_id,b.template_id,np.eye(3),4)
        refine_templates(e,lambda f:self.image)
        self.assertEqual(len(e.refinement_batches),1)
        self.assertEqual(e.refinement_batches[0]['members'],[0,1])
        self.assertEqual(len(e.detector.history),2)

    def test_probe_stops_on_success_and_merged_completion_reuses_both_sources(self):
        e=self.engine(fine=[self.coarse]);a,_=self.seed(e)
        for frame in range(1,10):self.seed(e,frame,a)
        obs=e.observe(10,10,self.image,self.coarse,1008,'global')
        b=e._provisional(obs,'new_template')
        for frame in range(11,20):self.seed(e,frame,b)
        with patch.object(e,'_matches',return_value=[Match(translation(20,20),.7,12,.8,())]):
            refine_templates(e,lambda f:self.image,probe=True)
            self.assertEqual(len(e.detector.history),2)
            sources={row['source'] for row in e.detector.history}
            self.assertEqual(len(sources),2)
            for source in (0,10):
                e.bank.propose_link(a.template_id,b.template_id,np.eye(3),source)
            refine_templates(e,lambda f:self.image)
        self.assertEqual(len(e.detector.history),8)
        final_sources=[row['source'] for row in e.detector.history]
        self.assertEqual(len(set(final_sources)),8)
        self.assertTrue(sources.issubset(final_sources))
        self.assertEqual(len(e.bank.fuse(0)['sources']),8)

    def test_failed_probe_tasks_are_not_repeated_during_completion(self):
        e=self.engine();owner,_=self.seed(e)
        for frame in range(1,20):self.seed(e,frame,owner)
        refine_templates(e,lambda f:self.image,probe=True)
        self.assertEqual(len(e.detector.history),8)
        refine_templates(e,lambda f:self.image)
        self.assertEqual(len(e.detector.history),8)
        self.assertTrue(e.refinement_batches[-1]['coarse_fallback'])

    def test_retired_duplicate_stops_tracking_and_revocation_restores_it(self):
        e=self.engine();first,_=self.seed(e)
        b=e.observe(1,1,self.image,self.coarse,1008,'global')
        second=e._provisional(b,'overlap_without_identity')
        for instance in (first,second):
            instance.poses={f:Pose(f,translation(20,20),.5,'measured',f,presence=True)
                            for f in (0,1)}
        for source in (0,1):e.bank.propose_link(0,1,np.eye(3),source)
        e.reconciler.refresh_aliases()
        self.assertEqual([i.id for i in e.active_instances()],[first.id])
        with patch.object(e.matcher,'extract',side_effect=AssertionError('duplicate must not track')):
            e._track(second,2,2,self.image)
            self.assertIsNone(e.presence._probe(second,2,lambda f:self.image))
        self.assertNotIn(2,second.poses)
        e.bank.revoke(0,1,'contradiction')
        self.assertEqual(len(e.active_instances()),2)
        e.reconciler.refresh_aliases()
        self.assertIsNone(second.alias_of)
        self.assertEqual(len(second.poses),2)

    def test_shared_reference_limit_and_distinct_occurrences_stay_active(self):
        e=self.engine();first,_=self.seed(e)
        for f in range(1,9):self.seed(e,f,first)
        b=e.observe(9,9,self.image,self.coarse,1008,'global')
        second=e._provisional(b,'new_template')
        for f in range(10,18):self.seed(e,f,second)
        for source in (30,31):e.bank.propose_link(0,1,np.eye(3),source)
        # Shared shape does not merge occurrences without common measured poses.
        e.reconciler.refresh_aliases()
        self.assertEqual(len(e.active_instances()),2)
        refs=e.bank.matching_references(0)
        self.assertEqual(len(refs),8)
        self.assertEqual(len({r.source_id for r in refs}),8)
        self.assertEqual({e.bank.observations[r.observation_id].template_id for r in refs},{0,1})
        target=first.observations[0]
        with patch.object(e.matcher,'fit',return_value=[]) as fit:
            e._matches(e.bank.templates[0],e.bank.observations[target].feature_key)
        self.assertEqual(fit.call_count,8)
        e.bank.revoke(0,1,'wrong_link')
        self.assertTrue(all(e.bank.observations[r.observation_id].template_id==0
                            for r in e.bank.matching_references(0)))

    def test_progressive_coarse_extent_preserves_anchor_and_partial_identity(self):
        e=self.engine();first,seed=self.seed(e)
        anchor=e.bank.templates[0].anchor
        pose=seed.pose.copy()
        p=self.p.copy();p[:5,:]=.9
        wider=Candidate(p,(20,10),(20,10,210,90),.9,'complete')
        obs=e.observe(1,1,self.image,wider,1008,'global')
        obs.pose,obs.template_id,obs.reliable=pose.copy(),first.template_id,True
        e._assign(first,obs)
        self.assertEqual(e.bank.templates[0].anchor,anchor)
        np.testing.assert_equal(seed.pose,pose)
        self.assertLess(e.bank.templates[0].box[1],0)

    def test_crop_exactly_1p7_or_35_percent_short_side(self):
        for shape,box,side in (((1080,1934),(900,500,1185,569),485),
                               ((1080,1934),(900,500,940,520),378),
                               ((1920,1080),(0,0,300,60),510)):
            cfg=refinement_config(self.config,shape)
            crop=context_crop(box,shape,cfg)
            self.assertEqual(crop[2]-crop[0],side)
            self.assertEqual(crop[3]-crop[1],side)

    def test_rejected_fine_pixels_do_not_bypass_coarse_fallback(self):
        bad=Candidate(np.ones_like(self.p),(20,20),self.box,.95,'bad')
        e=self.engine([bad]);self.seed(e)
        with patch.object(e,'_matches',return_value=[]):refine_templates(e,lambda f:self.image)
        out=e.render(0)
        self.assertFalse(out[22,207])
        self.assertTrue(e.render_metrics[0]['coarse_fallback'])
        self.assertFalse(e.render_metrics[0]['uncertainty'])

    def test_fallback_preview_shows_the_mask_actually_used_and_original_sources(self):
        e=self.engine();self.seed(e)
        final=e.render(0)
        e._mark_first_frame(0)
        e.cache.put(e.cache.key('final_mask',0),packed=np.packbits(final))
        frames=Mock(count=1,fps=24,return_value=self.image)
        encoder=Mock();encoder.wait.return_value=0
        with patch('uav.template_mask.review_video.subprocess.Popen',return_value=encoder), \
             patch('uav.ffmpeg.executable',return_value='ffmpeg'), \
             patch('uav.template_mask.review.cv2.imwrite',return_value=True) as write:
            write_review(e,frames,self.temp.name)
        preview=write.call_args.args[1]
        self.assertTrue(preview.any())
        rows=json.loads((Path(self.temp.name)/'templates.json').read_text())
        self.assertTrue(rows[0]['coarse_fallback'])
        self.assertFalse(rows[0]['refined'])
        self.assertEqual(rows[0]['mask_sources'],[0])
        self.assertEqual([s['source_frame'] for s in rows[0]['sources']],[0])

    def test_fallback_warning_file_has_one_json_record_per_line(self):
        e=self.engine();self.seed(e);e.render(0)
        self.log.add(2,3,8,'another_reason','cover_uncertainty')
        self.log.flush()
        rows=[json.loads(line) for line in self.log.path.read_text().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['action'],'coarse_mask_double_dilation')
        self.assertEqual(rows[1]['instance'],2)

    def test_publication_labels_double_dilation_and_preserves_prior_results(self):
        from uav.template_mask.benchmark import publish
        root=Path(self.temp.name)/'page';root.mkdir()
        (root/'index.html').write_text('<style></style><section>previous result</section>')
        folder=root/'deferred-test/video-04';folder.mkdir(parents=True)
        (folder/'mask-review.mp4').touch()  # Existence-only fixture, never decoded.
        (folder/'templates.json').write_text(json.dumps([dict(id=0,path='template-0.png',
            coarse_fallback=True,sources=[dict(source_frame=0)])]))
        report=dict(complete=True,skipped=False,config=self.config.as_dict(),review={},
            frames=24,rendered_frames=24,timing={},total_seconds=1)
        rows=[dict(index=4,name='synthetic',start=0,status='complete',report=report)]
        publish(root,'deferred-test',rows)
        page=(root/'index.html').read_text()
        self.assertIn('8–24',page)
        self.assertIn('粗 mask 回退 · 原始帧 0',page)
        self.assertLess(page.index('template-mask-deferred-test'),page.index('previous result'))


if __name__=='__main__':unittest.main()
