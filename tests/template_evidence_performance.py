"""Lossless cache optimizations, tested with synthetic feature arrays only."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from pathlib import Path
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.features import Match, SiftMatcher, ratio_matches
from uav.template_mask.geometry import translation
from uav.template_mask.records import Observation, Reference, Template
from uav.template_mask.records import Pose, Candidate
from uav.template_mask.motion import MotionPrior, MotionFitCache, fit_pose_component


class EvidencePerformanceTests(unittest.TestCase):
    def test_prepared_windows_reuse_exact_inputs_and_invalidate_changed_uncertainty(self):
        from uav.template_mask.motion import prepare_window
        cache=MotionFitCache()
        times=np.arange(12,dtype=float)
        values=np.zeros((12,4));values[:,2]=np.linspace(2.8,3.5,12)
        values[values[:,2]>np.pi,2]-=2*np.pi
        sigmas=np.ones(12)
        expected=prepare_window(times,values,sigmas,100.)
        with patch('uav.template_mask.motion.prepare_window',wraps=prepare_window) as compute:
            first=cache.prepare_window(times,values,sigmas,100.)
            second=cache.prepare_window(times.copy(),values.copy(),sigmas.copy(),100.)
            self.assertIs(first,second)
            self.assertEqual(compute.call_count,1)
            sigmas[:]=2
            changed=cache.prepare_window(times,values,sigmas,100.)
            self.assertEqual(compute.call_count,2)
            self.assertNotEqual(first[2],changed[2])
        for a,b in zip(first,expected):np.testing.assert_array_equal(a,b)
        self.assertFalse(first[0].flags.writeable)
        self.assertFalse(first[1].flags.writeable)

    def test_reference_location_prefix_reuses_fit_but_not_other_threshold(self):
        fits=[Match(translation(12+i,9),.7,18,.8,reference_indices=(0,1)) for i in range(2)]
        with patch.object(self.matcher,'fit',return_value=fits) as compute:
            self.matcher.reference_matches('reference','target',4)
            one=self.matcher.reference_matches('reference','target',1)
            self.assertEqual(len(one),1)
            np.testing.assert_array_equal(one[0].matrix,fits[0].matrix)
            self.assertEqual(compute.call_count,1)
            self.matcher.reference_matches('reference','target',1,min_inliers=4)
            self.assertEqual(compute.call_count,2)

    def test_failed_first_location_also_ends_multi_location_query(self):
        with patch.object(self.matcher,'fit',return_value=[]) as compute:
            self.matcher.reference_matches('reference','target',1)
            self.matcher.reference_matches('reference','target',4)
            self.assertEqual(compute.call_count,1)

    def test_group_match_cache_ignores_version_but_tracks_geometry(self):
        obs=Observation('obs',0,1,'a',(0,0,20,20),(0,0),
            'probability','reference',.9,1008,template_id=0,
            pose=translation(),reliable=True)
        template=Template(0,(0,0,20,20),references=[Reference('obs',1,'reference',1.)])
        with patch.object(self.matcher,'fit',return_value=[]) as compute:
            self.matcher.template_matches(template,{'obs':obs},'target')
            template.version+=1
            with patch.object(self.matcher,'_prepare_neighbours',
                    side_effect=AssertionError('unchanged match inputs recomputed')):
                self.matcher.template_matches(template,{'obs':obs},'target')
            self.assertEqual(compute.call_count,1)
            template.box=(0,0,21,20)
            with patch.object(self.matcher,'_prepare_neighbours',wraps=self.matcher._prepare_neighbours) as prepare:
                self.matcher.template_matches(template,{'obs':obs},'target')
                self.assertEqual(prepare.call_count,1)
            obs.pose=translation(1,0)
            with patch.object(self.matcher,'_prepare_neighbours',wraps=self.matcher._prepare_neighbours) as prepare:
                self.matcher.template_matches(template,{'obs':obs},'target')
                self.assertEqual(prepare.call_count,1)
            for field,value in (('ambiguous',True),('reliable',False),('source_id',2)):
                setattr(obs,field,value)
                with patch.object(self.matcher,'_prepare_neighbours',wraps=self.matcher._prepare_neighbours) as prepare:
                    self.matcher.template_matches(template,{'obs':obs},'target')
                    self.assertEqual(prepare.call_count,1)
            with patch.object(self.matcher,'_prepare_neighbours',wraps=self.matcher._prepare_neighbours) as prepare:
                self.matcher.template_matches(template,{'obs':obs},'target',
                    member_transforms={0:translation(2,0)})
                self.assertEqual(prepare.call_count,1)
            template.references=[]
            with patch.object(self.matcher,'_prepare_neighbours',wraps=self.matcher._prepare_neighbours) as prepare:
                self.matcher.template_matches(template,{'obs':obs},'target')
                self.assertEqual(prepare.call_count,1)

    def test_continuity_alternatives_reuse_validation_and_own_returned_values(self):
        from uav.template_mask.geometry import discrepancy, footprint
        rows=[Pose(f,translation(20+f,40),.7,'measured',f,(),True) for f in range(32)]
        prior=MotionPrior(rows,(0,0,100,30),(50,15)).excluding_source(17)
        expected=[]
        seen=[]
        segments=sorted((s for s in prior.segments if all(m is not None for m in s[0])),
            key=lambda s:(max(s[1]-17,17-s[2],0),-(s[2]-s[1])))
        for segment in [None,*segments]:
            prediction=prior.predict(17,segment=segment)
            if (prediction is None or not prediction[2] or not np.isfinite(prediction[1])
                    or prediction[1]>.08*footprint(prior.box)):
                continue
            matrix,sigma,_=prediction
            if any(discrepancy(matrix,m,prior.box)<=sigma for m in seen):continue
            seen.append(matrix)
            evidence=prior.continuity(17,matrix,sigma,segment=segment)
            if evidence is not None:expected.append((matrix,sigma,evidence))
        self.assertTrue(expected)
        with patch.object(prior,'continuity',wraps=prior.continuity) as validate:
            first=prior.continuity_predictions(17)
            calls=validate.call_count
            first[0][0][0,2]=999
            first[0][2]['components'][0]='mutated'
            actual=prior.continuity_predictions(17)
            self.assertEqual(validate.call_count,calls)
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(a[1:],b[1:])
        self.assertNotIn('_continuity_predictions',prior.__getstate__())

    def test_match_cache_keys_do_not_serialize_nested_evidence_ids(self):
        obs=Observation('obs',0,1,'a',(0,0,20,20),(0,0),
            'probability','reference',.9,1008,template_id=0,
            pose=translation(),reliable=True)
        template=Template(0,(0,0,20,20),references=[Reference('obs',1,'reference',1.)])
        with patch.object(self.cache,'key',side_effect=AssertionError('JSON key on matching path')):
            with patch.object(self.matcher,'fit',return_value=[]):
                self.assertEqual(self.matcher.template_matches(template,{'obs':obs},'target'),[])
                self.assertEqual(self.matcher.template_matches(template,{'obs':obs},'target'),[])

    def test_temporal_view_selection_preserves_exhaustive_utilities_and_ties(self):
        from uav.template_mask.templates import TemplateBank
        bank=TemplateBank(self.cache,TemplateConfig(sift_backend='cpu'))
        rng=np.random.default_rng(28)
        for count in (8,9,16,40):
            for tied in (False,True):
                sources=rng.choice(2000,count,replace=False).tolist()
                quality={s:1. if tied else float(rng.random()) for s in sources}
                expected={s:s for s in sources}
                while len(expected)>bank.config.mask_reference_limit:
                    span=max(1,max(expected)-min(expected))
                    utility={s:quality[s]+.1*min(abs(s-t) for t in expected if t!=s)/span for s in expected}
                    del expected[min(expected,key=lambda s:(utility[s],s))]
                with patch.object(bank,'_view_quality',side_effect=lambda jobs:quality[jobs]):
                    actual=bank._select_views({s:s for s in sources})
                self.assertEqual(list(actual),list(expected))

    def test_prediction_reuses_immutable_prior_and_returns_owned_matrix(self):
        rows=[Pose(f,translation(20+f,40),.7,'measured',f,(),True) for f in range(32)]
        prior=MotionPrior(rows,(0,0,100,30),(50,15))
        expected=prior._component_prediction(17)
        with patch.object(prior,'_component_prediction',wraps=prior._component_prediction) as compute:
            first=prior.predict(17)
            first[0][0,2]=999
            second=prior.predict(17)
            self.assertEqual(compute.call_count,1)
        np.testing.assert_array_equal(second[0],expected[0])
        self.assertEqual(second[1:],expected[1:])
        self.assertNotIn('_predictions',prior.__getstate__())
        segment=prior._window(17)
        expected_segment=prior._segment_prediction(17,segment)
        with patch.object(prior,'_segment_prediction',wraps=prior._segment_prediction) as compute:
            prior.predict(17,segment=segment)[0][0,2]=999
            reused=prior.predict(17,segment=segment)
            self.assertEqual(compute.call_count,1)
        np.testing.assert_array_equal(reused[0],expected_segment[0])
        self.assertEqual(reused[1:],expected_segment[1:])

    def test_neighbours_reused_across_ransac_options(self):
        rng=np.random.default_rng(14)
        xy=rng.uniform(0,60,(24,2))
        desc=rng.random((24,128),dtype=np.float32)
        for key,shift in (('reference',0),('target',10)):
            self.cache.put(key,xy=xy+shift,desc=desc,size=np.ones(24))
        with patch('uav.template_mask.features.ratio_matches',wraps=ratio_matches) as search:
            one=self.matcher.reference_matches('reference','target',1)
            many=self.matcher.reference_matches('reference','target',4,min_inliers=4)
            self.assertEqual(search.call_count,1)
        np.testing.assert_array_equal(one[0].matrix,many[0].matrix)

    def test_pending_residuals_are_reused_without_changing_predictions(self):
        from uav.template_mask.geometry import components, footprint
        from uav.template_mask.numerics import quantile
        rows=[Pose(f,translation(20+(35 if f==31 else 0),40),.7,
                   'measured',f,(),True) for f in range(32)]
        prior=MotionPrior(rows,(0,0,100,30),(50,15))
        expected=tuple(tuple(float(quantile([
            abs(components(prior.evidence[k].matrix,prior.anchor)[j]
                -model.predict(prior._times[k]))*(1. if j<2 else footprint(prior.box))
            for k in ids],.8)) for model,ids,_ in changes)
            for j,changes in enumerate(prior.unconfirmed_changes))
        self.assertTrue(any(expected))
        self.assertEqual(prior._pending_deviations,expected)
        with patch('uav.template_mask.motion.quantile',side_effect=AssertionError('residual recomputed')):
            for frame in (31,32,33,31):
                prior.predict(frame)
        self.assertNotIn('_pending_deviations',prior.__getstate__())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.matcher = SiftMatcher(self.cache, TemplateConfig(sift_backend='cpu'))
        for key in ('reference', 'target'):
            self.cache.put(key, xy=np.array([[0, 0], [10, 10]], np.float32),
                           desc=np.zeros((2, 128), np.float32), size=np.ones(2))

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def test_memory_cache_never_serializes_or_evicts_arrays(self):
        with patch('numpy.savez_compressed', side_effect=AssertionError('must not serialize')), \
             patch('builtins.open', side_effect=AssertionError('must not read/write files')):
            original = np.arange(4096, dtype=np.float32)
            self.cache.put('data', value=original)
            self.cache.put('other', value=np.arange(10000))
            original[0] = -1
            self.assertEqual(self.cache.get('data')['value'][0], 0)
            self.assertIn('reference', self.cache)
            self.assertIn('target', self.cache)
            self.assertEqual(self.cache.stats()['storage'], 'memory')
            self.assertEqual(self.cache.stats()['disk_bytes'], 0)
            self.assertGreater(self.cache.stats()['memory_bytes'], 80000)
        self.assertFalse(hasattr(self.cache, 'db'))
        self.assertFalse(hasattr(self.cache, 'directory'))

    def test_overwrite_counts_actual_arrays_and_close_releases_memory(self):
        self.cache.close()
        self.cache = EvidenceCache()
        self.cache.put('same', value=np.zeros(100, np.float32))
        self.assertEqual(self.cache.bytes, 400)
        self.cache.put('same', value=np.zeros(10, np.float32))
        self.assertEqual(self.cache.bytes, 40)
        self.assertEqual(self.cache.peak_bytes, 400)
        self.cache.close()
        self.assertEqual(self.cache.bytes, 0)
        self.assertFalse(self.cache.memory)
        with self.assertRaises(KeyError):
            self.cache.get('same')

    def test_reference_pair_reuse_is_lossless_and_returned_matrices_are_owned(self):
        fit = Match(translation(12, 9), .7, 18, .8, reference_indices=(0,3,17))
        with patch.object(self.matcher, 'fit', return_value=[fit]) as compute:
            first = self.matcher.reference_matches('reference', 'target', 4)
            first[0].matrix[0, 2] = 999
            reused = self.matcher.reference_matches('reference', 'target', 4)
            self.assertEqual(compute.call_count, 1)
        np.testing.assert_array_equal(reused[0].matrix, translation(12, 9))
        self.assertEqual((reused[0].uncertainty, reused[0].inliers, reused[0].coverage), (.7, 18, .8))
        self.assertEqual(reused[0].reference_indices,(0,3,17))
        self.assertEqual(self.matcher.reference_cache_hits, 1)

    def test_keyframe_rigidity_combines_cached_candidates_without_matching_again(self):
        from types import SimpleNamespace
        observations={'seed':SimpleNamespace(id='seed',feature_key='reference',
            reliable=True,ambiguous=False,source_id=0,box=(0,0,100,40))}
        for source,offset in ((1,80),(2,180)):
            self.matcher.begin_frame_evidence()
            for target,shift,ids in (('left',0,range(12)),('right',offset,range(12,24))):
                fit=Match(translation(shift,0),.7,12,.3,reference_indices=tuple(ids))
                key=target+str(source)
                self.cache.put(key,**self.cache.get('target'))
                with patch.object(self.matcher,'fit',return_value=[fit]):
                    self.matcher.reference_matches('reference',key)
                with patch.object(self.matcher,'fit',side_effect=AssertionError('cached pair')):
                    self.matcher.reference_matches('reference',key)
            with patch.object(self.matcher,'fit',side_effect=AssertionError('review cannot search')):
                rejected=self.matcher.contradicted_references(observations,source)
            self.assertEqual(rejected,[] if source==1 else ['seed'])
            self.assertIsNone(self.matcher.frame_pairs)

    def test_keyframe_evidence_excludes_historical_target_queries(self):
        fit=Match(translation(),.7,12,.3,reference_indices=tuple(range(12)))
        self.matcher.feature_sources.update(current=10,historical=9)
        self.matcher.begin_frame_evidence(10)
        self.matcher._record_pair('reference','historical',[fit])
        self.matcher._record_pair('reference','current',[fit])
        self.assertEqual(set(self.matcher.frame_pairs['reference']),{'current'})

    def test_failed_pairs_are_cached_and_parameters_have_distinct_entries(self):
        with patch.object(self.matcher, 'fit', return_value=[]) as compute:
            self.assertEqual(self.matcher.reference_matches('reference', 'target'), [])
            self.assertEqual(self.matcher.reference_matches('reference', 'target'), [])
            self.assertEqual(compute.call_count, 1)
            self.matcher.reference_matches('reference', 'target', min_inliers=4)
            self.matcher.reference_matches('reference', 'target', max_locations=4)
            self.assertEqual(compute.call_count, 2)

    def test_template_updates_reuse_image_fit_but_apply_current_coordinates(self):
        obs = Observation('obs', 0, 1, 'a', (0, 0, 20, 20), (0, 0),
                          'probability', 'reference', .9, 1008,
                          template_id=0, pose=translation(3, 4), reliable=True)
        template = Template(0, (0, 0, 20, 20), references=[Reference('obs', 1, 'reference', 1.)])
        with patch.object(self.matcher, 'fit', return_value=[Match(translation(12, 9), .7, 18, .8)]) as compute:
            first = self.matcher.template_matches(template, {'obs':obs}, 'target')
            obs.pose = translation(7, 8)
            second = self.matcher.template_matches(template, {'obs':obs}, 'target')
            self.assertEqual(compute.call_count, 1)
        np.testing.assert_array_equal(first[0].matrix, translation(15, 13))
        np.testing.assert_array_equal(second[0].matrix, translation(19, 17))
        self.assertEqual(second[0].references, ('obs',))

    def test_group_cache_rechecks_geometry_when_only_template_extent_changes(self):
        observations={name:Observation(name,0,n,name,(0,0,20,20),(0,0),
            'probability','reference',.9,1008,template_id=0,
            pose=translation(),reliable=True) for n,name in enumerate(('a','b'))}
        template=Template(0,(0,0,20,20),references=[
            Reference(name,n,'reference',1.) for n,name in enumerate(('a','b'))])
        from uav.template_mask.geometry import from_components
        fits=[[Match(translation(),.7,18,.8)],
              [Match(from_components((0,0,.01,0),(0,0)),.7,18,.8)]]
        with patch.object(self.matcher,'reference_matches',side_effect=fits*2):
            first=self.matcher.template_matches(template,observations,'target')
            template.box=(0,0,1000,1000)
            second=self.matcher.template_matches(template,observations,'target')
        self.assertEqual(len(first),1)
        self.assertEqual(len(second),2)

    def test_multiple_occurrences_share_neighbours_and_preserve_both_fits(self):
        rng=np.random.default_rng(187)
        xy=rng.uniform(0,60,(24,2))
        desc=rng.random((24,128),dtype=np.float32)
        reference=dict(xy=xy,desc=desc,size=np.ones(24))
        target=dict(xy=np.concatenate((xy+[20,30],xy+[160,90])),
                    desc=np.concatenate((desc,desc)),size=np.ones(48))
        with (patch('uav.template_mask.features.ratio_matches',wraps=ratio_matches) as neighbours,
              patch('uav.template_mask.features.np.ptp',wraps=np.ptp) as extent):
            fits=self.matcher.fit(reference,target,max_locations=4)
            self.assertEqual(neighbours.call_count,1)
            # One full-reference extent plus one inlier extent per occurrence.
            self.assertEqual(extent.call_count,1)
        self.assertEqual(len(fits),2)
        matrices=sorted((m.matrix for m in fits),key=lambda m:m[0,2])
        np.testing.assert_allclose(matrices[0],translation(20,30),atol=1e-5)
        np.testing.assert_allclose(matrices[1],translation(160,90),atol=1e-5)

    def test_array_neighbours_preserve_opencv_query_order_ties_and_ratio(self):
        rng=np.random.default_rng(317)
        for queries,references in ((8,8),(40,300),(600,3000)):
            a=rng.random((queries,128),dtype=np.float32)
            b=rng.random((references,128),dtype=np.float32)
            for tied in (False,True):
                if tied:b[1:4]=b[0]
                pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(a,b,k=2)
                good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.76*p[1].distance]
                ri,ti=ratio_matches(b,a)
                np.testing.assert_array_equal(ri,[p.trainIdx for p in good])
                np.testing.assert_array_equal(ti,[p.queryIdx for p in good])

    def test_ratio_threshold_retains_python_float_precision(self):
        b=np.zeros((8,128),np.float32);a=np.zeros((3,128),np.float32)
        values=np.array([np.nextafter(np.float32(.76),np.float32(0)),
                         np.float32(.76),np.nextafter(np.float32(.76),np.float32(1))])
        distances=np.column_stack((values,np.ones(3,np.float32)))
        indices=np.tile([0,1],(3,1)).astype(np.int32)
        with patch('cv2.batchDistance',return_value=(distances,indices)):
            _,queries=ratio_matches(b,a)
        np.testing.assert_array_equal(queries,[i for i,v in enumerate(values) if float(v)<.76])

    def test_empty_queries_or_one_reference_cannot_pass_two_neighbour_ratio(self):
        for n,m in ((0,8),(8,0),(8,1)):
            reference=np.zeros((m,128),np.float32)
            target=np.zeros((n,128),np.float32)
            with patch('cv2.batchDistance',side_effect=AssertionError('no two neighbours')):
                ri,ti=ratio_matches(reference,target)
            self.assertEqual((len(ri),len(ti)),(0,0))

    def test_fit_reuse_includes_failures_and_every_numeric_input(self):
        cache=MotionFitCache()
        times=np.arange(16,dtype=float);values=times*.4
        with patch('uav.template_mask.motion.fit_pose_component',wraps=fit_pose_component) as fit:
            first=cache.fit(times,values,.7,0)
            reused=cache.fit(times.copy(),values.copy(),.7,0)
            self.assertIsNot(reused,first)
            np.testing.assert_array_equal(reused.coefficient,first.coefficient)
            first.coefficient[0]=999
            np.testing.assert_array_equal(cache.fit(times,values,.7,0).coefficient,reused.coefficient)
            self.assertEqual(fit.call_count,1)
            for t,v,s,j,straight in [(times+1,values,.7,0,False),
                (times,values+1,.7,0,False),(times,values,.8,0,False),
                (times,values,.7,1,False),(times,values,.7,0,True)]:
                cache.fit(t,v,s,j,straight_only=straight)
            self.assertEqual(fit.call_count,6)
            self.assertIsNone(cache.fit(times[:3],values[:3],.7,0))
            self.assertIsNone(cache.fit(times[:3],values[:3],.7,0))
            self.assertEqual(fit.call_count,7)

    def test_reused_curve_fits_preserve_predictions_and_source_exclusion(self):
        rows=[Pose(f,translation(30+.4*f,40+.01*f*f),.7,'measured',f,(),True) for f in range(40)]
        box=(0,0,100,40);anchor=(50,20)
        plain=MotionPrior(rows,box,anchor)
        cache=MotionFitCache()
        a=MotionPrior(rows,box,anchor,fit_cache=cache)
        b=MotionPrior(rows,box,anchor,fit_cache=cache)
        self.assertGreater(cache.hits,0)
        self.assertEqual(a.report(),plain.report());self.assertEqual(b.report(),plain.report())
        for f in (-2,10,20,45):
            for reused in (a,b):
                p,q=plain.predict(f),reused.predict(f)
                np.testing.assert_array_equal(p[0],q[0]);self.assertEqual(p[1:],q[1:])
        matrix=rows[20].matrix
        self.assertEqual(plain.continuity(20,matrix,.7,20),
                         a.continuity(20,matrix,.7,20,fit_cache=cache))

    def test_boundary_shortlist_checks_whole_fit_before_local_windows(self):
        from uav.template_mask.gaps import boundary_prior
        rows=[Pose(f,translation(30+.01*f*f,40+.02*f*f),.7,'measured',f,(),True) for f in range(64)]
        template=Template(0,(0,0,100,40),anchor=(50,20))
        constructions=[]
        def construct(*args,**kwargs):
            constructions.append(kwargs.get('whole_only',False))
            return MotionPrior(*args,**kwargs)
        with patch('uav.template_mask.gaps.MotionPrior',side_effect=construct):
            selected=boundary_prior(rows,template,'left',fit_cache=MotionFitCache())
        self.assertTrue(all(constructions[:-1]));self.assertFalse(constructions[-1])
        # Preserve the original longest validated straight-prefix selection.
        expected=None
        for length in (64,32,16,8):
            prior=MotionPrior(rows[-length:],template.box,template.anchor)
            whole=prior.whole
            if whole and all(m is not None for m in whole[0]) and all(m.mode in ('static','linear') for m in whole[0][:2]):
                expected=prior;break
        if expected is None:expected=MotionPrior(rows,template.box,template.anchor)
        self.assertEqual(selected.report(),expected.report())

    def test_motion_rebuild_requires_changed_measurements_or_template_coordinates(self):
        from uav.template_mask.engine import TemplateEngine
        from uav.template_mask.policy import WarningLog
        e=TemplateEngine(TemplateConfig(sift_backend='cpu'),self.cache,None,WarningLog(Path(self.temp.name)/'w.jsonl'),(120,200))
        p=np.ones((30,80),np.float32)*.9
        obs=e.observe(0,0,np.zeros((120,200,3),np.uint8),
            Candidate(p,(20,30),(20,30,100,60),.9,'synthetic'),1008,'global')
        owner=e._provisional(obs,'new_template')
        for f in range(24):owner.poses[f]=Pose(f,translation(20+.3*f,30),.7,'measured',f,(),True)
        e._motion(owner);prior=owner.motion;revision=e.motion_revision
        owner.poses[24]=Pose(24,translation(30,30),5.,'predicted')
        e._motion(owner)
        self.assertIs(owner.motion,prior);self.assertEqual(e.motion_revision,revision+1)
        owner.poses[10].matrix[0,2]+=.1
        e._motion(owner);self.assertIsNot(owner.motion,prior)
        prior=owner.motion
        e.bank.templates[0].box=(0,0,90,30)
        e._motion(owner);self.assertIsNot(owner.motion,prior)


if __name__ == '__main__':
    unittest.main()
