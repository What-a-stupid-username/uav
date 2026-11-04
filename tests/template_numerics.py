"""Exact reductions for the short finite vectors used by trajectory fits."""
import unittest
import numpy as np
from uav.template_mask.numerics import least_squares,median,quantile


class NumericsTests(unittest.TestCase):
    def test_finite_reductions_equal_numpy(self):
        rng=np.random.default_rng(831)
        for length in range(1,180):
            for values in (rng.normal(size=length),np.zeros(length),np.arange(length,dtype=float)):
                self.assertEqual(median(values),np.median(values))
                for q in (0.,.23,.5,.8,.9,1.):
                    self.assertEqual(quantile(values,q),np.quantile(values,q))

    def test_nan_fallback_preserves_numpy(self):
        self.assertTrue(np.isnan(median([1.,np.nan])))
        self.assertTrue(np.isnan(quantile([1.,np.nan],.8)))

    def test_supported_dtypes_extremes_and_nonvectors_follow_numpy(self):
        for dtype in (np.float32,np.float64,np.int32,np.int64):
            for length in (1,2,3,8,17):
                values=np.arange(length,dtype=dtype)-3
                self.assertEqual(median(values),np.median(values))
                for q in (0,.23,.8,.9,1):
                    self.assertEqual(quantile(values,q),np.quantile(values,q))
        matrix=np.arange(12.).reshape(3,4)
        self.assertEqual(median(matrix),np.median(matrix))
        self.assertEqual(quantile(matrix,.8),np.quantile(matrix,.8))

    def test_compiled_robust_fit_preserves_predictions(self):
        from uav.template_mask.motion import robust_fit
        rng=np.random.default_rng(881)
        def original(design,values):
            coefficient=np.linalg.lstsq(design,values,rcond=None)[0]
            for _ in range(5):
                residual=np.abs(values-design@coefficient)
                scale=max(1e-8,float(np.median(residual))*1.4826)
                weight=np.minimum(1.,1.5*scale/np.maximum(residual,1e-9))
                coefficient=np.linalg.lstsq(design*weight[:,None],values*weight,rcond=None)[0]
            return coefficient
        for count in (8,16,32,100):
            t=np.linspace(0,1,count)
            for degree in (0,1,2,5):
                design=np.polynomial.legendre.legvander(t,degree)
                clean=design@rng.normal(size=degree+1)
                for values in (clean,clean+rng.normal(size=count),np.zeros(count)):
                    actual, expected = robust_fit(design,values),original(design,values)
                    np.testing.assert_allclose(design@actual,design@expected,rtol=2e-11,atol=2e-11)

    def test_system_reuse_is_independent_of_tolerance_and_owns_results(self):
        from uav.template_mask.motion import robust_fit
        from uav.template_mask.native_fit import fit_and_score
        from unittest.mock import patch
        design=np.column_stack((np.ones(16),np.arange(16)))
        values=np.arange(16)*.4;systems={}
        expected=robust_fit(design,values)
        with patch('uav.template_mask.motion.fit_and_score',wraps=fit_and_score) as solve:
            a=robust_fit(design,values,systems=systems)
            calls=solve.call_count
            a[0]=999
            b=robust_fit(design.copy(),values.copy(),systems=systems)
            np.testing.assert_array_equal(b,expected)
            self.assertEqual(solve.call_count,calls)
            robust_fit(design,values+1,systems=systems)
            self.assertGreater(solve.call_count,calls)

    def test_design_reuse_keeps_basis_and_time_coordinates_separate(self):
        from dataclasses import replace
        from uav.template_mask.motion import ComponentModel,design_matrix
        times=np.arange(16,dtype=float);cache={}
        model=ComponentModel('linear',np.zeros(2),0.,0.,15.)
        a=design_matrix(model,times,cache)
        self.assertIs(design_matrix(model,times.copy(),cache),a)
        for alternative in (replace(model,origin=2.),replace(model,span=30.),
                            replace(model,mode='accelerating')):
            expected=alternative.design(times)
            actual=design_matrix(alternative,times,cache)
            np.testing.assert_array_equal(actual,expected)
            self.assertIsNot(actual,a)

    def test_sampling_layout_reuses_exact_times_without_sharing_inputs(self):
        from uav.template_mask.motion import _sampling_layout
        times=np.r_[np.arange(17.),np.arange(18.,39.)]
        cache={}
        layout=_sampling_layout(times,cache)
        self.assertIs(_sampling_layout(times.copy(),cache),layout)
        np.testing.assert_array_equal(layout.steps,np.diff(times))
        for actual,expected in zip(layout.blocks,np.array_split(np.arange(len(times)),4)):
            np.testing.assert_array_equal(actual,expected)
        self.assertTrue(layout.dense)
        self.assertIsNot(_sampling_layout(times+1,cache),layout)
        times[0]=-999
        self.assertEqual(layout.times[0],0)

    def test_direct_lapack_matches_numpy_with_rank_deficiency_and_owned_inputs(self):
        rng=np.random.default_rng(571)
        for rows in (6,8,16,32,64,128,360):
            for columns in (1,2,3,6,11):
                if columns>rows:
                    continue
                for kind in ('random','deficient','ill_conditioned','zero'):
                    design=rng.normal(size=(rows,columns))
                    if kind=='deficient' and columns>1:design[:,-1]=design[:,0]
                    if kind=='ill_conditioned':design*=np.geomspace(1,1e-16,columns)
                    if kind=='zero':design[:]=0
                    values=rng.normal(size=rows)
                    original,original_values=design.copy(),values.copy()
                    expected=np.linalg.lstsq(design,values,rcond=None)[0]
                    for a,b in ((design,values),(np.asfortranarray(design),values),
                                (design[::-1][::-1],values[::-1][::-1])):
                        actual=least_squares(a,b)
                        if columns==1:
                            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
                        else:
                            np.testing.assert_array_equal(actual,expected)
                    np.testing.assert_array_equal(design,original)
                    np.testing.assert_array_equal(values,original_values)

    def test_periodic_preparation_reuses_exact_basis_and_system_keys(self):
        from uav.template_mask.motion import _sampling_layout,ComponentModel,_system_key
        times=np.arange(96,dtype=float);values=np.sin(times/7)
        layout=_sampling_layout(times,{})
        prepared=layout.periodic_designs
        self.assertIs(layout.periodic_designs,prepared)
        self.assertEqual(len(prepared),100)
        for frequency,design,prefix in prepared:
            model=ComponentModel('periodic',np.empty(0),0.,0.,95.,frequency,0)
            np.testing.assert_array_equal(design,model.design(times))
            self.assertEqual(prefix+(values.dtype.str,values.tobytes()),_system_key(design,values))

    def test_rejected_periodic_candidates_do_not_construct_model_objects(self):
        from unittest.mock import patch
        from uav.template_mask import motion
        times=np.arange(96,dtype=float);values=np.sin(times/7)
        sampling={};layout=motion._sampling_layout(times,sampling)
        _=layout.periodic_designs
        with (patch.object(motion,'fit_batch',return_value=(np.zeros((100,3)),np.full(100,10.))),
              patch.object(motion,'ComponentModel',wraps=motion.ComponentModel) as model):
            result=motion.fit_component(times,values,.1,modes={'periodic'},sampling=sampling)
        self.assertIsNone(result)
        model.assert_not_called()

    def test_periodic_equal_residuals_keep_first_frequency(self):
        from uav.template_mask.motion import fit_component
        times=np.arange(96,dtype=float)
        # Restricted periodic search validates every frequency on constant data;
        # the residual floor makes all accepted scores tie exactly.
        result=fit_component(times,np.zeros(96),1.,modes={'periodic'})
        self.assertIsNotNone(result)
        self.assertEqual(result.cycles,2.5)
        self.assertEqual(result.residual,.25)

    def test_holdout_reads_cached_coefficients_without_copying_or_mutating_them(self):
        from unittest.mock import patch
        from uav.template_mask.motion import _FitDiagnostics,_fit_system,robust_fit
        design=np.column_stack((np.ones(32),np.arange(32,dtype=float)))
        values=np.arange(32)*.3;held=np.arange(8,16)
        systems={};keep=np.ones(32,bool);keep[held]=False
        cached=_fit_system(design[keep],values[keep],systems)
        before=cached.coefficient.copy()
        expected=float(quantile(np.abs(values[held]-design[held]@robust_fit(
            design[keep],values[keep],systems=systems)),.8))
        trial=_FitDiagnostics(np.zeros(2))
        with patch('uav.template_mask.motion.robust_fit',side_effect=AssertionError('unneeded owned copy')):
            self.assertEqual(trial.held_error(design,values,held,systems),expected)
        np.testing.assert_array_equal(cached.coefficient,before)

    def test_shared_holdouts_preserve_rows_keys_and_read_only_masks(self):
        from uav.template_mask.motion import _SamplingLayout,_FitDiagnostics
        times=np.arange(96,dtype=float)
        layout=_SamplingLayout(times)
        for ordinary,prepared in ((layout.blocks,layout.validation_blocks),
                                  (layout.spline_blocks,layout.spline_validation_blocks)):
            for held,partition in zip(ordinary,prepared):
                np.testing.assert_array_equal(partition.held,held)
                self.assertEqual(partition.key,tuple(held))
                expected=np.ones(len(times),bool);expected[held]=False
                np.testing.assert_array_equal(partition.keep,expected)
                self.assertFalse(partition.held.flags.writeable)
                self.assertFalse(partition.keep.flags.writeable)
        self.assertIs(layout.validation_blocks,layout.validation_blocks)
        self.assertIs(layout.spline_validation_blocks,layout.spline_validation_blocks)
        design=np.column_stack((np.ones(96),times,times*times))
        values=.3*times+.01*times*times+np.sin(times)*.1
        for partition in layout.validation_blocks:
            for rank in (False,True):
                a=_FitDiagnostics(np.zeros(3));b=_FitDiagnostics(np.zeros(3))
                expected=a.held_error(design,values,partition.held,{},check_rank=rank)
                actual=b.held_error(design,values,partition,{},check_rank=rank)
                self.assertEqual(actual,expected)
                self.assertEqual(a.held_errors,b.held_errors)

    def test_general_lstsq_forms_keep_numpy_semantics(self):
        for design,values in ((np.ones((2,4)),np.arange(2.)),
                (np.eye(4,dtype=np.float32),np.arange(4,dtype=np.float32)),
                (np.eye(4,dtype=complex),np.arange(4)+1j),
                (np.ones((4,0)),np.arange(4.)),
                (np.eye(4),np.ones((4,2)))):
            np.testing.assert_array_equal(least_squares(design,values),
                np.linalg.lstsq(design,values,rcond=None)[0])

    def test_scalar_solver_preserves_extreme_norms(self):
        for scale in (1e-170,1e-160,1.,1e160):
            design=np.full((16,1),scale)
            values=np.arange(16,dtype=float)*scale
            np.testing.assert_allclose(least_squares(design,values),
                np.linalg.lstsq(design,values,rcond=None)[0],rtol=2e-13,atol=2e-13)

    def test_sampling_reuse_preserves_models_and_acceptance_across_tolerances(self):
        from uav.template_mask.motion import fit_pose_component
        times=np.arange(96,dtype=float)
        sampling={}
        for values,component in ((times*.3,0),(.01*times**2,1),
                (.02*times,2),(.05*np.sin(times/5),3)):
            for tolerance in (.01,.3,1.):
                a=fit_pose_component(times,values,tolerance,component)
                b=fit_pose_component(times,values,tolerance,component,sampling=sampling)
                self.assertEqual(a is None,b is None)
                if a is not None:
                    self.assertEqual((a.mode,a.residual,a.cycles),(b.mode,b.residual,b.cycles))
                    np.testing.assert_array_equal(a.coefficient,b.coefficient)


if __name__=='__main__':unittest.main()
