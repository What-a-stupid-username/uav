"""Native numeric edge cases, thread reuse and persistent build cache in RAM."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.native_fit import (_cached_library,_kernel,_pool,fit_and_score,
                                         fit_batch,robust_coefficients)


def reference_fit(design,values):
    coefficient=np.linalg.lstsq(design,values,rcond=None)[0]
    previous=np.ones(len(values))
    for _ in range(5):
        residual=np.abs(values-design@coefficient)
        scale=max(1e-8,float(np.median(residual))*1.4826)
        weight=np.minimum(1.,1.5*scale/np.maximum(residual,1e-9))
        if np.array_equal(weight,previous):break
        coefficient=np.linalg.lstsq(design*weight[:,None],values*weight,rcond=None)[0]
        previous=weight
    return coefficient


class NativeFitTests(unittest.TestCase):
    def test_distribution_fast_decisions_match_svd_and_boundary_defers(self):
        from uav.template_mask.native_fit import _module
        rng=np.random.default_rng(83)
        for count in (4,8,40,180):
            for aspect in (0.,1e-8,.01,1.,100.):
                points=rng.normal(size=(count,2))*[1,aspect]+[100,200]
                sigma=np.linalg.svd(points-points.mean(0),compute_uv=False)[-1]
                for threshold in (.7,float(sigma),float(sigma)*.5,float(sigma)*2):
                    result=_module().distributed(points,threshold)
                    if result>=0:self.assertEqual(bool(result),sigma>=threshold)
                    if threshold==sigma:self.assertEqual(result,-1)
    def test_localization_matches_float32_and_float64_medians(self):
        from uav.template_mask.native_fit import _module
        from uav.template_mask.numerics import median
        rng=np.random.default_rng(41)
        for dtype in (np.float32,np.float64):
            for n in (4,5,8,31,180):
                sizes=rng.uniform(.5,100,n).astype(dtype)
                expected=max(.7,float(median(sizes))*.12)
                self.assertEqual(_module().localization(sizes),expected)
    def test_match_statistics_match_numpy_residual_and_coverage(self):
        from uav.template_mask.native_fit import _module
        from uav.template_mask.geometry import points
        from uav.template_mask.numerics import quantile
        rng=np.random.default_rng(71)
        for count in (4,8,40,180):
            src=rng.normal(size=(count,2))
            matrix=rng.normal(size=(3,3))
            dst=points(matrix,src)+rng.normal(0,.3,(count,2))
            extent=np.array([4.,6.])
            expected=(float(quantile(np.linalg.norm(points(matrix,src)-dst,axis=1),.9)),
                float(np.prod(np.minimum(1.,np.ptp(src,axis=0)/np.maximum(1.,extent)))))
            np.testing.assert_allclose(_module().match_statistics(matrix,src,dst,*extent),
                expected,rtol=2e-15,atol=1e-12)
    def test_degenerate_and_extreme_systems_match_lapack_predictions(self):
        rng=np.random.default_rng(519)
        for rows,columns in ((8,1),(16,3),(32,6),(80,11)):
            base=rng.normal(size=(rows,columns))
            for kind in ('normal','duplicate','zero','near_rank_loss','tiny','huge'):
                a=base.copy();b=rng.normal(size=rows)
                if kind=='duplicate' and columns>1:a[:,-1]=a[:,0]
                if kind=='zero':a[:]=0
                if kind=='near_rank_loss':a*=np.geomspace(1,1e-16,columns)
                if kind=='tiny':a*=1e-170;b*=1e-170
                if kind=='huge':a*=1e160;b*=1e160
                original_a,original_b=a.copy(),b.copy()
                actual,score=fit_and_score(a,b)
                expected=reference_fit(a,b)
                magnitude=max(1.,float(np.abs(b).max()))
                np.testing.assert_allclose(a@actual/magnitude,a@expected/magnitude,rtol=2e-9,atol=2e-9)
                expected_score=np.quantile(np.abs(b-a@actual),.8)
                self.assertAlmostEqual(score/magnitude,expected_score/magnitude,places=10)
                np.testing.assert_array_equal(a,original_a)
                np.testing.assert_array_equal(b,original_b)

    def test_batch_pool_matches_single_fits_and_reuses_threads(self):
        rng=np.random.default_rng(791)
        matrices=rng.normal(size=(40,64,3));values=rng.normal(size=64)
        expected=np.array([np.r_[c,s] for c,s in (fit_and_score(a,values) for a in matrices)])
        for workers in (1,2,4,8):
            for _ in range(2):
                coefficients,scores=fit_batch(matrices,values,workers=workers)
                np.testing.assert_array_equal(np.column_stack((coefficients,scores)),expected)
        self.assertIs(_pool(4),_pool(4))
        self.assertEqual(_pool(4)._max_workers,4)

    def test_shared_kernel_is_reentrant_with_thread_local_workspaces(self):
        rng=np.random.default_rng(341)
        cases=[(rng.normal(size=(n,3)),rng.normal(size=n)) for n in (16,80,32,120)*4]
        expected=[robust_coefficients(a,b) for a,b in cases]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(lambda pair:robust_coefficients(*pair),cases))
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_native_buffer_validation_and_empty_batch(self):
        for a,b in ((np.ones((2,4)),np.ones(2)),(np.ones((8,2)),np.ones(7))):
            with self.assertRaises(ValueError):fit_and_score(a,b)
        c,s=fit_batch(np.empty((0,8,3)),np.ones(8))
        self.assertEqual((c.shape,s.shape),((0,3),(0,)))
        function,capsule=_kernel()
        with self.assertRaises(TypeError):
            function(np.ones((8,2),np.float32),np.ones(8),np.ones(3),8,2,capsule,100,100)
        with self.assertRaises(ValueError):
            function(np.ones((8,2)),np.ones(8),np.ones(2),8,2,capsule,100,100)


class NativeBuildTests(unittest.TestCase):
    def test_cache_hit_needs_no_compiler_and_failure_keeps_previous_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'fit.cpp';source.write_text('// synthetic source')
            cache=Path(folder)/'cache'
            def compile(command,*,check):Path(command[-1]).write_bytes(b'synthetic library')
            with patch('uav.template_mask.native_fit.shutil.which',return_value='/image/g++'),\
                 patch('uav.template_mask.native_fit.subprocess.run',side_effect=compile) as build:
                library=_cached_library(source,cache)
                self.assertEqual(build.call_count,1)
            stamp=(cache/'source.sha256').read_text()
            with patch('uav.template_mask.native_fit.shutil.which',side_effect=AssertionError('cache hit')),\
                 patch('uav.template_mask.native_fit.subprocess.run',side_effect=AssertionError('cache hit')):
                self.assertEqual(_cached_library(source,cache),library)
            source.write_text('// changed source')
            with patch('uav.template_mask.native_fit.shutil.which',return_value='/image/g++'),\
                 patch('uav.template_mask.native_fit.subprocess.run',side_effect=subprocess.CalledProcessError(1,'g++')):
                with self.assertRaises(subprocess.CalledProcessError):_cached_library(source,cache)
            self.assertEqual(library.read_bytes(),b'synthetic library')
            self.assertEqual((cache/'source.sha256').read_text(),stamp)
            self.assertFalse((cache/'libmotion_fit.build.so').exists())


if __name__=='__main__':unittest.main()
