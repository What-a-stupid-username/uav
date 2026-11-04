"""CUDA search semantics using synthetic descriptors, with CPU execution too."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.features import SiftMatcher, ratio_matches
from uav.template_mask.neighbours import CudaNeighbours


class NeighbourTests(unittest.TestCase):
    device = 'cpu'

    def search(self, **kwargs):
        result = CudaNeighbours(device=self.device, **kwargs)
        self.addCleanup(result.close)
        return result

    def arrays(self):
        rng = np.random.default_rng(317)
        reference = rng.random((180,128), dtype=np.float32)
        reference /= np.linalg.norm(reference,axis=1)[:,None]
        target = np.concatenate((reference[::3]+rng.normal(0,.0005,(60,128)),
            rng.random((31,128),dtype=np.float32))).astype(np.float32)
        return reference, target

    def test_chunking_and_query_order_match_cpu_for_clear_matches(self):
        reference, target = self.arrays()
        expected = ratio_matches(reference,target)
        for pairs in (1000, 1 << 20):
            result = self.search(max_pairs=pairs)(reference,target)
            for actual,wanted in zip(result,expected):
                np.testing.assert_array_equal(actual,wanted)

    def test_grouped_readback_preserves_separate_searches_and_query_order(self):
        reference,target=self.arrays()
        refs=[reference,reference[::2].copy(),np.zeros((1,128),np.float32),reference[::-1].copy()]
        separate=self.search()
        expected=[separate(ref,target) for ref in refs]
        grouped=self.search()
        actual=grouped.many(refs,target)
        for pair,wanted in zip(actual,expected):
            for a,b in zip(pair,wanted):np.testing.assert_array_equal(a,b)
        self.assertEqual(grouped.calls,len(refs))
        self.assertEqual(grouped.many([],target),[])
        self.assertEqual(len(grouped.many(refs,target[:0])),len(refs))

    def test_grouped_template_fit_matches_sequential_reference_fit(self):
        from uav.template_mask.records import Observation,Reference,Template
        rng=np.random.default_rng(92)
        reference,target=self.arrays()
        xy=rng.uniform(0,100,(len(reference),2))
        tx=rng.uniform(0,100,(len(target),2))
        tx[:60]=xy[::3]+[17.,23.]
        caches=[EvidenceCache(),EvidenceCache()]
        matchers=[]
        for cache in caches:
            self.addCleanup(cache.close)
            matcher=SiftMatcher(cache,TemplateConfig(sift_backend='cpu'))
            matcher.gpu_neighbours=self.search()
            matchers.append(matcher)
            cache.put('target',xy=tx,size=np.ones(len(tx)),desc=target)
            for n in range(2):cache.put('ref'+str(n),xy=xy,size=np.ones(len(xy)),desc=reference)
        observations={}
        refs=[]
        for n in range(2):
            oid='o'+str(n)
            obs=Observation(oid,n,n,str(n),(0,0,100,100),(0,0),'p','ref'+str(n),.9,1008)
            obs.template_id=0;obs.pose=np.eye(3);obs.reliable=True
            observations[oid]=obs
            refs.append(Reference(oid,n,obs.feature_key,1.))
        template=Template(0,(0,0,100,100),(50,50));template.references=refs
        with patch.object(matchers[0],'_prepare_neighbours',return_value={}):
            expected=matchers[0].template_matches(template,observations,'target')
        with patch.object(matchers[1].gpu_neighbours,'many',wraps=matchers[1].gpu_neighbours.many) as calls:
            actual=matchers[1].template_matches(template,observations,'target')
            self.assertEqual(calls.call_count,1)
            self.assertEqual(len(calls.call_args.args[0]),2)
            matchers[1].template_matches(template,observations,'target')
            self.assertEqual(calls.call_count,1)
        self.assertEqual(len(actual),len(expected));self.assertTrue(actual)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a.matrix,b.matrix)
            self.assertEqual((a.uncertainty,a.inliers,a.coverage,a.references),
                             (b.uncertainty,b.inliers,b.coverage,b.references))

    def test_reference_preparation_shares_search_across_template_queries(self):
        from types import SimpleNamespace
        from uav.template_mask.records import Reference
        reference,target=self.arrays()
        cache=EvidenceCache()
        self.addCleanup(cache.close)
        matcher=SiftMatcher(cache,TemplateConfig(sift_backend='cpu'))
        matcher.gpu_neighbours=self.search()
        cache.put('target',xy=np.zeros((len(target),2)),size=np.ones(len(target)),desc=target)
        refs=[]
        observations={}
        for i in range(3):
            key=f'ref{i}'
            cache.put(key,xy=np.zeros((len(reference),2)),size=np.ones(len(reference)),desc=reference)
            refs.append(Reference(key,i,key,1.))
            observations[key]=SimpleNamespace(reliable=True,ambiguous=False)
        with patch.object(matcher.gpu_neighbours,'many',wraps=matcher.gpu_neighbours.many) as search:
            matcher.prepare_references(refs,observations,'target')
            self.assertEqual(search.call_count,1)
            self.assertEqual(len(search.call_args.args[0]),3)
            for ref in refs:
                matcher._prepare_neighbours([ref],observations,'target',1,None)
            self.assertEqual(search.call_count,1)


    def test_empty_inputs_and_duplicate_neighbours_are_rejected(self):
        search = self.search()
        for n,m in ((0,8),(8,0),(8,1),(8,8)):
            ri,ti = search(np.zeros((m,128),np.float32),np.zeros((n,128),np.float32))
            self.assertEqual((len(ri),len(ti)),(0,0))

    def test_cache_reuses_only_immutable_arrays_and_close_releases_them(self):
        reference,target = self.arrays()
        reference.flags.writeable = target.flags.writeable = False
        search = self.search()
        search(reference,target);search(reference,target)
        self.assertEqual((search.uploads, search.hits),(2,2))
        self.assertGreater(search.bytes,0)
        search.close()
        self.assertEqual(search.bytes,0)
        self.assertFalse(search.tensors)
        mutable = reference.copy()
        search(mutable,target);mutable[:]=0
        ri,_ = search(mutable,target)
        self.assertEqual(len(ri),0)
        self.assertEqual(search.uploads,5)

    def test_descriptor_cache_evicts_without_changing_results(self):
        reference,target = self.arrays()
        reference.flags.writeable = target.flags.writeable = False
        search = self.search(cache_bytes=reference.nbytes)
        expected=ratio_matches(reference,target)
        for _ in range(3):
            actual=search(reference,target)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
            self.assertLessEqual(search.bytes,reference.nbytes)

    def test_autocast_and_tf32_settings_are_restored(self):
        reference,target = self.arrays()
        previous=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            with torch.autocast(self.device, dtype=torch.bfloat16):
                actual=self.search()(reference,target)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            for a,b in zip(actual,ratio_matches(reference,target)):
                np.testing.assert_array_equal(a,b)
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous

    def test_backend_selection_cache_isolation_and_release(self):
        config=TemplateConfig(sift_backend='cpu')
        self.assertEqual(config.resolved_match_backend,'cpu')
        with self.assertRaises(ValueError):TemplateConfig(match_backend='other')
        cache=EvidenceCache()
        self.addCleanup(cache.close)
        cpu=SiftMatcher(cache,config)
        with patch('uav.template_mask.neighbours.CudaNeighbours') as gpu:
            cuda=SiftMatcher(cache,TemplateConfig(sift_backend='cpu',match_backend='cuda'))
            self.assertNotEqual(cpu._match_key('pair'),cuda._match_key('pair'))
            cuda.close()
            gpu.return_value.close.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime required')
class CudaNeighbourTests(NeighbourTests):
    device = 'cuda'

    def test_squared_distances_match_separate_clamp_sqrt_exactly(self):
        from uav.template_mask.nearest_kernel import nearest_indices
        generator=torch.Generator(device='cuda').manual_seed(513)
        for count in (2,31,180,800,3000):
            squared=torch.rand((129,count),generator=generator,device='cuda')+.01
            squared[0]=0
            squared[1]=1
            squared[2,:2]=torch.tensor([-.01,.01],device='cuda')
            for row,x in zip((3,4,5),(np.nextafter(np.float32(.76),np.float32(0)),
                                     np.float32(.76),np.nextafter(np.float32(.76),np.float32(1)))):
                squared[row]=4.
                squared[row,:2]=torch.tensor([x*x,1.],device='cuda')
            squared[6]=4.
            squared[6,:2]=torch.tensor([1.,np.nextafter(np.float32(1),np.float32(2))],device='cuda')
            distances=squared.clamp_min(0).sqrt()
            self.assertTrue(torch.equal(nearest_indices(squared,squared=True),
                                        nearest_indices(distances)),count)

    def test_fused_reduction_writes_supplied_output_slice_only(self):
        from uav.template_mask.nearest_kernel import nearest_indices
        distances=torch.tensor([[.1,1.,2.],[1.,1.,2.],[.5,2.,3.]],device='cuda')
        storage=torch.full((11,),99,dtype=torch.int64,device='cuda')
        result=nearest_indices(distances,out=storage[4:7])
        self.assertEqual(result.data_ptr(),storage[4:7].data_ptr())
        self.assertEqual(storage.cpu().tolist(),[99]*4+[0,-1,0]+[99]*4)

    def test_prepared_distance_operands_match_cdist_and_track_memory(self):
        rng=np.random.default_rng(917)
        search=self.search()
        for n,m in ((17,31),(513,180),(90,800),(512,3000)):
            reference=rng.random((m,128),dtype=np.float32)
            target=rng.random((n,128),dtype=np.float32)
            reference.flags.writeable=target.flags.writeable=False
            with torch.cuda.stream(search.stream) if search.stream is not None else torch.cuda.device(0):
                a,b=search._tensor(target),search._tensor(reference)
                for start in range(0,n,512):
                    expected=torch.cdist(a.values[start:start+512].contiguous(),b.values.contiguous(),
                        compute_mode='use_mm_for_euclid_dist')
                    actual=(a.left[start:start+512]@b.right.T).clamp_min_(0).sqrt_()
                    self.assertTrue(torch.equal(actual,expected),(n,m,start))
            self.assertEqual(search.bytes,sum(data.bytes for _,data in search.tensors.values()))

    def test_fused_reduction_matches_torch_including_threshold_and_ties(self):
        from uav.template_mask.nearest_kernel import nearest_indices
        rng=np.random.default_rng(914)
        for count in (2,3,31,180,800,3000,8192):
            values=rng.uniform(.1,3,(65,count)).astype(np.float32)
            values[0]=1. # equal nearest distances must fail the ratio
            values[1]=0.
            values[2,:2]=[0.,.1]
            for row,x in zip((3,4,5),(np.nextafter(np.float32(.76),np.float32(0)),
                                    np.float32(.76),np.nextafter(np.float32(.76),np.float32(1)))):
                values[row]=2.;values[row,:2]=[x,1.]
            distances=torch.from_numpy(values).cuda()
            best=distances.topk(2,largest=False,sorted=True)
            scores=best.values.double()
            expected=torch.where(scores[:,0]<.76*scores[:,1],best.indices[:,0],-1)
            actual=nearest_indices(distances)
            self.assertTrue(torch.equal(actual,expected),(count,actual,expected))



if __name__ == '__main__':
    unittest.main()
