"""Run-local CUDA L2 search; original-image evidence remains on the CPU."""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np


@dataclass
class _Descriptors:
    values: object
    left: object = None
    right: object = None

    @property
    def bytes(self):
        tensors = (self.values,) if self.left is None else (self.left,self.right)
        return sum(t.numel()*t.element_size() for t in tensors)


class CudaNeighbours:
    def __init__(self, *, device='cuda', cache_bytes=128 << 20, max_pairs=2 << 20):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.stream = None
        self.cache_limit = cache_bytes
        self.max_pairs = max_pairs
        if cache_bytes < 0 or max_pairs < 1:
            raise ValueError('invalid descriptor cache or distance block size')
        self.tensors = OrderedDict()
        self.bytes = self.peak_bytes = self.calls = self.uploads = self.hits = 0

    def _tensor(self, array):
        key = id(array)
        if key in self.tensors:
            self.hits += 1
            self.tensors.move_to_end(key)
            return self.tensors[key][1]
        # Copy once, including read-only evidence arrays. Holding the source
        # prevents id reuse. Mutable public inputs are deliberately not cached.
        torch = self.torch
        tensor = torch.tensor(array,dtype=torch.float32,device=self.device)
        data = _Descriptors(tensor)
        if self.device.type == 'cuda':
            # These are the exact augmented operands used by torch.cdist's
            # Euclidean GEMM path. Reuse norms/padding without changing distance
            # arithmetic, dtype or the query block shape.
            norm = tensor.pow(2).sum(-1,keepdim=True)
            left = torch.cat((-2*tensor,norm,torch.ones_like(norm)),-1)
            right = torch.cat((tensor,torch.ones_like(norm),norm),-1)
            data = _Descriptors(right[:,:array.shape[1]],left,right)
        self.uploads += 1
        size = data.bytes
        if not array.flags.writeable and size <= self.cache_limit:
            while self.bytes+size > self.cache_limit:
                _,(_,previous) = self.tensors.popitem(last=False)
                self.bytes -= previous.bytes
            self.tensors[key] = (array,data)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes,self.bytes)
        return data

    def _indices(self, reference, target, *, out=None, prepared_target=None):
        """Fixed-size device result avoids a GPU nonzero/shape synchronization."""
        torch = self.torch
        a = self._tensor(target) if prepared_target is None else prepared_target
        b = self._tensor(reference)
        mode = ('donot_use_mm_for_euclid_dist' if torch.backends.cuda.matmul.allow_tf32
                else 'use_mm_for_euclid_dist')
        rows = max(1,min(512,self.max_pairs//len(reference)))
        output = (torch.empty(len(target),dtype=torch.int64,device=self.device)
                  if out is None else out)
        for start in range(0,len(target),rows):
            squared = mode == 'use_mm_for_euclid_dist' and a.left is not None
            if squared:
                distances = a.left[start:start+rows] @ b.right.T
            else:
                distances = torch.cdist(a.values[start:start+rows],b.values,compute_mode=mode)
            if self.device.type == 'cuda':
                from .nearest_kernel import nearest_indices
                nearest_indices(distances,out=output[start:start+len(distances)],squared=squared)
            else:
                best = distances.topk(2,largest=False,sorted=True)
                scores = best.values.double()
                output[start:start+len(distances)] = torch.where(
                    scores[:,0] < .76*scores[:,1],best.indices[:,0],-1)
        return output

    def many(self, references, target):
        """Independent searches in input order, with one fixed-size CPU transfer.

        Keep the original distance formula, query blocks and ratio test. Share
        the readback without mixing descriptors between independent references.
        """
        self.calls += len(references)
        empty = lambda: (np.empty(0,dtype=int),np.empty(0,dtype=int))
        if not len(target):
            return [empty() for _ in references]
        active = [i for i,ref in enumerate(references) if len(ref)>=2]
        if not active:
            return [empty() for _ in references]
        torch = self.torch
        if self.device.type == 'cuda' and self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device,priority=-1)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with (torch.cuda.stream(self.stream) if self.stream is not None else nullcontext(),
              torch.inference_mode(),torch.autocast(self.device.type,enabled=False)):
            # Write each independent search directly into its final readback
            # buffer. No per-block concatenations or per-reference stack copy.
            output = torch.empty((len(active),len(target)),dtype=torch.int64,device=self.device)
            prepared_target = self._tensor(target)
            for row,i in enumerate(active):
                self._indices(references[i],target,out=output[row],prepared_target=prepared_target)
            indices = output.cpu().numpy()
        results = [empty() for _ in references]
        for i,row in zip(active,indices):
            queries = np.flatnonzero(row>=0)
            results[i] = row[queries],queries
        return results

    def __call__(self, reference, target):
        return self.many([reference],target)[0]

    def report(self):
        return dict(backend='cuda', calls=self.calls, uploads=self.uploads,
            cache_hits=self.hits, cache_bytes=self.bytes, peak_cache_bytes=self.peak_bytes,
            max_distance_pairs=self.max_pairs,
            stream_priority=None if self.stream is None else self.stream.priority)

    def close(self):
        self.tensors.clear()
        self.bytes = 0
