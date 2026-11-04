"""Buffered CUDA SIFT; the extended geometry API is explicitly opt-in."""
import ctypes
import cv2
import numpy as np
from .sift_build import library_path


class CudaSift:
    def __init__(self,threshold=1.0,scale_up=True,library=None,*,nfeatures=512,geometry=False,stream=None,priority=None):
        if type(nfeatures) is not int or not 1 <= nfeatures <= 8192:
            raise ValueError('nfeatures must be between 1 and 8192')
        self.lib=ctypes.CDLL(library or str(library_path()))
        self.lib.uav_sift_create.restype=ctypes.c_void_p
        self.lib.uav_sift_destroy.argtypes=[ctypes.c_void_p]
        floats=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        uints=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        if priority is not None:
            if stream is not None:
                raise ValueError('specify a stream or priority, not both')
            import torch
            stream = torch.cuda.Stream(priority=priority)
            stream.wait_stream(torch.cuda.current_stream())
        if stream is not None and not geometry:
            raise ValueError('an explicit stream requires the geometry API')
        self.stream = stream
        self.extract=(self.lib.uav_sift_extract_geometry_stream if stream is not None else
                      self.lib.uav_sift_extract_geometry if geometry else self.lib.uav_sift_extract)
        self.extract.argtypes=[ctypes.c_void_p,floats,uints,ctypes.c_int,ctypes.c_int,
            ctypes.c_float,ctypes.c_int,floats,floats,ctypes.c_int]+([floats] if geometry else [])+([ctypes.c_void_p] if stream is not None else [])
        self.extract.restype=ctypes.c_int
        self.handle=self.lib.uav_sift_create()
        self.threshold,self.scale_up,self.geometry=threshold,scale_up,geometry
        self.xy=np.empty((nfeatures,2),np.float32)
        self.descriptors=np.empty((nfeatures,128),np.float32)
        self.metadata=np.empty((nfeatures,2),np.float32) if geometry else None

    def close(self):
        if self.handle:
            self.lib.uav_sift_destroy(self.handle)
            self.handle=None

    def _extract(self,image,mask):
        if not self.handle:
            raise RuntimeError('extractor closed')
        if image.ndim != 2 or not image.size:
            raise ValueError('CUDA SIFT requires a nonempty grayscale image')
        if mask is not None and mask.shape != image.shape:
            raise ValueError('SIFT mask must match grayscale image')
        source=np.ascontiguousarray(image,dtype=np.float32)
        support=np.ones(image.shape,np.uint8) if mask is None else np.ascontiguousarray(mask,dtype=np.uint8)
        h,w=image.shape
        args=[self.handle,source,support,w,h,self.threshold,int(self.scale_up),
              self.xy,self.descriptors,len(self.xy)]
        n=self.extract(*args,*([self.metadata] if self.geometry else []),
                       *([self.stream.cuda_stream] if self.stream is not None else []))
        if not 0 <= n <= len(self.xy):
            raise RuntimeError('invalid CUDA SIFT result size')
        return n

    def detect_arrays(self,image,mask):
        """Owned arrays for tracking, without an intermediate KeyPoint list."""
        n=self._extract(image,mask)
        size=self.metadata[:n,0].copy() if self.geometry else np.ones(n,np.float32)
        return self.xy[:n].copy(),size,self.descriptors[:n].copy()

    def detectAndCompute(self,image,mask):
        n=self._extract(image,mask)
        if self.geometry:
            keypoints=[cv2.KeyPoint(float(x),float(y),float(size),float(angle))
                for (x,y),(size,angle) in zip(self.xy[:n],self.metadata[:n])]
        else:
            keypoints=[cv2.KeyPoint(float(x),float(y),1) for x,y in self.xy[:n]]
        return keypoints,self.descriptors[:n].copy() if n else None
