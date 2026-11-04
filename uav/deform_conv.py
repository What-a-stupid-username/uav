"""FP16 torchvision-derived im2col launched on the current PyTorch CUDA stream."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import glob
import torch


def _check(code, operation):
    if code:
        raise RuntimeError(f'{operation} failed with CUDA status {code}')


@lru_cache(maxsize=8)
def _kernel(device):
    # Compile/load before graph capture. NVRTC and driver already ship in CUDA runtime.
    root=Path(torch.__file__).parent.parent/'nvidia'
    libraries=glob.glob(str(root/'cuda_nvrtc/lib/libnvrtc.so*'))
    if not libraries:
        raise RuntimeError('NVRTC is required; set PROPAINTER_STREAM_DEFORM=0 to use torchvision')
    nvrtc=C.CDLL(libraries[0])
    driver=C.CDLL('libcuda.so.1')
    source=(Path(__file__).parent/'csrc/deform_im2col.cu').read_bytes()
    program=C.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(C.byref(program),source,b'deform.cu',0,None,None),'nvrtcCreateProgram')
    major,minor=torch.cuda.get_device_capability(device)
    options=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++17',
             f'--include-path={root}/cuda_runtime/include'.encode()]
    try:
        status=nvrtc.nvrtcCompileProgram(program,len(options),(C.c_char_p*len(options))(*options))
        if status:
            size=C.c_size_t();nvrtc.nvrtcGetProgramLogSize(program,C.byref(size))
            log=C.create_string_buffer(size.value);nvrtc.nvrtcGetProgramLog(program,log)
            raise RuntimeError(log.value.decode())
        size=C.c_size_t();_check(nvrtc.nvrtcGetPTXSize(program,C.byref(size)),'nvrtcGetPTXSize')
        ptx=C.create_string_buffer(size.value);_check(nvrtc.nvrtcGetPTX(program,ptx),'nvrtcGetPTX')
    finally:
        nvrtc.nvrtcDestroyProgram(C.byref(program))
    module=C.c_void_p();function=C.c_void_p()
    _check(driver.cuModuleLoadData(C.byref(module),ptx),'cuModuleLoadData')
    _check(driver.cuModuleGetFunction(C.byref(function),module,b'deformable_im2col_kernel'),'cuModuleGetFunction')
    driver.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
    return driver,module,function


def deform_conv2d(input,offset,weight,bias,padding=(1,1),mask=None):
    padding=(padding,padding) if isinstance(padding,int) else padding
    # ProPainter's inference alignment: batch=1, stride=dilation=1, groups=1.
    if (torch.is_grad_enabled() or input.dtype!=torch.float16 or input.shape[0]!=1 or
            weight.shape[1]!=input.shape[1] or mask is None or bias is None or
            not input.is_cuda):
        from torchvision.ops import deform_conv2d as original
        return original(input,offset,weight,bias,padding=padding,mask=mask)
    with torch.cuda.device(input.device):
        driver,module,function=_kernel(input.device.index)
        x=input.contiguous();offset=offset.contiguous();mask=mask.contiguous()
        _,channels,height,width=x.shape;out_h,out_w=offset.shape[-2:];kh,kw=weight.shape[-2:]
        groups=offset.shape[1]//(2*kh*kw)
        if groups<1 or channels%groups or offset.shape[1]!=groups*2*kh*kw:
            raise ValueError('invalid deformable offset groups')
        if mask.shape!=(1,groups*kh*kw,out_h,out_w):
            raise ValueError('invalid deformable mask shape')
        if any(t.device!=x.device or t.dtype!=x.dtype for t in (offset,mask,weight,bias)):
            raise ValueError('deformable convolution tensors must share device and dtype')
        if channels*kh*kw*out_h*out_w>=2**31:
            raise ValueError('FP16 stream kernel requires 32-bit indexing')
        columns=torch.empty((channels*kh*kw,out_h*out_w),device=x.device,dtype=x.dtype)
        count=channels*out_h*out_w
        values=[C.c_int(count),C.c_void_p(x.data_ptr()),C.c_void_p(offset.data_ptr()),C.c_void_p(mask.data_ptr())]
        values += [C.c_int(v) for v in (height,width,kh,kw,*padding,1,1,1,1,1,channels,groups,out_h,out_w)]
        values += [C.c_bool(True),C.c_void_p(columns.data_ptr())]
        pointers=(C.c_void_p*len(values))(*[C.cast(C.byref(v),C.c_void_p) for v in values])
        _check(driver.cuLaunchKernel(function,(count+511)//512,1,1,512,1,1,0,
               C.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None),'cuLaunchKernel')
        # Match torchvision's addmm output layout and bias arithmetic.
        output=torch.empty((weight.shape[0],out_h*out_w),device=x.device,dtype=x.dtype)
        # beta=0 overwrites every element: no separate zero-fill kernel needed.
        output.addmm_(weight.flatten(1),columns,beta=0)
        return output.view(1,weight.shape[0],out_h,out_w)+bias.view(1,-1,1,1)
