"""Build CUDA SIFT into the project's persistent, ignored temporary directory."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def _find_nvcc():
    """Only a cache miss needs the compiler supplied by the runtime image."""
    from torch.utils.cpp_extension import CUDA_HOME
    candidates = [shutil.which('nvcc')]
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME)/'bin/nvcc')
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError('Building CUDA SIFT requires nvcc; use the UAV CUDA devel image and mount tmp/ writable')


@lru_cache(maxsize=1)
def library_path():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA SIFT requires a CUDA GPU')
    source=Path(__file__).parent/'csrc/cudasift'
    directory=Path(__file__).resolve().parents[1]/'tmp/extensions/uav_cudasift'
    major,minor=torch.cuda.get_device_capability()
    return _cached_library(source, directory, f'sm_{major}{minor}', torch.version.cuda)


def _cached_library(source, directory, architecture, cuda_version):
    """Reuse across processes/containers without locating or running nvcc."""
    directory.mkdir(parents=True,exist_ok=True)
    library=directory/'libuav_sift.so'
    from filelock import FileLock
    with FileLock(str(directory/'build.lock')):
        # Exclude absolute paths: changing the mount or compiler location does
        # not change this C ABI library. CUDA version/architecture still do.
        sources=[source/n for n in ('uav_bridge.cu','cudaSiftH.cu','cudaImage.cu')]
        options=['-std=c++17','-O3',f'-arch={architecture}','--shared',
                 '-Xcompiler','-fPIC','--cudart=shared']
        digest=hashlib.sha256(json.dumps(dict(options=options, cuda=cuda_version,
            sources=[p.name for p in sources])).encode())
        for p in sorted(source.iterdir()):
            if p.is_file():
                digest.update(p.name.encode()+b'\0')
                digest.update(p.read_bytes())
        fingerprint=digest.hexdigest();stamp=directory/'source.sha256'
        if not library.exists() or not stamp.exists() or stamp.read_text()!=fingerprint:
            temporary=directory/'libuav_sift.build.so'
            temporary_stamp=directory/'source.sha256.tmp'
            try:
                command=[_find_nvcc(),*options,'-I'+str(source),
                         *map(str,sources),'-o',str(temporary)]
                subprocess.run(command,check=True)
                os.replace(temporary,library)
                temporary_stamp.write_text(fingerprint)
                os.replace(temporary_stamp,stamp)
            finally:
                temporary.unlink(missing_ok=True)
                temporary_stamp.unlink(missing_ok=True)
    return library


if __name__=='__main__':print(library_path())
