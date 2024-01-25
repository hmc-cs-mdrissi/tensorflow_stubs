from typing import TypedDict

class _BuildInfo(TypedDict):
    cuda_version: str
    cudnn_version: str
    is_cuda_build: bool
    is_rocm_build: bool
    is_tensorrt_build: bool
    msvcp_dll_names: list[str]
    nvcuda_dll_name: str
    cudart_dll_name: str
    cudnn_dll_name: str

def get_build_info() -> _BuildInfo: ...
