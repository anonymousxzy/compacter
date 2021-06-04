import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
#export TORCH_CUDA_ARCH_LIST="3.5;3.7;6.1;7.0;7.5;8.6+PTX"
ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'hadamard_cuda', [
            'hadamard_cuda.cpp',
            'hadamard_cuda_kernel.cu'
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ext_modules.append(extension)

setup(
    name='hadamard_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
