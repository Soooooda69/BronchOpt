import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='BronchOpt',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'third_party/lietorch/include'), 
                osp.join(ROOT, 'third_party/eigen-3.4.0')],
            sources=[
                'third_party/lietorch/src/lietorch.cpp', 
                'third_party/lietorch/src/lietorch_gpu.cu',
                'third_party/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })