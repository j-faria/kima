from setuptools import setup
from setuptools.extension import Extension, Library
from glob import glob


# Build kima shared library
kima_srcs = glob("kima/distributions/*.cpp")
kima_srcs += [
    "kima/kepler.cpp",
    "kima/AMDstability.cpp",
    "kima/Data.cpp",
    "kima/ConditionalPrior.cpp",
    "kima/RVmodel.cpp",
    "kima/GPmodel.cpp",
    "kima/RVFWHMmodel.cpp",
    "kima/BDmodel.cpp",
]

kima_ext = Library(
    name="kima.kima",
    sources=kima_srcs,
    include_dirs=[
        "kima/vendor/cpp-loadtxt/src", "kima/vendor/DNest4/code",
        "kima/vendor/eigen"
    ],
    extra_compile_args=['-w', "-pthread", "-std=c++17", "-O3"],
    libraries=["dnest4"],  # see below
)

# Build DNest4 static library
dnest4_srcs = glob("kima/vendor/DNest4/code/*.cpp")
dnest4_srcs += glob("kima/vendor/DNest4/code/Distributions/*.cpp")
dnest4_srcs += glob("kima/vendor/DNest4/code/RJObject/ConditionalPriors/*.cpp")

dnest4_cflags = [
    '-std=c++17', '-O3', '-march=native', '-Wall', '-Wextra', '-pedantic',
    '-DNDEBUG'
]

dnest4_lib = Library(
    name='kima.dnest4',
    sources=dnest4_srcs,
    cflags=dnest4_cflags,
)


# Build pykepler dynamic library (with pybind11)
from pybind11.setup_helpers import Pybind11Extension

pykepler_ext = Pybind11Extension("pykima.pykepler",
                                 ['kima/kepler.cpp', 'kima/pykepler.cpp'])

setup(
    # libraries=[
    #     ("dnest4", {
    #         "sources": dnest4_srcs,
    #         "cflags": dnest4_cflags,
    #     }),
    # ],
    ext_modules=[dnest4_lib, kima_ext, pykepler_ext],
)
