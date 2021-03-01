# -*- coding: utf-8 -*-
"""
    Setup file for rasenna.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

module = Extension('PersistencePython,
                   sources = ['cPers/cPers/PersistencePython.cpp', 
                   'cPers/cPers/Debugging.cpp', 'cPers/cPers/PersistenceIO.cpp',],
                   include_dirs = ['cPers/blitz'],
                   extra_compile_args=['-fPIC', '-O3', '-w', '-shared', '-std=c++11', '-I cPers/cPers/pybind11-stable/include `python3.7-config --cflags --ldflags --libs`'])



if __name__ == "__main__":
    setup(use_pyscaffold=True,
            name="Rasenna",
            url="https://github.com/elmo0082/Rasenna",
            author="Jamie Grieser",
            install_requires=[
                "numpy",
                "scipy",
                "speedrun",
                "h5py",
                "torch",
                "matplotlib", 
                "tensorboardx",
            ],)
