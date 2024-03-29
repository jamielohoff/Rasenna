#!/usr/bin/env bash

# python2 and python3 should both work, depends on your own environment.
g++ -O3 -fPIC -w -shared -std=c++11 -I ../ -I pybind11-stable/include `python3.7-config --cflags --ldflags --libs` PersistencePython.cpp Debugging.cpp PersistenceIO.cpp -o ../../rasenna/src/rasenna/criteria/PersistencePython.so
