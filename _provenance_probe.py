"""PROVENANCE PROBE — NOT a benchmark.

Imports vllm and prints interpreter/path metadata only. It does NOT:
  - load any model
  - run prefill or decode
  - launch any model kernel
  - measure any timing
It exists solely to capture sys.executable / sys.path / vllm.__file__ /
vllm.__version__ / cwd for the exact interpreter+cwd used by the functional run.
"""
import os
import sys

import vllm

print("=== PROVENANCE (import-only, no model, no kernels, no timing) ===")
print("1. sys.executable        =", sys.executable)
print("2. os.getcwd()           =", os.getcwd())
print("3. vllm.__file__         =", vllm.__file__)
print("4. vllm.__version__      =", getattr(vllm, "__version__", "<no __version__ attr>"))
print("5. list(vllm.__path__)   =", list(vllm.__path__))
print("6. sys.path (in order):")
for p in sys.path:
    print("       ", p)

is_mainline = "vllm-mainline" in vllm.__file__
is_fork = "rankaiyx-vllm-pascal" in vllm.__file__
print("=== VERDICT ===")
print("   vllm.__file__ is the MAINLINE tree  :", is_mainline)
print("   vllm.__file__ is the FORK (rankaiyx) :", is_fork)
