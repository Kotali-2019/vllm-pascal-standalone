#!/usr/bin/env bash
#
# run_collect_env.sh
#
# Prints the vLLM "collect-env" report for the MAINLINE Pascal build
#   v0.28.1rc1.dev158+g04f9bc118.d20260903
# (NOT the stale editable rankaiyx-vllm-pascal fork).
#
# Usage:
#   bash /root/MyRepo/vllm-mainline/run_collect_env.sh
#
# Why each step is required:
#   1. source /root/cuda_pascal.sh
#        -> puts CUDA 12.8 (nvcc + cudart) on PATH / LD_LIBRARY_PATH so the
#           built _C*.so extensions load. (CUDA 13 dropped sm_61.)
#   2. cd /root/MyRepo/vllm-mainline
#        -> `import vllm` resolves via sys.path[0] = CWD, so we MUST sit in
#           the mainline tree for the mainline package to be the one imported.
#   3. use the vllm-mainline-test interpreter
#        -> has torch 2.11 + triton via the cuda12Pascal .pth.
#   4. run `python -m vllm.collect_env`
#        -> the SAME report `vllm collect-env` prints (the CLI subcommand is
#           literally `from vllm.collect_env import main; main()`), invoked
#           directly to AVOID the broken full-CLI import chain (see below).
#
# DO NOT use these (verified broken / misleading in this env):
#   * `vllm collect-env`  /  `python -m vllm.entrypoints.cli.main collect-env`
#       -> the CLI main() eagerly imports the launch/serve/sagemaker/xgrammar
#          stack, which is not installed here -> ImportError (exit 120).
#   * `importlib.metadata.version("vllm")`  or  `pip show vllm`
#       -> reports the STALE FORK (0.11.0rc2...) because the cuda12Pascal
#          site-packages (with the old editable dist-info) is on sys.path via
#          a .pth. Use `vllm.__version__` (below) for the true version.
#
set -euo pipefail

# 1) CUDA 12.8 toolchain (Pascal / sm_61 capable)
source /root/cuda_pascal.sh

# 2) Run from the mainline tree so `import vllm` == the mainline package
cd /root/MyRepo/vllm-mainline

PY=/root/vllm-mainline-test/bin/python

# 3) Provenance check: prove the imported package is the 0.28.1 mainline tree.
#    Uses vllm.__version__ (reliable), NOT importlib.metadata (stale here).
"$PY" - <<'PYEOF'
import vllm
print("== provenance check ==")
print("  vllm.__file__    =", vllm.__file__)
print("  vllm.__version__ =", vllm.__version__)
assert "vllm-mainline" in vllm.__file__, f"WRONG tree: {vllm.__file__}"
assert vllm.__version__.startswith("0.28.1"), f"unexpected version: {vllm.__version__}"
print("  -> OK: mainline 0.28.1 tree confirmed")
PYEOF

# 4) The collect-env report (identical output to `vllm collect-env`)
"$PY" -m vllm.collect_env
