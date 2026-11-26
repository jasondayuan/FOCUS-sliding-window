"""Run exps/gsm8k_eval.py with the NVTX instrumentation installed.

Usage: python run_prof.py <repo root> <gsm8k_eval.py args...>
"""
import os
import sys

if __name__ == '__main__':
    # guarded: the engine's env check spawns worker processes that re-import this file
    repo = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.join(repo, 'exps'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import prof_patch  # noqa: F401  (installs the wrappers)
    import gsm8k_eval

    sys.argv = ['gsm8k_eval.py'] + sys.argv[2:]
    gsm8k_eval.main()
