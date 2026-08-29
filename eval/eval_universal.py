"""MOVED: the canonical module now lives at MAEMMBench/eval_universal.py.

This shim keeps old imports (`from eval_universal import run_eval` with eval/ on sys.path, or
`import eval.eval_universal`) and the old CLI path (`python eval/eval_universal.py ...`) working.
Requires the repo root on PYTHONPATH (same requirement as `import mxf` always had).
"""
from MAEMMBench.eval_universal import *                                    # noqa: F401,F403
from MAEMMBench.eval_universal import (                                    # noqa: F401
    _reencode, _gen_batches, _load_heldout_pool, _build_heldout_eval_sets, main)

if __name__ == "__main__":
    main()
