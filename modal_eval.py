"""MOVED: the eval daemon now lives at MAEMMBench/modal_eval.py.

This shim keeps `MODAL_PROFILE=... modal deploy modal_eval.py` (and `modal run modal_eval.py`)
working from the repo root — Modal discovers the re-exported `app` just fine.
"""
from MAEMMBench.modal_eval import app, daemon, main  # noqa: F401
