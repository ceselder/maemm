#!/usr/bin/env python3
"""Spawn / poll a function of a DEPLOYED Modal app without keeping a client attached (a `modal run` dies with the
terminal; a spawned call on a deployed app does not).

    python scripts/modal_call.py spawn <app> <function> ['{"kw": 1}']   -> prints the FunctionCall id
    python scripts/modal_call.py poll <call_id> [--wait S]              -> result JSON / "PENDING" / the exception
Requires MODAL_PROFILE (e.g. safety-sahan) in the environment.
"""
import argparse
import json
import sys

import modal


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spawn"); s.add_argument("app"); s.add_argument("function"); s.add_argument("kwargs", nargs="?", default="{}")
    p = sub.add_parser("poll"); p.add_argument("call_id"); p.add_argument("--wait", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "spawn":
        fc = modal.Function.from_name(a.app, a.function).spawn(**json.loads(a.kwargs))
        print(fc.object_id)
        return
    fc = modal.FunctionCall.from_id(a.call_id)
    try:
        res = fc.get(timeout=a.wait)
    except TimeoutError:
        print("PENDING")
        sys.exit(3)
    print(json.dumps(res, default=str, indent=1))


if __name__ == "__main__":
    main()
