"""Inline EXTRA evals for rl/rl.py — a standalone module (rl.py is untouched; the trainer wires
these three calls in next to `inline_eval`). Everything a held-out checkpoint used to need an
offline eval script for now runs INSIDE the trainer, per checkpoint:

  GPU stage   run_extra_evals_gpu()   ALL ranks, target << 1 min on 4-8 ranks
      * N rollouts (default 4, T=1, testbed min/max new tokens) per testbed feature (64 held-out
        SAE features, /data/eval_autointerp/testbed_v2.json) with the CURRENT policy: this rank's
        vLLM engine + the LoRA it published this step (== the weights of ckpt_step), vllm_lens
        steering built by the trainer's own _steer_vec / _marker_norm — request construction
        mirrors inline_eval; features sharded i % world == rank.
      * snippet-locality metrics on the CLEAN base: per-token SAE-feature activation profile of
        every rollout on the shared read path (BOS sink prepended + skipped, right-pad masked,
        10x-median norm filter, ReLU encode of the target feature) -> eval/snippet_locality
        .profile_metrics (win3/win5 share, peak share, Gini, tokens >= 50% peak) + peak position.
      * clean-base SAE scoring of the adversarial texts the judge wrote for the PREVIOUS
        checkpoint (rank 0 broadcasts them, every rank scores its shard, results are gathered and
        pushed to the results queue under the checkpoint they belong to).
      * every rank ALWAYS joins the broadcast + all_gather (errors travel as data) -> a failure
        on one rank cannot deadlock DDP.
  Judge stage  launch_judge_stage()   rank 0 only, background thread, never blocks training
      * autointerp detection AUC: eval/autointerp_detection.py's judge prompt + Mann-Whitney AUC;
        description = the first N in {1, 4} rollouts; tests = the testbed's positives vs its
        random / embedding-NN hard negatives (N=1 here is the first T=1 sample — the offline
        eval used a greedy rollout for N=1).
      * WildChat fire-prediction AUC: same judge prompt; tests = 4 firing + 4 non-firing
        64-token WildChat-1M windows per feature scored with OUR L42 SAE — a one-time bank at
        /data/eval_wildchat/windows.json (eval/wildchat_bank.py via modal_wildchat_bank.py).
      * adversarial confirmation: the judge summarizes the rollouts into a description and writes
        4 texts fitting it; a NAIVE corpus-only description (judge over the feature's top-8 corpus
        max-act windows) and ITS 4 texts are computed ONCE and cached (they do not depend on the
        checkpoint). Texts are scored standalone on the clean base next checkpoint; HOLDS iff
        mean(true-fit) > mean(naive-fit) AND mean(true-fit) > 0.25 x corpus peak.
      * results -> poll_judge_results() -> [(ckpt_step, {"extra/...": float})] which the trainer
        logs as wandb.log({**m, "ckpt_step": ckpt_step}) — WITHOUT step= (they arrive late;
        wandb.define_metric("extra/*", step_metric="ckpt_step") makes ckpt_step the x-axis).

JUDGE. Project rule: Sonnet 5. Provider from env: ANTHROPIC_API_KEY (+ ANTHROPIC_WORKSPACE_ID
header when set; needs `pip install anthropic`) -> Anthropic native `claude-sonnet-5`; else
OPENROUTER_API_KEY -> OpenRouter `anthropic/claude-sonnet-5` (plain HTTPS, no extra deps).
EXTRA_EVAL_JUDGE_MODEL / EXTRA_EVAL_JUDGE_PROVIDER override. Thinking is DISABLED on every call
(Sonnet 5 thinks by default and silently eats an 8-token max_tokens); no sampling params are sent
(Sonnet 5 rejects them). Bounded thread pool, exponential backoff with jitter on 429 / 5xx /
overloaded / network errors, hard per-stage deadline, per-stage cost from the API's usage.

ENV KNOBS (all optional):
  EXTRA_EVAL_TESTBED (=/data/eval_autointerp/testbed_v2.json)  EXTRA_EVAL_WILDCHAT (=/data/eval_wildchat/windows.json)
  EXTRA_EVAL_SAE (= --eval-sae)  EXTRA_EVAL_NAIVE_CACHE (=/data/eval_autointerp/naive_desc_cache.json)
  EXTRA_EVAL_N_ROLLOUTS (=4)  EXTRA_EVAL_N_LIST (=1,4)  EXTRA_EVAL_NEG_KINDS (=random,embnn)
  EXTRA_EVAL_N_TESTS (=10 positives + 10 per negative kind)  EXTRA_EVAL_ADV_K (=4 texts per arm)
  EXTRA_EVAL_JUDGE_EVERY (=1: judge every checkpoint the GPU stage runs)  EXTRA_EVAL_JUDGE_CONCURRENCY (=16)
  EXTRA_EVAL_JUDGE_TIMEOUT_S (=1800 hard stage deadline)  EXTRA_EVAL_DISABLE (=comma list of
  autointerp,wildchat,adversarial to switch judge sub-evals off)

Smoke (no GPU; fakes the rollouts with the testbed's stored samples, runs the judge stage live):
    OPENROUTER_API_KEY=... python eval/inline_extra_evals.py --testbed eval/out/testbed_v2.json \
        --wildchat /path/windows.json --n-features 2 --n-tests 2
"""
import json
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# ---- eval modules: repo layout (eval/ siblings) or the Modal layout
# (/pmx/eval, /pmx/MAEMMBench). Appended, so the trainer's own sys.path entries win. ----
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, ".."),
           "/pmx/eval", "/pmx/MAEMMBench", "/pmx/helpers"):
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
try:
    import autointerp_detection as AD      # JUDGE_SYSTEM prompt + _auc (Mann-Whitney, ties 0.5)
    import snippet_locality as SL          # profile_metrics / crop_to_best_window / GEN_SEED
    _IMPORT_ERR = None
except Exception as _e:  # noqa — surfaced by prepare_extra_eval_assets (never raise at import)
    AD = SL = None
    _IMPORT_ERR = _e

# ----------------------------------------------------------------------------------------------
# config (env)
# ----------------------------------------------------------------------------------------------
DEFAULT_TESTBED = "/data/eval_autointerp/testbed_v2.json"
DEFAULT_WILDCHAT = "/data/eval_wildchat/windows.json"
DEFAULT_NAIVE_CACHE = "/data/eval_autointerp/naive_desc_cache.json"
DEFAULT_SAE = "/data/sae/ae.pt"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NORM_FILTER_MULT = 10.0         # shared clean-base read path (eval_universal / snippet_locality)
MAX_TEXT_TOKENS = 95            # shared clean-base read path truncation
HOLDS_FRAC = 0.25               # adversarial: mean(true-fit) must exceed this x corpus peak
CROP_LEN = 32                   # locality length control: best contiguous 32-token window (= max-act window)
MAX_CONCURRENT_STAGES = 2       # judge stages allowed in flight at once (later ones are skipped + logged)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _env_list(name, default):
    raw = os.environ.get(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


N_ROLLOUTS = _env_int("EXTRA_EVAL_N_ROLLOUTS", 4)
N_LIST = sorted(int(x) for x in _env_list("EXTRA_EVAL_N_LIST", "1,4"))
NEG_KINDS = _env_list("EXTRA_EVAL_NEG_KINDS", "random,embnn")
N_TESTS = _env_int("EXTRA_EVAL_N_TESTS", 10)
ADV_K = _env_int("EXTRA_EVAL_ADV_K", 4)
JUDGE_EVERY = max(1, _env_int("EXTRA_EVAL_JUDGE_EVERY", 1))
JUDGE_CONCURRENCY = max(1, _env_int("EXTRA_EVAL_JUDGE_CONCURRENCY", 16))
JUDGE_TIMEOUT_S = float(os.environ.get("EXTRA_EVAL_JUDGE_TIMEOUT_S", 1800))
DISABLED = set(_env_list("EXTRA_EVAL_DISABLE", ""))

# USD per 1M tokens (input, output) — used only when the API does not return a cost itself
# (Anthropic native); OpenRouter returns usage.cost directly.
PRICE_PER_M = {"claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0), "claude-opus-5": (5.0, 25.0),
               "claude-opus-4-7": (5.0, 25.0), "claude-haiku-4-5": (1.0, 5.0)}

# ----------------------------------------------------------------------------------------------
# judge prompts
# ----------------------------------------------------------------------------------------------
DESC_SYSTEM = (
    "You are an expert interpretability researcher characterizing a feature (a 'neuron') inside a "
    "language model from text it responds to. Be concrete and specific about the pattern, topic, "
    "or linguistic form the feature detects; if the texts share a narrow trigger (a word, a format, "
    "a domain), name it. Output ONLY the description: one or two sentences, no preamble.")
DESC_USER_CORPUS = (
    "Below are the {n} text windows from a large web corpus on which the feature activates most "
    "strongly (strongest first). The feature typically fires on a specific span inside each window. "
    "Describe what the feature detects.\n\n{block}")
DESC_USER_GENERATED = (
    "Below are {n} short texts that a model was steered to generate so as to activate the feature as "
    "strongly as possible. They may be noisy or repetitive; infer the underlying concept. Describe what "
    "the feature detects.\n\n{block}")
ADV_SYSTEM = (
    "You write test texts for interpretability experiments. Given a description of a language-model "
    "feature, write natural, diverse text snippets that would strongly exhibit the described "
    "concept/pattern. Output ONLY a JSON array of strings, nothing else.")
ADV_USER = (
    "Feature description: {desc}\n\nWrite {k} diverse text snippets (each 30-60 words, plain prose or "
    "the natural format of the concept) that would activate this feature as strongly as possible. Vary "
    "the angle, register and wording across snippets. Output ONLY a JSON array of {k} strings.")


def _detect_prompt(desc_snippets, text):
    """EXACT user-prompt template of eval/autointerp_detection.py cmd_judge (byte-identical format)."""
    block = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(desc_snippets))
    return (f"DESCRIPTION SNIPPETS:\n{block}\n\nTEST SNIPPET:\n<<<\n{text}\n>>>"
            "\n\nProbability (integer 0-100) that the feature activates on the test snippet:")


def _snippet_block(snips):
    return "\n".join(f"{j + 1}. <<<{t}>>>" for j, t in enumerate(snips))


# ----------------------------------------------------------------------------------------------
# pure helpers (CPU, unit-tested)
# ----------------------------------------------------------------------------------------------
def _shard_rows(n, rank, world):
    return list(range(rank, n, world))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_score(txt):
    """First integer in the judge reply, clamped to [0, 100]; None if unparseable."""
    m = re.search(r"\d+", txt or "")
    return None if not m else min(100, max(0, int(m.group())))


def parse_json_list(txt, k):
    """First JSON array of strings in txt (code fences tolerated) -> up to k non-empty strings ([] on failure)."""
    s = (txt or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        arr = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()][:k]


def auc(pos, neg):
    """Mann-Whitney AUC with ties = 0.5 (== autointerp_detection._auc); None when a side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    if AD is not None:
        return float(AD._auc(pos, neg))
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def locality_rows_from_profiles(profiles, fire, crop_len=CROP_LEN):
    """profiles: list of 1-D kept-token ReLU activation profiles (one per text) -> per-text rows:
    n_tokens, peak, fired (peak > fire), metrics (snippet_locality.profile_metrics), win5 share of the
    best contiguous crop_len window (length control), relative peak position, peak-in-last-5 flag."""
    rows = []
    for p in profiles:
        p = np.asarray(p, dtype=np.float64)
        n = len(p)
        peak = float(p.max()) if n else 0.0
        arg = int(p.argmax()) if n else 0
        rows.append({"n_tokens": n, "peak": peak, "fired": bool(peak > fire),
                     "metrics": SL.profile_metrics(p) if n else {m: None for m in SL.METRICS},
                     "crop_win5": (SL.profile_metrics(SL.crop_to_best_window(p, crop_len))["win5_share"]
                                   if n else None),
                     "peak_pos": (arg / (n - 1)) if n > 1 else 0.0,
                     "peak_in_last5": bool(n and arg >= n - 5)})
    return rows


def aggregate_locality(rows_by_feature, corpus_peak):
    """{feature: [row, ...]} -> {"extra/locality/<metric>": float}. Locality metrics are aggregated over
    FIRING texts only (as in snippet_locality); firing fractions over everything. NaN when no text fires."""
    all_rows = [r for rows in rows_by_feature.values() for r in rows]
    fired = [r for r in all_rows if r["fired"]]
    out = {}

    def mean_or_nan(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else float("nan")
    for m in SL.METRICS:
        out[f"extra/locality/{m}"] = mean_or_nan([r["metrics"][m] for r in fired])
    out["extra/locality/win5_share_crop32"] = mean_or_nan([r["crop_win5"] for r in fired])
    out["extra/locality/peak_pos_mean"] = mean_or_nan([r["peak_pos"] for r in fired])
    out["extra/locality/peak_pos_median"] = (float(np.median([r["peak_pos"] for r in fired]))
                                             if fired else float("nan"))
    out["extra/locality/peak_in_last5_frac"] = mean_or_nan([float(r["peak_in_last5"]) for r in fired])
    out["extra/locality/peak_act_fired_mean"] = mean_or_nan([r["peak"] for r in fired])
    out["extra/locality/fire_frac"] = mean_or_nan([float(r["fired"]) for r in all_rows])
    out["extra/locality/feat_fire_frac"] = mean_or_nan([float(any(r["fired"] for r in rows))
                                                        for rows in rows_by_feature.values()])
    out["extra/locality/n_tokens_mean"] = mean_or_nan([r["n_tokens"] for r in all_rows])
    out["extra/locality/peak_norm_best_mean"] = mean_or_nan(
        [max(r["peak"] for r in rows) / max(float(corpus_peak.get(f, 0.0)), 1e-6)
         for f, rows in rows_by_feature.items() if rows])
    out["extra/locality/n_features"] = float(len(rows_by_feature))
    out["extra/locality/n_texts"] = float(len(all_rows))
    return out


def adversarial_metrics(acts_true, acts_naive, corpus_peak, frac=HOLDS_FRAC, fire=1.0):
    """acts_*: {feature: [clean-base max act of each judge-written text]}. Per feature:
    HOLDS iff mean(true) > mean(naive) AND mean(true) > frac * corpus_peak. Aggregated over features
    that have BOTH arms scored."""
    feats = [f for f in acts_true if acts_true.get(f) and acts_naive.get(f)]
    out = {"extra/adversarial/n_features": float(len(feats))}
    if not feats:
        return out
    tm = np.array([np.mean(acts_true[f]) for f in feats], dtype=np.float64)
    nm = np.array([np.mean(acts_naive[f]) for f in feats], dtype=np.float64)
    cp = np.array([max(float(corpus_peak.get(f, 0.0)), 1e-6) for f in feats], dtype=np.float64)
    holds = (tm > nm) & (tm > frac * cp)
    out.update({
        "extra/adversarial/holds_frac": float(holds.mean()),
        "extra/adversarial/true_gt_naive_frac": float((tm > nm).mean()),
        "extra/adversarial/true_gt_quarter_peak_frac": float((tm > frac * cp).mean()),
        "extra/adversarial/naive_gt_quarter_peak_frac": float((nm > frac * cp).mean()),
        "extra/adversarial/true_act_norm_mean": float((tm / cp).mean()),
        "extra/adversarial/naive_act_norm_mean": float((nm / cp).mean()),
        "extra/adversarial/true_fire_frac": float(np.mean([np.mean(np.array(acts_true[f]) > fire) for f in feats])),
        "extra/adversarial/naive_fire_frac": float(np.mean([np.mean(np.array(acts_naive[f]) > fire) for f in feats])),
    })
    return out


# ----------------------------------------------------------------------------------------------
# judge client
# ----------------------------------------------------------------------------------------------
class JudgeError(Exception):
    def __init__(self, msg, retryable):
        super().__init__(msg)
        self.retryable = retryable


def _http_post_json(url, headers, body, timeout_s):
    """POST JSON -> (status, parsed body). `requests` when present (vLLM images have it), stdlib otherwise.
    Network / timeout errors propagate (the caller retries them)."""
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text[:500]}
        return r.status_code, data
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
        except Exception:  # noqa
            data = {}
        return e.code, data


class JudgeClient:
    """Thread-safe LLM-judge client. complete(system, user, max_tokens) -> (text, usage) with
    usage = {"in": tokens, "out": tokens, "cost": USD}. Retries with exponential backoff + jitter on
    429 / 408 / 409 / 5xx / 529 / network errors; raises JudgeError when attempts are exhausted or the
    error is not retryable. `call_fn(system, user, max_tokens)` injects a fake backend for tests."""

    RETRY_STATUS = {408, 409, 429, 529}

    def __init__(self, provider, model, api_key=None, workspace_id=None, timeout_s=60.0, max_attempts=6,
                 call_fn=None):
        self.provider, self.model = provider, model
        self.timeout_s, self.max_attempts = timeout_s, max_attempts
        self._lock = threading.Lock()
        self.totals = {"calls": 0, "fails": 0, "in": 0, "out": 0, "cost": 0.0}
        if call_fn is not None:
            self._call = call_fn
        elif provider == "anthropic":
            import anthropic   # optional dependency: only the native path needs it
            headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout_s,
                                               default_headers=headers)
            self._call = self._anthropic
        elif provider == "openrouter":
            self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                             "X-Title": "maemm-inline-extra-evals"}
            self._call = self._openrouter
        else:
            raise ValueError(f"unknown judge provider {provider!r}")

    # -- backends -------------------------------------------------------------------------------
    def _price(self, n_in, n_out):
        key = next((k for k in PRICE_PER_M if k in self.model.replace(".", "-")), None)
        pi, po = PRICE_PER_M.get(key, (2.0, 10.0))
        return (n_in * pi + n_out * po) / 1e6

    def _anthropic(self, system, user, max_tokens):
        import anthropic
        try:
            r = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                             messages=[{"role": "user", "content": user}],
                                             thinking={"type": "disabled"})
        except anthropic.RateLimitError as e:
            raise JudgeError(f"429 {e.message}", True)
        except anthropic.APIStatusError as e:
            raise JudgeError(f"{e.status_code} {e.message}", e.status_code in self.RETRY_STATUS or e.status_code >= 500)
        except anthropic.APIConnectionError as e:      # includes APITimeoutError
            raise JudgeError(f"connection: {e}", True)
        if r.stop_reason == "refusal":
            raise JudgeError("refusal", False)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        u = r.usage
        return text, {"in": int(u.input_tokens), "out": int(u.output_tokens),
                      "cost": self._price(u.input_tokens, u.output_tokens)}

    def _openrouter(self, system, user, max_tokens):
        body = {"model": self.model, "max_tokens": max_tokens,
                "reasoning": {"enabled": False},            # Sonnet 5 thinks by default -> would eat max_tokens
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "usage": {"include": True}}
        status, data = _http_post_json(OPENROUTER_URL, self._headers, body, self.timeout_s)
        if status != 200:
            raise JudgeError(f"HTTP {status}: {str(data)[:200]}", status in self.RETRY_STATUS or status >= 500)
        err = data.get("error") if isinstance(data, dict) else None
        if err:                                         # provider errors can arrive as 200 + error body
            code = int(err.get("code") or 500)
            raise JudgeError(f"provider {code}: {str(err)[:200]}", code in self.RETRY_STATUS or code >= 500)
        try:
            ch = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise JudgeError(f"malformed response: {str(data)[:200]}", True)
        if ch.get("finish_reason") in ("content_filter",):
            raise JudgeError("refusal/content_filter", False)
        text = ch["message"].get("content") or ""
        u = data.get("usage") or {}
        n_in, n_out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        cost = u.get("cost")
        return text, {"in": n_in, "out": n_out, "cost": float(cost) if cost is not None else self._price(n_in, n_out)}

    # -- public ---------------------------------------------------------------------------------
    def complete(self, system, user, max_tokens, max_attempts=None):
        attempts = max_attempts or self.max_attempts
        delay = 1.0
        for attempt in range(attempts):
            try:
                text, usage = self._call(system, user, max_tokens)
            except JudgeError as e:
                if not e.retryable or attempt == attempts - 1:
                    self._account(None)
                    raise
            except Exception as e:  # noqa — network / timeout / decode errors: retry
                if attempt == attempts - 1:
                    self._account(None)
                    raise JudgeError(f"{type(e).__name__}: {e}", True)
            else:
                self._account(usage)
                return text, usage
            time.sleep(min(60.0, delay) + random.uniform(0.0, 0.5))
            delay *= 2
        raise JudgeError("unreachable", False)

    def _account(self, usage):
        with self._lock:
            self.totals["calls"] += 1
            if usage is None:
                self.totals["fails"] += 1
            else:
                self.totals["in"] += usage["in"]
                self.totals["out"] += usage["out"]
                self.totals["cost"] += usage["cost"]

    def snapshot(self):
        with self._lock:
            return dict(self.totals)


def make_judge_from_env(verbose=True):
    """Provider by env (EXTRA_EVAL_JUDGE_PROVIDER=auto|anthropic|openrouter): Anthropic native when
    ANTHROPIC_API_KEY is set (+ ANTHROPIC_WORKSPACE_ID header if set), else OpenRouter when
    OPENROUTER_API_KEY is set. Each candidate is SELF-TESTED with one tiny call; the first that answers
    is used. Returns None (judge stage disabled) when nothing works — never raises."""
    provider = os.environ.get("EXTRA_EVAL_JUDGE_PROVIDER", "auto").lower()
    model = os.environ.get("EXTRA_EVAL_JUDGE_MODEL") or None
    cands = []
    if provider in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        cands.append(("anthropic", model or "claude-sonnet-5", os.environ["ANTHROPIC_API_KEY"]))
    if provider in ("auto", "openrouter") and os.environ.get("OPENROUTER_API_KEY"):
        cands.append(("openrouter", model or "anthropic/claude-sonnet-5", os.environ["OPENROUTER_API_KEY"]))
    if not cands:
        if verbose:
            print("[extra-eval] judge DISABLED: set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY) for the judge stage", flush=True)
        return None
    for prov, mdl, key in cands:
        try:
            jc = JudgeClient(prov, mdl, api_key=key, workspace_id=os.environ.get("ANTHROPIC_WORKSPACE_ID"),
                             timeout_s=45.0)
            t0 = time.time()
            txt, usage = jc.complete("Reply with ONLY the requested integer.", "Reply with the integer 7.", 8,
                                     max_attempts=2)
            if parse_score(txt) != 7:
                raise JudgeError(f"self-test reply {txt!r} (expected 7)", False)
            if verbose:
                print(f"[extra-eval] judge OK: {prov} {mdl} ({time.time() - t0:.1f}s, ${usage['cost']:.6f}/call)", flush=True)
            jc.totals = {"calls": 0, "fails": 0, "in": 0, "out": 0, "cost": 0.0}   # don't bill the self-test
            return jc
        except Exception as e:  # noqa
            if verbose:
                print(f"[extra-eval] judge candidate {prov} {mdl} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    if verbose:
        print("[extra-eval] judge DISABLED: no candidate passed the self-test", flush=True)
    return None


# ----------------------------------------------------------------------------------------------
# sub-SAE: only the testbed features' encoder columns (1.3 MB instead of the 2.7 GB W_enc)
# ----------------------------------------------------------------------------------------------
class SubSAE:
    """W [d, k] = W_enc[:, feats], b [k] = b_enc[feats], b_dec [d]. encode_features(h, ids) is
    numerically identical to mxf.sae.BatchTopKSAE.encode_features for these feature ids."""

    def __init__(self, W, b, b_dec, feats):
        self.W, self.b, self.b_dec = W, b, b_dec
        self.feats = [int(f) for f in feats]
        self.col = {f: i for i, f in enumerate(self.feats)}

    def _idx(self, feature_ids):
        return torch.as_tensor([self.col[int(f)] for f in feature_ids], device=self.W.device, dtype=torch.long)

    def encode_features(self, acts_BLD, feature_ids):
        idx = self._idx(feature_ids)
        return torch.relu((acts_BLD.to(self.W.dtype) - self.b_dec) @ self.W[:, idx] + self.b[idx])

    def enc_dirs(self, feature_ids):
        return F.normalize(self.W[:, self._idx(feature_ids)].T, dim=-1)

    @classmethod
    def from_sae(cls, sae, feats, device):
        idx = torch.as_tensor([int(f) for f in feats], device=sae.W_enc.device, dtype=torch.long)
        return cls(sae.W_enc[:, idx].detach().float().clone().to(device), sae.b_enc[idx].detach().float().clone().to(device),
                   sae.b_dec.detach().float().clone().to(device), feats)

    @classmethod
    def from_file(cls, path, feats, device):
        """Slice straight out of the dictionary_learning state dict (encoder.weight is [F, d] -> the
        feature ROWS are contiguous); mmap avoids reading the 5 GB file when the format allows it."""
        try:
            params = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except Exception:  # noqa — legacy (non-zip) serialization
            params = torch.load(path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor([int(f) for f in feats], dtype=torch.long)
        W = params["encoder.weight"][rows].float().T.contiguous()             # [d, k]
        b = params["encoder.bias"][rows].float().contiguous()
        b_dec = (params["bias"] if "bias" in params else params["b_dec"]).float().contiguous()
        return cls(W.to(device), b.to(device), b_dec.to(device), feats)


# ----------------------------------------------------------------------------------------------
# assets
# ----------------------------------------------------------------------------------------------
def _load_wildchat(path, feats, verbose):
    if not os.path.exists(path):
        if verbose:
            print(f"[extra-eval] wildchat eval OFF: no bank at {path} (build it once with modal_wildchat_bank.py)", flush=True)
        return None
    bank = _load_json(path)
    by_f = {}
    for r in bank["features"]:
        f = int(r["feature"])
        if f not in feats:
            continue
        pos = [w for w in r["windows"] if w["fires"]]
        neg = [w for w in r["windows"] if not w["fires"]]
        if len(pos) >= 2 and len(neg) >= 2:
            by_f[f] = {"windows": pos + neg}
    if verbose:
        print(f"[extra-eval] wildchat bank {path}: {len(by_f)}/{len(feats)} testbed features with >=2 firing + >=2 "
              f"non-firing windows (win={bank.get('config', {}).get('win')})", flush=True)
    return by_f or None


def _naive_cache_path(a):
    p = os.environ.get("EXTRA_EVAL_NAIVE_CACHE", DEFAULT_NAIVE_CACHE)
    d = os.path.dirname(p) or "."
    if os.path.isdir(d) and os.access(d, os.W_OK):
        return p
    return os.path.join(getattr(a, "save_dir", None) or "/tmp", "extra_evals", os.path.basename(p))


def _load_naive_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        raw = _load_json(path).get("entries", {})
        return {int(k): v for k, v in raw.items()}
    except Exception as e:  # noqa
        print(f"[extra-eval] naive cache unreadable ({e}) — recomputing", flush=True)
        return {}


def _save_json_atomic(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def build_assets(tb, subsae=None, wildchat=None, judge=None, naive=None, naive_cache_path=None, out_dir="/tmp/extra_evals",
                 n_rollouts=None):
    """Assemble the assets dict from an already-loaded testbed (shared by prepare and the tests)."""
    feats = [int(r["feature"]) for r in tb["features"]]
    recs = {int(r["feature"]): r for r in tb["features"]}
    dirs = subsae.enc_dirs(feats).float().cpu() if subsae is not None else None
    return {"tb_config": tb["config"], "feats": feats, "fidx": {f: i for i, f in enumerate(feats)}, "recs": recs,
            "corpus_peak": {f: float(recs[f]["corpus_peak"]) for f in feats},
            "fire": float(tb["config"].get("fire", 1.0)), "subsae": subsae, "dirs": dirs, "wildchat": wildchat,
            "judge": judge, "naive": naive if naive is not None else {}, "naive_lock": threading.Lock(),
            "naive_cache_path": naive_cache_path, "pending_adv": queue.Queue(), "last_rollouts": None,
            "n_launch": 0, "out_dir": out_dir, "n_rollouts": n_rollouts or N_ROLLOUTS}


def prepare_extra_eval_assets(a, device, rank, world, is_main, sae=None):
    """Load the testbed (+ the testbed features' SAE encoder columns), the WildChat bank and the cached
    naive descriptions ONCE per rank. `sae`: pass the trainer's already-loaded SAE (EV["sae"]) to slice
    from it instead of touching /data/sae/ae.pt. Never raises: returns None (+ prints why) — and all
    ranks agree on None-vs-assets via an all_reduce, so no rank can be left waiting in a collective."""
    assets, err = None, None
    try:
        if _IMPORT_ERR is not None:
            raise ImportError(f"eval modules not importable ({_IMPORT_ERR}); mount eval/snippet_locality.py and "
                              "eval/autointerp_detection.py on sys.path (e.g. /pmx/eval)")
        tb_path = os.environ.get("EXTRA_EVAL_TESTBED") or getattr(a, "extra_testbed", None) or DEFAULT_TESTBED
        tb = _load_json(tb_path)
        feats = [int(r["feature"]) for r in tb["features"]]
        need = ("desc_examples", "positives", "rollouts_temp") + tuple(f"negatives_{k}" for k in NEG_KINDS
                                                                       if "autointerp" not in DISABLED)
        for k in need:
            assert k in tb["features"][0], f"testbed {tb_path} lacks '{k}' (run autointerp_detection.py augment?)"
        sae_path = os.environ.get("EXTRA_EVAL_SAE") or getattr(a, "eval_sae", None) or DEFAULT_SAE
        subsae = SubSAE.from_sae(sae, feats, device) if sae is not None else SubSAE.from_file(sae_path, feats, device)
        out_dir = os.path.join(getattr(a, "save_dir", None) or "/tmp", "extra_evals")
        wildchat = judge = None
        naive, ncp = {}, None
        if is_main:
            if "wildchat" not in DISABLED:
                wildchat = _load_wildchat(os.environ.get("EXTRA_EVAL_WILDCHAT", DEFAULT_WILDCHAT), set(feats), True)
            judge = make_judge_from_env(verbose=True)
            ncp = _naive_cache_path(a)
            naive = _load_naive_cache(ncp)
        assets = build_assets(tb, subsae=subsae, wildchat=wildchat, judge=judge, naive=naive, naive_cache_path=ncp,
                              out_dir=out_dir)
        assets["testbed_path"] = tb_path
    except Exception as e:  # noqa
        err = f"{type(e).__name__}: {str(e)[:300]}"
    ok = torch.tensor([0 if assets is None else 1], dtype=torch.long)
    if world > 1 and dist.is_available() and dist.is_initialized():
        try:
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        except Exception as e:  # noqa
            print(f"[extra-eval] rank {rank}: agreement all_reduce failed ({e}) — DISABLED", flush=True)
            return None
    if int(ok.item()) == 0:
        print(f"[extra-eval] DISABLED on rank {rank}: " + (err or "another rank failed to load its assets"), flush=True)
        return None
    if is_main:
        n_missing = sum(1 for f in assets["feats"] if not (assets["naive"].get(f, {}).get("desc")
                                                          and assets["naive"].get(f, {}).get("texts")))
        print(f"[extra-eval] ready: {len(assets['feats'])} testbed features ({assets['testbed_path']}) | "
              f"{assets['n_rollouts']} rollouts/feature, T={assets['tb_config'].get('temp')}, "
              f"{assets['tb_config'].get('min_new')}-{assets['tb_config'].get('max_new')} tok | judge "
              f"{(assets['judge'].provider + ' ' + assets['judge'].model) if assets['judge'] else 'OFF'} | wildchat "
              f"{len(assets['wildchat']) if assets['wildchat'] else 0} feats | naive cache {assets['naive_cache_path']} "
              f"({len(assets['feats']) - n_missing}/{len(assets['feats'])} cached)", flush=True)
        if assets["judge"] is not None and n_missing and "adversarial" not in DISABLED:
            threading.Thread(target=_ensure_naive, args=(assets, assets["judge"], time.time() + JUDGE_TIMEOUT_S),
                             daemon=True, name="extra-naive-warm").start()
    return assets


# ----------------------------------------------------------------------------------------------
# GPU stage
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _profiles(texts, feats, actor, tok, device, subsae, batch=32):
    """Per-text kept-token activation profile of ITS paired feature on the CLEAN base — the exact
    snippet_locality.cmd_build read path (BOS sink prepended + skipped, right padding masked, 10x-median
    norm filter, ReLU SAE encode). Returns list of 1-D np.float32 arrays."""
    from mxf.config import READ_LAYER
    from mxf.inject import read_resid
    prev = tok.padding_side
    tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out = []
    try:
        for s in range(0, len(texts), batch):
            chunk = [t if t.strip() else " " for t in texts[s:s + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_TOKENS,
                      add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            with actor.disable_adapter():
                h, mask = read_resid(actor, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone()
            keep[:, 0] = False                                                   # attention-sink guard
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            per = subsae.encode_features(h, feats[s:s + B])                     # [B, T, B]
            bi = torch.arange(B, device=per.device)
            act = per[bi, :, bi].float().cpu().numpy()                           # [B, T] ReLU acts
            kb = keep.cpu().numpy()
            for b in range(B):
                out.append(act[b][kb[b]].astype(np.float32))
    finally:
        tok.padding_side = prev
    return out


def _broadcast_pending(assets, rank, world):
    """Rank 0 drains the judge's pending adversarial texts; every rank receives the same list."""
    items = []
    if rank == 0:
        q = assets["pending_adv"]
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break
    if world > 1 and dist.is_available() and dist.is_initialized():
        obj = [items if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        items = obj[0] or []
    return items


@torch.no_grad()
def run_extra_evals_gpu(llm, actor, submodule, tok, prompt_ids, marker, a, device, ckpt_step, lora_step, rank, world,
                        assets, steer_fn, marker_norm_fn, eos_ids, trim_fn):
    """GPU stage for the checkpoint whose weights this rank published at loop step `lora_step`
    (== step_{ckpt_step}). Mirrors inline_eval's request construction: steer_fn == rl._steer_vec,
    marker_norm_fn == rl._marker_norm, eos_ids == rl._eos_ids(tok, actor), trim_fn == rl._trim_at_stop.
    Every rank ALWAYS joins the broadcast + all_gather. Rank 0 returns {"extra/locality/...",
    "time/extra_eval_gpu_s"} (or {"error": ...}); other ranks return {}. Adversarial scores of the
    previous checkpoint's judge texts are pushed to the results queue under THEIR ckpt_step. Rank 0
    also stores the rollouts in assets["last_rollouts"] for launch_judge_stage."""
    t0 = time.time()
    if assets is None:
        return {}
    pending = _broadcast_pending(assets, rank, world)
    local = {}
    try:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        feats, cfg, dirs, subsae = assets["feats"], assets["tb_config"], assets["dirs"], assets["subsae"]
        n_roll = assets["n_rollouts"]
        rows = _shard_rows(len(feats), rank, world)
        texts_by_feat = {}
        if rows:
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            eos = set(int(x) for x in eos_ids)
            hnorm = marker_norm_fn(actor, submodule, prompt, marker, device, adapter=True)
            lora_req = LoRARequest(lora_name=f"step{lora_step}", lora_int_id=lora_step + 1,
                                   lora_path=f"/tmp/rl_lora/rank{rank}/step{lora_step}")
            max_new = min(int(cfg.get("max_new", 64)), int(getattr(a, "max_new_tokens", 96)))
            min_new = min(int(cfg.get("min_new", 16)), max_new)
            reqs, params = [], []
            for i in rows:
                sv = steer_fn(F.normalize(dirs[i].float(), dim=-1), hnorm, marker)
                reqs.append({"prompt_token_ids": list(prompt_ids)})
                params.append(SamplingParams(n=n_roll, temperature=float(cfg.get("temp", 1.0)), top_p=1.0, top_k=0,
                                             min_p=0.0, repetition_penalty=1.0, max_tokens=max_new, min_tokens=min_new,
                                             stop_token_ids=sorted(eos), seed=(SL.GEN_SEED * 1000 + i) % 2147483647,
                                             extra_args={"apply_steering_vectors": [sv]}))
            outs = llm.generate(reqs, params, lora_request=lora_req, use_tqdm=False)
            for i, out in zip(rows, outs):                                      # request order == rows order
                assert len(out.outputs) == n_roll, f"expected {n_roll} samples, got {len(out.outputs)}"
                texts_by_feat[feats[i]] = [(tok.decode(trim_fn(list(o.token_ids), eos), skip_special_tokens=True).strip() or " ")
                                           for o in out.outputs]
        t_gen = time.time() - t0
        # ---- locality profiles of my rollouts (clean base) ----
        flat_t = [t for i in rows for t in texts_by_feat[feats[i]]]
        flat_f = [feats[i] for i in rows for _ in range(n_roll)]
        loc_rows = locality_rows_from_profiles(_profiles(flat_t, flat_f, actor, tok, device, subsae) if flat_t else [],
                                               assets["fire"])
        loc = {feats[i]: loc_rows[k * n_roll:(k + 1) * n_roll] for k, i in enumerate(rows)}
        # ---- adversarial texts from the previous checkpoint(s): score my shard of features ----
        adv = []
        fidx = assets["fidx"]
        for item in pending:
            res = {"src": item["src_ckpt_step"], "true": {}, "naive": {}}
            at, af, tags = [], [], []
            for arm in ("true", "naive"):
                for f, txts in item[arm].items():
                    f = int(f)
                    if f in fidx and fidx[f] % world == rank:
                        for t in txts:
                            at.append(t); af.append(f); tags.append((arm, f))
            profs = _profiles(at, af, actor, tok, device, subsae) if at else []
            for (arm, f), p in zip(tags, profs):
                res[arm].setdefault(f, []).append(float(p.max()) if len(p) else 0.0)
            adv.append(res)
        local = {"texts": texts_by_feat, "loc": loc, "adv": adv, "t_gen": t_gen}
    except Exception as e:  # noqa
        local = {"error": f"rank{rank}: {type(e).__name__}: {str(e)[:300]}"}
    gathered = [None] * world
    if world > 1 and dist.is_available() and dist.is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return {}
    errs = [g["error"] for g in gathered if "error" in g]
    if errs:
        return {"error": " | ".join(errs)}
    texts, loc = {}, {}
    adv_true, adv_naive = {}, {}
    for g in gathered:
        texts.update(g["texts"])
        loc.update(g["loc"])
        for res in g["adv"]:
            for arm, store in (("true", adv_true), ("naive", adv_naive)):
                for f, acts in res[arm].items():
                    store.setdefault(res["src"], {}).setdefault(int(f), []).extend(acts)
    out = aggregate_locality(loc, assets["corpus_peak"])
    for src in sorted(set(adv_true) | set(adv_naive)):
        m = adversarial_metrics(adv_true.get(src, {}), adv_naive.get(src, {}), assets["corpus_peak"], fire=assets["fire"])
        _RESULTS_Q.put((src, m))
        print(f"[extra-eval] adversarial scores for ckpt {src}: holds {m.get('extra/adversarial/holds_frac', float('nan')):.3f} "
              f"over {int(m['extra/adversarial/n_features'])} feats (queued for wandb)", flush=True)
    out["extra/adversarial/n_pending_scored"] = float(len(pending))
    out["time/extra_eval_gpu_s"] = time.time() - t0
    out["time/extra_eval_gen_s"] = float(max(g.get("t_gen", 0.0) for g in gathered))
    assets["last_rollouts"] = {"ckpt_step": ckpt_step, "rollouts": texts}
    try:
        os.makedirs(assets["out_dir"], exist_ok=True)
        _save_json_atomic({"ckpt_step": ckpt_step, "rollouts": {str(f): v for f, v in texts.items()},
                           "locality": {str(f): v for f, v in loc.items()}},
                          os.path.join(assets["out_dir"], f"rollouts_ckpt{ckpt_step}.json"))
    except Exception as e:  # noqa
        print(f"[extra-eval] could not write rollouts artifact: {e}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# judge stage (rank 0, background)
# ----------------------------------------------------------------------------------------------
_RESULTS_Q = queue.Queue()
_EXECUTOR = None
_EXECUTOR_LOCK = threading.Lock()
_STAGE_SEM = threading.BoundedSemaphore(MAX_CONCURRENT_STAGES)
_STAGE_THREADS = []


def _executor():
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY, thread_name_prefix="extra-judge")
        return _EXECUTOR


def _run_futures(futs, deadline, on_done=None):
    """futs: {future: tag}. Waits for all (or the deadline). on_done(tag, value) may return extra
    {future: tag} to add (chaining). Returns ({tag: value-or-Exception}, timed_out)."""
    results, pending = {}, set(futs)
    timed_out = False
    while pending:
        remaining = deadline - time.time()
        if remaining <= 0:
            timed_out = True
            break
        done, pending = wait(pending, timeout=min(remaining, 5.0), return_when=FIRST_COMPLETED)
        for fu in done:
            tag = futs[fu]
            try:
                val = fu.result()
            except Exception as e:  # noqa
                val = e
            results[tag] = val
            if on_done is not None:
                for nf, ntag in (on_done(tag, val) or {}).items():
                    futs[nf] = ntag
                    pending.add(nf)
    for fu in pending:
        fu.cancel()
        results[futs[fu]] = TimeoutError("stage deadline")
    return results, timed_out


def _score_call(judge, desc_snippets, text):
    txt, usage = judge.complete(AD.JUDGE_SYSTEM, _detect_prompt(desc_snippets, text), 8)
    return parse_score(txt), usage


def _describe_call(judge, snippets, source):
    tmpl = DESC_USER_CORPUS if source == "corpus" else DESC_USER_GENERATED
    txt, usage = judge.complete(DESC_SYSTEM, tmpl.format(n=len(snippets), block=_snippet_block(snippets)), 200)
    return " ".join((txt or "").split())[:600], usage


def _gen_texts_call(judge, desc, k):
    txt, usage = judge.complete(ADV_SYSTEM, ADV_USER.format(desc=desc, k=k), 1200)
    return parse_json_list(txt, k), usage


def _usage_of(val):
    return val[1] if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], dict) else None


def _ensure_naive(assets, judge, deadline):
    """Naive corpus-only description (judge over the top-8 corpus max-act windows) + ADV_K texts fitting
    it, per feature, computed ONCE and cached on disk. Blocking (call from a background thread)."""
    with assets["naive_lock"]:
        naive = assets["naive"]
        todo = [f for f in assets["feats"] if not (naive.get(f, {}).get("desc") and naive.get(f, {}).get("texts"))]
        if not todo or judge is None:
            return naive
        t0 = time.time()
        ex = _executor()
        futs = {}
        for f in todo:
            if naive.get(f, {}).get("desc"):
                futs[ex.submit(_gen_texts_call, judge, naive[f]["desc"], ADV_K)] = ("texts", f)
            else:
                snips = [e["text"] for e in assets["recs"][f]["desc_examples"][:8]]
                futs[ex.submit(_describe_call, judge, snips, "corpus")] = ("desc", f)

        def chain(tag, val):
            if tag[0] == "desc" and isinstance(val, tuple) and val[0]:
                naive.setdefault(tag[1], {})["desc"] = val[0]
                naive[tag[1]]["model"] = judge.model
                return {ex.submit(_gen_texts_call, judge, val[0], ADV_K): ("texts", tag[1])}
            if tag[0] == "texts" and isinstance(val, tuple) and val[0]:
                naive.setdefault(tag[1], {})["texts"] = val[0]
            return None
        res, timed_out = _run_futures(futs, deadline, on_done=chain)
        cost = sum(_usage_of(v)["cost"] for v in res.values() if _usage_of(v))
        n_fail = sum(1 for v in res.values() if isinstance(v, Exception) or (isinstance(v, tuple) and not v[0]))
        done = sum(1 for f in assets["feats"] if naive.get(f, {}).get("desc") and naive.get(f, {}).get("texts"))
        if assets.get("naive_cache_path"):
            try:
                _save_json_atomic({"entries": {str(f): v for f, v in naive.items()}, "judge_model": judge.model,
                                   "testbed": assets.get("testbed_path"), "adv_k": ADV_K},
                                  assets["naive_cache_path"])
            except Exception as e:  # noqa
                print(f"[extra-eval] naive cache write failed: {e}", flush=True)
        print(f"[extra-eval] naive descriptions+texts: {done}/{len(assets['feats'])} ready ({len(res)} calls, {n_fail} failed, "
              f"${cost:.3f}, {time.time() - t0:.0f}s{', TIMED OUT' if timed_out else ''}) -> {assets.get('naive_cache_path')}", flush=True)
        return naive


def _judge_stage(rollouts, ckpt_step, assets, a, judge):
    t0 = time.time()
    deadline = t0 + JUDGE_TIMEOUT_S
    ex = _executor()
    feats, recs, wc = assets["feats"], assets["recs"], assets["wildchat"]
    do_ai = "autointerp" not in DISABLED
    do_wc = "wildchat" not in DISABLED and wc
    do_adv = "adversarial" not in DISABLED
    n_list = [n for n in N_LIST if n <= assets["n_rollouts"]] or [assets["n_rollouts"]]
    naive = _ensure_naive(assets, judge, deadline) if do_adv else {}
    futs = {}
    for f in feats:
        rolls = [t for t in (rollouts.get(f) or rollouts.get(str(f)) or []) if t.strip()]
        if not rolls:
            continue
        rec = recs[f]
        if do_ai:
            tests = [(("pos", i), p["text"], 1) for i, p in enumerate(rec["positives"][:N_TESTS])]
            for kind in NEG_KINDS:
                tests += [((kind, i), n["text"], 0) for i, n in enumerate(rec.get(f"negatives_{kind}", [])[:N_TESTS])]
            for N in n_list:
                for tid, text, label in tests:
                    futs[ex.submit(_score_call, judge, rolls[:N], text)] = ("ai", f, N, tid, label)
        if do_wc and f in wc:
            for N in n_list:
                for wi, w in enumerate(wc[f]["windows"]):
                    futs[ex.submit(_score_call, judge, rolls[:N], w["text"])] = ("wc", f, N, wi, int(w["fires"]))
        if do_adv:
            futs[ex.submit(_describe_call, judge, rolls, "generated")] = ("rdesc", f)

    def chain(tag, val):
        if tag[0] == "rdesc" and isinstance(val, tuple) and val[0]:
            return {ex.submit(_gen_texts_call, judge, val[0], ADV_K): ("adv", tag[1])}
        return None
    res, timed_out = _run_futures(futs, deadline, on_done=chain)

    # ---- aggregate ----
    n_calls = len(res)
    usages = [_usage_of(v) for v in res.values()]
    n_fail = sum(1 for v in res.values() if isinstance(v, Exception))
    cost = sum(u["cost"] for u in usages if u)
    n_in = sum(u["in"] for u in usages if u)
    n_out = sum(u["out"] for u in usages if u)
    m = {"extra/judge/n_calls": float(n_calls), "extra/judge/n_fail": float(n_fail),
         "extra/judge/fail_frac": (n_fail / n_calls) if n_calls else 0.0, "extra/judge/timed_out": float(timed_out),
         "extra/judge/latency_s": time.time() - t0, "extra/cost_usd": cost, "extra/judge/in_tokens": float(n_in),
         "extra/judge/out_tokens": float(n_out), "extra/cost_usd_total": judge.snapshot()["cost"]}

    def score_of(v):
        return v[0] if isinstance(v, tuple) and v[0] is not None else 50      # offline convention: failure -> uninformative
    if do_ai:
        det = {}
        for tag, v in res.items():
            if tag[0] == "ai":
                _, f, N, tid, label = tag
                det.setdefault((f, N), {}).setdefault("pos" if label else tid[0], []).append(score_of(v))
        per_feat = {}
        for N in n_list:
            for kind in NEG_KINDS:
                vals = [auc(d["pos"], d[kind]) for (f, n), d in det.items() if n == N and d.get("pos") and d.get(kind)]
                vals = [v for v in vals if v is not None]
                m[f"extra/autointerp/auc_{kind}_n{N}"] = float(np.mean(vals)) if vals else float("nan")
                m[f"extra/autointerp/n_features_{kind}_n{N}"] = float(len(vals))
            bal = [0.5 * (np.mean(np.array(d["pos"]) >= 50) + np.mean(np.array(sum((d.get(k, []) for k in NEG_KINDS), [])) < 50))
                   for (f, n), d in det.items() if n == N and d.get("pos") and any(d.get(k) for k in NEG_KINDS)]
            m[f"extra/autointerp/bal_acc_n{N}"] = float(np.mean(bal)) if bal else float("nan")
        for (f, N), d in det.items():
            per_feat.setdefault(str(f), {})[f"N{N}"] = {k: auc(d["pos"], d[k]) for k in NEG_KINDS if d.get(k)}
        m["extra/autointerp/n_features"] = float(len({f for (f, n) in det}))
    else:
        per_feat = {}
    if do_wc:
        wcd = {}
        for tag, v in res.items():
            if tag[0] == "wc":
                _, f, N, wi, fires = tag
                wcd.setdefault((f, N), {}).setdefault("pos" if fires else "neg", []).append(score_of(v))
        for N in n_list:
            vals = [auc(d["pos"], d["neg"]) for (f, n), d in wcd.items() if n == N and d.get("pos") and d.get("neg")]
            vals = [v for v in vals if v is not None]
            m[f"extra/wildchat/auc_n{N}"] = float(np.mean(vals)) if vals else float("nan")
            bal = [0.5 * (np.mean(np.array(d["pos"]) >= 50) + np.mean(np.array(d["neg"]) < 50))
                   for (f, n), d in wcd.items() if n == N and d.get("pos") and d.get("neg")]
            m[f"extra/wildchat/bal_acc_n{N}"] = float(np.mean(bal)) if bal else float("nan")
            m[f"extra/wildchat/n_features_n{N}"] = float(len(vals))
    rdesc, adv_true = {}, {}
    if do_adv:
        for tag, v in res.items():
            if tag[0] == "rdesc" and isinstance(v, tuple) and v[0]:
                rdesc[tag[1]] = v[0]
            elif tag[0] == "adv" and isinstance(v, tuple) and v[0]:
                adv_true[tag[1]] = v[0]
        adv_naive = {f: naive[f]["texts"] for f in adv_true if naive.get(f, {}).get("texts")}
        m["extra/adversarial/n_generated"] = float(len(adv_true))
        m["extra/adversarial/n_naive_ready"] = float(len(adv_naive))
        if adv_true:
            assets["pending_adv"].put({"src_ckpt_step": ckpt_step, "true": adv_true, "naive": adv_naive})
    _RESULTS_Q.put((ckpt_step, m))
    try:
        os.makedirs(assets["out_dir"], exist_ok=True)
        _save_json_atomic({"ckpt_step": ckpt_step, "judge": f"{judge.provider}:{judge.model}", "metrics": m,
                           "rollouts": {str(f): v for f, v in rollouts.items()},
                           "rollout_desc": {str(f): v for f, v in rdesc.items()},
                           "adv_true_texts": {str(f): v for f, v in adv_true.items()},
                           "naive": {str(f): naive.get(f) for f in adv_true} if do_adv else {},
                           "autointerp_per_feature": per_feat,
                           "failures": [f"{tag}: {v}" for tag, v in res.items() if isinstance(v, Exception)][:50]},
                          os.path.join(assets["out_dir"], f"judge_ckpt{ckpt_step}.json"))
    except Exception as e:  # noqa
        print(f"[extra-eval] could not write judge artifact: {e}", flush=True)
    summary = " ".join(f"{k.split('/', 1)[1]}={v:.3f}" for k, v in sorted(m.items())
                       if k.startswith(("extra/autointerp/auc", "extra/wildchat/auc")) and not np.isnan(v))
    print(f"[extra-eval] judge ckpt {ckpt_step}: {summary} | {n_calls} calls ({n_fail} failed"
          f"{', TIMED OUT' if timed_out else ''}) ${cost:.2f} in {time.time() - t0:.0f}s", flush=True)


def _judge_stage_wrapper(rollouts, ckpt_step, assets, a, judge):
    try:
        _judge_stage(rollouts, ckpt_step, assets, a, judge)
    except Exception as e:  # noqa
        print(f"[extra-eval] judge stage for ckpt {ckpt_step} CRASHED: {type(e).__name__}: {e}", flush=True)
        _RESULTS_Q.put((ckpt_step, {"extra/judge/error": 1.0}))
    finally:
        _STAGE_SEM.release()


def launch_judge_stage(rollouts_by_feature, ckpt_step, assets, a):
    """Rank 0 only. Starts the background judge thread for ckpt_step's rollouts ({feature: [texts]};
    None -> the rollouts run_extra_evals_gpu just stored in assets["last_rollouts"]). Returns the
    Thread or None (judge off / EXTRA_EVAL_JUDGE_EVERY skip / too many stages in flight)."""
    if assets is None or assets.get("judge") is None:
        return None
    if rollouts_by_feature is None:
        lr = assets.get("last_rollouts")
        if not lr or lr["ckpt_step"] != ckpt_step:
            print(f"[extra-eval] no rollouts stored for ckpt {ckpt_step} — judge stage skipped", flush=True)
            return None
        rollouts_by_feature = lr["rollouts"]
    assets["n_launch"] = assets.get("n_launch", 0) + 1
    if (assets["n_launch"] - 1) % JUDGE_EVERY != 0:
        return None
    if not _STAGE_SEM.acquire(blocking=False):
        print(f"[extra-eval] {MAX_CONCURRENT_STAGES} judge stages already in flight — ckpt {ckpt_step} skipped", flush=True)
        _RESULTS_Q.put((ckpt_step, {"extra/judge/skipped": 1.0}))
        return None
    th = threading.Thread(target=_judge_stage_wrapper, args=(rollouts_by_feature, ckpt_step, assets, a, assets["judge"]),
                          daemon=True, name=f"extra-judge-{ckpt_step}")
    th.start()
    _STAGE_THREADS.append(th)
    return th


def poll_judge_results():
    """Drain finished judge results (non-blocking) -> [(ckpt_step, {"extra/...": float}), ...]."""
    out = []
    while True:
        try:
            out.append(_RESULTS_Q.get_nowait())
        except queue.Empty:
            return out


def wait_for_judge_stages(timeout_s=600.0):
    """Block up to timeout_s for in-flight judge stages (call once before the trainer exits, then poll)."""
    end = time.time() + timeout_s
    for th in list(_STAGE_THREADS):
        th.join(max(0.0, end - time.time()))
    return all(not th.is_alive() for th in _STAGE_THREADS)


# ----------------------------------------------------------------------------------------------
# smoke: no GPU — fake rollouts from the testbed's stored samples, run the judge stage LIVE
# ----------------------------------------------------------------------------------------------
def _smoke():
    import argparse
    ap = argparse.ArgumentParser(description="judge-stage smoke (no GPU)")
    ap.add_argument("--testbed", default=os.path.join(_HERE, "..", "eval", "out", "testbed_v2.json"))
    ap.add_argument("--wildchat", default=None)
    ap.add_argument("--n-features", type=int, default=2)
    ap.add_argument("--n-tests", type=int, default=2)
    ap.add_argument("--out-dir", default="/tmp/extra_evals_smoke")
    ap.add_argument("--naive-cache", default="/tmp/extra_evals_smoke/naive_desc_cache.json")
    args = ap.parse_args()
    global N_TESTS
    N_TESTS = args.n_tests
    tb = _load_json(args.testbed)
    tb["features"] = tb["features"][:args.n_features]
    feats = {int(r["feature"]) for r in tb["features"]}
    judge = make_judge_from_env()
    if judge is None:
        sys.exit("no judge configured")
    wc = _load_wildchat(args.wildchat, feats, True) if args.wildchat else None
    assets = build_assets(tb, wildchat=wc, judge=judge, naive=_load_naive_cache(args.naive_cache),
                          naive_cache_path=args.naive_cache, out_dir=args.out_dir)
    rollouts = {f: r["rollouts_temp"][:N_ROLLOUTS] for f, r in assets["recs"].items()}
    th = launch_judge_stage(rollouts, 0, assets, None)
    th.join()
    for step, m in poll_judge_results():
        print(f"ckpt {step}:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(m.items())}, indent=1))
    pend = _broadcast_pending(assets, 0, 1)
    print(f"pending adversarial items: {len(pend)}; features with true texts: "
          f"{[len(v) for it in pend for v in it['true'].values()]}")
    print("judge totals:", judge.snapshot())


if __name__ == "__main__":
    _smoke()
