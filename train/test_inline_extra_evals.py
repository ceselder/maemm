"""CPU-only unit tests for train/inline_extra_evals.py (pure parts: locality metrics, AUC math,
judge response parsing, retry/accounting, the background judge stage + queue plumbing, and the
never-raise contract of prepare_extra_eval_assets). No GPU, no network: the judge is a fake backend.

    python -m unittest train/test_inline_extra_evals.py -v
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inline_extra_evals as IE  # noqa: E402


def make_testbed(n_feats=2, n=10):
    feats = []
    for i in range(n_feats):
        f = 1000 + i
        feats.append({
            "feature": f, "corpus_peak": 10.0,
            "desc_examples": [{"text": f"corpus zebra window {j} for feature {f}", "act": 9.0 - j * 0.1} for j in range(8)],
            "positives": [{"text": f"a striped zebra grazes on the savanna {j}", "act": 5.0} for j in range(n)],
            "negatives": [{"text": f"quarterly revenue rose by {j} percent", "act": 0.0} for j in range(n)],
            "negatives_random": [{"text": f"quarterly revenue rose by {j} percent", "act": 0.0} for j in range(n)],
            "negatives_embnn": [{"text": f"a brown horse grazes on the savanna {j}", "act": 0.0} for j in range(n)],
            "negatives_nearmiss": [{"text": f"a giraffe grazes on the savanna {j}", "act": 0.5} for j in range(n)],
            "rollout_greedy": "zebra stripes everywhere",
            "rollouts_temp": [f"zebra rollout number {j}" for j in range(8)],
            "rollout_self_acts": {"greedy": 5.0, "temp": [5.0] * 8},
        })
    return {"config": {"fire": 1.0, "temp": 1.0, "max_new": 64, "min_new": 16, "n_max": 8, "n_desc": 8, "n_pos": n,
                       "n_neg": n}, "features": feats}


def make_wildchat(feats):
    return {f: {"windows": [{"text": f"user: tell me about zebra {j}", "act": 6.0, "fires": 1} for j in range(4)]
                + [{"text": f"user: fix my python bug {j}", "act": 0.0, "fires": 0} for j in range(4)]} for f in feats}


class FakeBackend:
    """Keyword oracle: detection score 90 if 'zebra' is in BOTH the description and the test snippet else 10.
    Descriptions/adversarial texts are canned. Counts calls; optional one-shot 429 for retry tests."""

    def __init__(self, fail_first_n_retryable=0, fail_all=False, latency=0.0):
        self.calls = 0
        self.lock = threading.Lock()
        self.fail_first = fail_first_n_retryable
        self.fail_all = fail_all
        self.latency = latency

    def __call__(self, system, user, max_tokens):
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.latency:
            time.sleep(self.latency)
        if self.fail_all:
            raise IE.JudgeError("400 bad request", False)
        if n <= self.fail_first:
            raise IE.JudgeError("429 rate limited", True)
        usage = {"in": 100, "out": 3, "cost": 0.001}
        if system == IE.AD.JUDGE_SYSTEM:
            desc = user.split("TEST SNIPPET:")[0]
            test = re.search(r"<<<\n(.*?)\n>>>", user, re.S).group(1)
            return ("90" if ("zebra" in desc and "zebra" in test) else "10"), usage
        if system == IE.DESC_SYSTEM:
            return "Text about zebras and their stripes.", usage
        if system == IE.ADV_SYSTEM:
            k = int(re.search(r"Write (\d+) diverse", user).group(1))
            return "```json\n" + json.dumps([f"Generated zebra text {i}" for i in range(k)]) + "\n```", usage
        return "7", usage


def fake_judge(**kw):
    be = FakeBackend(**kw)
    return IE.JudgeClient("openrouter", "fake/model", call_fn=be), be


class TestPure(unittest.TestCase):
    def test_shard_rows_cover_all_exactly_once(self):
        for world in (1, 3, 8):
            rows = sorted(r for rk in range(world) for r in IE._shard_rows(64, rk, world))
            self.assertEqual(rows, list(range(64)))

    def test_parse_score(self):
        self.assertEqual(IE.parse_score("85"), 85)
        self.assertEqual(IE.parse_score(" Probability: 120%"), 100)
        self.assertEqual(IE.parse_score("about 7 percent"), 7)
        self.assertIsNone(IE.parse_score("no digits"))
        self.assertIsNone(IE.parse_score(None))

    def test_parse_json_list(self):
        self.assertEqual(IE.parse_json_list('["a", "b", "c"]', 2), ["a", "b"])
        self.assertEqual(IE.parse_json_list('```json\n["x", 3, "", " y "]\n```', 4), ["x", "y"])
        self.assertEqual(IE.parse_json_list("Here: [\"q\"] done", 4), ["q"])
        self.assertEqual(IE.parse_json_list("not json", 4), [])
        self.assertEqual(IE.parse_json_list('{"a": 1}', 4), [])

    def test_auc(self):
        self.assertEqual(IE.auc([90, 80], [10, 20]), 1.0)
        self.assertEqual(IE.auc([10, 20], [90, 80]), 0.0)
        self.assertAlmostEqual(IE.auc([50, 50], [50, 50]), 0.5)
        self.assertAlmostEqual(IE.auc([1, 3], [2, 2]), 0.5)
        self.assertIsNone(IE.auc([], [1]))
        # same numbers as autointerp_detection._auc
        self.assertAlmostEqual(IE.auc([3, 1, 2], [2, 0]), IE.AD._auc([3, 1, 2], [2, 0]))

    def test_locality_rows(self):
        delta = np.zeros(20); delta[4] = 8.0
        uniform = np.ones(20) * 2.0
        tail = np.zeros(20); tail[-1] = 3.0
        long_uniform = np.ones(40)
        rows = IE.locality_rows_from_profiles([delta, uniform, tail, np.zeros(10), np.zeros(0), long_uniform], fire=1.0)
        d, u, t, z, e, lu = rows
        self.assertTrue(d["fired"]); self.assertAlmostEqual(d["metrics"]["win5_share"], 1.0)
        self.assertAlmostEqual(d["metrics"]["peak_share"], 1.0); self.assertEqual(d["metrics"]["spread_half"], 1)
        self.assertAlmostEqual(d["peak_pos"], 4 / 19); self.assertFalse(d["peak_in_last5"])
        self.assertAlmostEqual(u["metrics"]["win5_share"], 5 / 20); self.assertAlmostEqual(u["metrics"]["peak_share"], 1 / 20)
        self.assertEqual(u["metrics"]["spread_half"], 20); self.assertAlmostEqual(u["metrics"]["gini"], 0.0, places=9)
        self.assertTrue(t["peak_in_last5"]); self.assertAlmostEqual(t["peak_pos"], 1.0)
        self.assertFalse(z["fired"]); self.assertIsNone(z["metrics"]["win5_share"]); self.assertEqual(z["n_tokens"], 10)
        self.assertEqual(e["n_tokens"], 0); self.assertFalse(e["fired"])
        self.assertAlmostEqual(lu["crop_win5"], 5 / 32)     # length control: best 32-token crop of a 40-token uniform
        self.assertAlmostEqual(lu["metrics"]["win5_share"], 5 / 40)

    def test_aggregate_locality(self):
        delta = np.zeros(20); delta[19] = 8.0
        uniform = np.ones(20) * 2.0
        rows = {1: IE.locality_rows_from_profiles([delta, uniform], 1.0),
                2: IE.locality_rows_from_profiles([np.zeros(5), np.zeros(5)], 1.0)}
        m = IE.aggregate_locality(rows, {1: 16.0, 2: 10.0})
        self.assertAlmostEqual(m["extra/locality/fire_frac"], 0.5)
        self.assertAlmostEqual(m["extra/locality/feat_fire_frac"], 0.5)
        self.assertAlmostEqual(m["extra/locality/win5_share"], (1.0 + 0.25) / 2)
        self.assertAlmostEqual(m["extra/locality/peak_in_last5_frac"], 0.5)    # delta yes, uniform argmax=0 no
        self.assertAlmostEqual(m["extra/locality/peak_norm_best_mean"], (8 / 16 + 0.0) / 2)
        self.assertEqual(m["extra/locality/n_features"], 2.0); self.assertEqual(m["extra/locality/n_texts"], 4.0)
        empty = IE.aggregate_locality({3: IE.locality_rows_from_profiles([np.zeros(4)], 1.0)}, {3: 1.0})
        self.assertTrue(np.isnan(empty["extra/locality/win5_share"])); self.assertEqual(empty["extra/locality/fire_frac"], 0.0)

    def test_adversarial_metrics(self):
        cp = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
        true = {1: [5.0, 6.0], 2: [1.0, 1.0], 3: [3.0, 3.0], 4: [4.0]}
        naive = {1: [2.0, 2.0], 2: [0.5, 0.5], 3: [4.0, 4.0]}          # feature 4 has no naive arm -> excluded
        m = IE.adversarial_metrics(true, naive, cp)
        self.assertEqual(m["extra/adversarial/n_features"], 3.0)
        # 1: 5.5 > 2 and 5.5 > 2.5 -> holds; 2: 1 > 0.5 but 1 < 2.5 -> no; 3: 3 < 4 -> no
        self.assertAlmostEqual(m["extra/adversarial/holds_frac"], 1 / 3)
        self.assertAlmostEqual(m["extra/adversarial/true_gt_naive_frac"], 2 / 3)
        self.assertAlmostEqual(m["extra/adversarial/true_act_norm_mean"], (0.55 + 0.1 + 0.3) / 3)
        self.assertEqual(IE.adversarial_metrics({}, {}, cp)["extra/adversarial/n_features"], 0.0)

    def test_detect_prompt_matches_offline_template(self):
        p = IE._detect_prompt(["s1", "s2"], "hello")
        self.assertEqual(p, "DESCRIPTION SNIPPETS:\n1. s1\n2. s2\n\nTEST SNIPPET:\n<<<\nhello\n>>>\n\nProbability "
                            "(integer 0-100) that the feature activates on the test snippet:")


class TestJudgeClient(unittest.TestCase):
    def setUp(self):
        self._sleep = IE.time.sleep
        IE.time.sleep = lambda s: None          # no real backoff waits in tests

    def tearDown(self):
        IE.time.sleep = self._sleep

    def test_retry_then_success_and_accounting(self):
        jc, be = fake_judge(fail_first_n_retryable=2)
        txt, usage = jc.complete("sys", "Reply with the integer 7.", 8)
        self.assertEqual(txt, "7"); self.assertEqual(be.calls, 3)
        self.assertEqual(jc.snapshot(), {"calls": 1, "fails": 0, "in": 100, "out": 3, "cost": 0.001})

    def test_non_retryable_fails_fast(self):
        jc, be = fake_judge(fail_all=True)
        with self.assertRaises(IE.JudgeError):
            jc.complete("sys", "u", 8)
        self.assertEqual(be.calls, 1); self.assertEqual(jc.snapshot()["fails"], 1)

    def test_exhausted_retries(self):
        jc, be = fake_judge(fail_first_n_retryable=99)
        with self.assertRaises(IE.JudgeError):
            jc.complete("sys", "u", 8, max_attempts=3)
        self.assertEqual(be.calls, 3)

    def test_price_fallback(self):
        jc, _ = fake_judge()
        jc.model = "anthropic/claude-sonnet-5"
        self.assertAlmostEqual(jc._price(1_000_000, 100_000), 2.0 + 1.0)

    def test_make_judge_from_env_without_keys(self):
        saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
        try:
            self.assertIsNone(IE.make_judge_from_env(verbose=False))
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestStage(unittest.TestCase):
    def setUp(self):
        IE.poll_judge_results()                 # drain anything left by other tests
        self.tmp = tempfile.mkdtemp()
        self.saved = (IE.N_TESTS, IE.N_LIST, IE.NEG_KINDS, IE.JUDGE_EVERY, IE.DISABLED, IE.JUDGE_TIMEOUT_S)
        IE.N_TESTS, IE.N_LIST, IE.NEG_KINDS, IE.JUDGE_EVERY, IE.DISABLED, IE.JUDGE_TIMEOUT_S = 3, [1, 4], ["random", "embnn"], 1, set(), 60.0

    def tearDown(self):
        IE.N_TESTS, IE.N_LIST, IE.NEG_KINDS, IE.JUDGE_EVERY, IE.DISABLED, IE.JUDGE_TIMEOUT_S = self.saved
        IE.poll_judge_results()

    def _assets(self, judge, wildchat=True):
        tb = make_testbed(2)
        feats = [r["feature"] for r in tb["features"]]
        return IE.build_assets(tb, wildchat=make_wildchat(feats) if wildchat else None, judge=judge, naive={},
                               naive_cache_path=os.path.join(self.tmp, "naive.json"), out_dir=os.path.join(self.tmp, "out"),
                               n_rollouts=4)

    def test_run_futures_deadline(self):
        ex = IE._executor()
        futs = {ex.submit(time.sleep, 0.5): ("slow", 0), ex.submit(lambda: 1): ("fast", 0)}
        res, timed_out = IE._run_futures(futs, time.time() + 0.2)
        self.assertTrue(timed_out); self.assertIsInstance(res[("slow", 0)], TimeoutError); self.assertEqual(res[("fast", 0)], 1)

    def test_judge_stage_end_to_end(self):
        jc, be = fake_judge()
        assets = self._assets(jc)
        rollouts = {f: assets["recs"][f]["rollouts_temp"][:4] for f in assets["feats"]}
        th = IE.launch_judge_stage(rollouts, 30, assets, None)
        self.assertIsNotNone(th); th.join(30); self.assertFalse(th.is_alive())
        res = IE.poll_judge_results()
        self.assertEqual(len(res), 1); step, m = res[0]; self.assertEqual(step, 30)
        for k in ("auc_random_n1", "auc_random_n4", "auc_embnn_n1", "auc_embnn_n4"):
            self.assertAlmostEqual(m[f"extra/autointerp/{k}"], 1.0, msg=k)
        self.assertAlmostEqual(m["extra/wildchat/auc_n4"], 1.0); self.assertAlmostEqual(m["extra/wildchat/auc_n1"], 1.0)
        self.assertAlmostEqual(m["extra/wildchat/bal_acc_n4"], 1.0)
        self.assertEqual(m["extra/autointerp/n_features"], 2.0); self.assertEqual(m["extra/judge/n_fail"], 0.0)
        # calls: per feature: detection 2 N x (3 pos + 3 rand + 3 emb) = 18, wildchat 2 N x 8 = 16, rdesc 1, adv 1 = 36
        #        + naive (desc 1 + texts 1) per feature = 2  -> stage counts 36/feat, naive is accounted separately
        self.assertEqual(m["extra/judge/n_calls"], 2 * 36.0)
        self.assertAlmostEqual(m["extra/cost_usd"], 2 * 36 * 0.001)
        self.assertEqual(be.calls, 2 * 36 + 2 * 2)
        self.assertAlmostEqual(m["extra/cost_usd_total"], be.calls * 0.001)
        self.assertEqual(m["extra/adversarial/n_generated"], 2.0); self.assertEqual(m["extra/adversarial/n_naive_ready"], 2.0)
        # pending adversarial texts for the next GPU stage (world=1 broadcast path)
        pend = IE._broadcast_pending(assets, 0, 1)
        self.assertEqual(len(pend), 1); item = pend[0]; self.assertEqual(item["src_ckpt_step"], 30)
        self.assertEqual(sorted(item["true"]), assets["feats"]); self.assertEqual(len(item["true"][1000]), 4)
        self.assertEqual(len(item["naive"][1001]), 4)
        self.assertEqual(IE._broadcast_pending(assets, 0, 1), [])           # drained
        # naive cache persisted + reused: a second stage makes no naive calls
        self.assertTrue(os.path.exists(assets["naive_cache_path"]))
        cached = IE._load_naive_cache(assets["naive_cache_path"])
        self.assertEqual(sorted(cached), assets["feats"]); self.assertEqual(len(cached[1000]["texts"]), 4)
        n0 = be.calls
        th = IE.launch_judge_stage(rollouts, 40, assets, None); th.join(30)
        self.assertEqual(be.calls - n0, 2 * 36)
        self.assertEqual([s for s, _ in IE.poll_judge_results()], [40])
        # artifacts written
        self.assertTrue(os.path.exists(os.path.join(assets["out_dir"], "judge_ckpt30.json")))
        with open(os.path.join(assets["out_dir"], "judge_ckpt40.json")) as fh:
            art = json.load(fh)
        self.assertEqual(art["rollout_desc"]["1000"], "Text about zebras and their stripes.")

    def test_judge_stage_without_wildchat_and_with_disable(self):
        jc, be = fake_judge()
        IE.DISABLED = {"adversarial"}
        assets = self._assets(jc, wildchat=False)
        rollouts = {f: assets["recs"][f]["rollouts_temp"][:4] for f in assets["feats"]}
        IE.launch_judge_stage(rollouts, 5, assets, None).join(30)
        (_, m), = IE.poll_judge_results()
        self.assertNotIn("extra/wildchat/auc_n4", m); self.assertNotIn("extra/adversarial/n_generated", m)
        self.assertEqual(m["extra/judge/n_calls"], 2 * 18.0); self.assertEqual(be.calls, 2 * 18)
        self.assertEqual(IE._broadcast_pending(assets, 0, 1), [])

    def test_failures_become_uninformative_scores(self):
        jc, be = fake_judge(fail_all=True)
        IE.DISABLED = {"adversarial", "wildchat"}
        assets = self._assets(jc, wildchat=False)
        rollouts = {f: assets["recs"][f]["rollouts_temp"][:4] for f in assets["feats"]}
        IE.launch_judge_stage(rollouts, 1, assets, None).join(30)
        (_, m), = IE.poll_judge_results()
        self.assertEqual(m["extra/judge/fail_frac"], 1.0)
        self.assertAlmostEqual(m["extra/autointerp/auc_random_n1"], 0.5)   # all 50s -> chance

    def test_judge_every_and_last_rollouts_fallback(self):
        jc, _ = fake_judge()
        IE.JUDGE_EVERY = 2
        IE.DISABLED = {"adversarial", "wildchat", "autointerp"}
        assets = self._assets(jc, wildchat=False)
        rollouts = {f: ["zebra"] for f in assets["feats"]}
        self.assertIsNone(IE.launch_judge_stage(None, 3, assets, None))        # nothing stored yet
        assets["last_rollouts"] = {"ckpt_step": 3, "rollouts": rollouts}
        th1 = IE.launch_judge_stage(None, 3, assets, None); self.assertIsNotNone(th1); th1.join(30)
        self.assertIsNone(IE.launch_judge_stage(rollouts, 4, assets, None))    # every-2nd skip
        th3 = IE.launch_judge_stage(rollouts, 5, assets, None); self.assertIsNotNone(th3); th3.join(30)
        self.assertEqual(sorted(s for s, _ in IE.poll_judge_results()), [3, 5])
        self.assertIsNone(IE.launch_judge_stage(rollouts, 6, None, None))      # assets None -> no-op

    def test_stage_cap(self):
        jc, be = fake_judge(latency=0.05)
        IE.DISABLED = {"adversarial", "wildchat"}
        assets = self._assets(jc, wildchat=False)
        rollouts = {f: assets["recs"][f]["rollouts_temp"][:4] for f in assets["feats"]}
        ths = [IE.launch_judge_stage(rollouts, s, assets, None) for s in (10, 20, 30)]
        self.assertIsNotNone(ths[0]); self.assertIsNotNone(ths[1]); self.assertIsNone(ths[2])
        for th in ths[:2]:
            th.join(60)
        res = dict(IE.poll_judge_results())
        self.assertEqual(res[30], {"extra/judge/skipped": 1.0}); self.assertIn(10, res); self.assertIn(20, res)
        self.assertTrue(IE.wait_for_judge_stages(5))

    def test_prepare_never_raises(self):
        class A:
            extra_testbed = os.path.join(self.tmp, "missing.json")
            eval_sae = os.path.join(self.tmp, "missing.pt")
            save_dir = self.tmp
        self.assertIsNone(IE.prepare_extra_eval_assets(A(), "cpu", 0, 1, True))
        with open(A.extra_testbed, "w") as fh:
            json.dump(make_testbed(1), fh)                                   # testbed ok, SAE missing -> still None
        self.assertIsNone(IE.prepare_extra_eval_assets(A(), "cpu", 0, 1, True))


class TestSubSAE(unittest.TestCase):
    def test_slice_matches_full_encode(self):
        import torch
        torch.manual_seed(0)
        d, Fn = 16, 40
        W_enc, b_enc, b_dec = torch.randn(d, Fn), torch.randn(Fn), torch.randn(d)

        class Full:
            pass
        full = Full(); full.W_enc, full.b_enc, full.b_dec = W_enc, b_enc, b_dec
        feats = [3, 17, 39]
        sub = IE.SubSAE.from_sae(full, feats, "cpu")
        h = torch.randn(2, 5, d)
        want = torch.relu((h - b_dec) @ W_enc[:, [17, 39]] + b_enc[[17, 39]])
        self.assertTrue(torch.allclose(sub.encode_features(h, [17, 39]), want, atol=1e-5))
        self.assertTrue(torch.allclose(sub.enc_dirs([3]), torch.nn.functional.normalize(W_enc[:, [3]].T, dim=-1)))
        # from_file on a dictionary_learning-style state dict (encoder.weight is [F, d])
        p = os.path.join(tempfile.mkdtemp(), "ae.pt")
        torch.save({"encoder.weight": W_enc.T.contiguous(), "encoder.bias": b_enc, "decoder.weight": W_enc,
                    "bias": b_dec}, p)
        sub2 = IE.SubSAE.from_file(p, feats, "cpu")
        self.assertTrue(torch.allclose(sub2.encode_features(h, feats), sub.encode_features(h, feats), atol=1e-6))



class TestGpuStageFakes(unittest.TestCase):
    """run_extra_evals_gpu with a fake vLLM engine + fake clean-base profiler (world=1): request construction,
    sharding/decoding, locality aggregation, pending adversarial scoring, results queue, assets['last_rollouts'] —
    everything except CUDA."""

    def setUp(self):
        IE.poll_judge_results()
        self.saved_mods = {k: sys.modules.get(k) for k in ("vllm", "vllm.lora", "vllm.lora.request")}
        vllm, lora, req = types.ModuleType("vllm"), types.ModuleType("vllm.lora"), types.ModuleType("vllm.lora.request")

        class SamplingParams:
            def __init__(self, **kw):
                self.kw = kw

        class LoRARequest:
            def __init__(self, lora_name, lora_int_id, lora_path):
                self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path
        vllm.SamplingParams, req.LoRARequest, vllm.lora, lora.request = SamplingParams, LoRARequest, lora, req
        sys.modules.update({"vllm": vllm, "vllm.lora": lora, "vllm.lora.request": req})
        self._profiles = IE._profiles

        def fake_profiles(texts, feats, actor, tok, device, subsae, batch=32):
            out = []
            for t in texts:                      # 'hot' -> peak 8 on the LAST of 12 tokens; 'warm' -> 2 mid; else flat 0
                p = np.zeros(12, np.float32)
                if "hot" in t:
                    p[-1] = 8.0
                if "warm" in t:
                    p[3] = 2.0
                out.append(p)
            return out
        IE._profiles = fake_profiles

    def tearDown(self):
        IE._profiles = self._profiles
        for k, v in self.saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        IE.poll_judge_results()

    def test_gpu_stage_world1(self):
        import torch
        tb = make_testbed(3)
        assets = IE.build_assets(tb, judge=None, naive={}, out_dir=tempfile.mkdtemp(), n_rollouts=2)
        assets["dirs"], assets["subsae"] = torch.randn(3, 8), object()
        assets["pending_adv"].put({"src_ckpt_step": 10, "true": {1000: ["hot a", "hot b"], 1001: ["cold", "cold"]},
                                   "naive": {1000: ["cold", "warm"], 1001: ["hot", "hot"]}})

        class LLM:
            calls = []

            def generate(self, reqs, params, lora_request=None, use_tqdm=True):
                self.calls.append((reqs, params, lora_request))
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[100 + k, 7 if s == 0 else 8, 0, 99])
                                                       for s in range(p.kw["n"])]) for k, p in enumerate(params)]

        class Tok:
            def decode(self, ids, skip_special_tokens=True):
                return " ".join("hot" if i == 7 else ("cold" if i == 8 else str(i)) for i in ids if i != 0)

        steer_calls = []

        def steer_fn(v, hnorm, marker):
            steer_calls.append((tuple(v.shape), hnorm, marker))
            return ("sv", marker)

        def trim_fn(g, eos):
            out = []
            for t in g:
                out.append(t)
                if t in eos:
                    break
            return out

        class A:
            max_new_tokens = 96
        llm = LLM()
        out = IE.run_extra_evals_gpu(llm, "actor", None, Tok(), [1, 2, 3], 2, A(), "cpu", 20, 21, 0, 1, assets, steer_fn,
                                     lambda actor, sub, prompt, marker, device, adapter=True: 123.0, {0}, trim_fn)
        self.assertNotIn("error", out)
        reqs, params, lora = llm.calls[0]
        self.assertEqual(len(reqs), 3); self.assertEqual(reqs[0], {"prompt_token_ids": [1, 2, 3]})
        self.assertEqual((lora.lora_path, lora.lora_int_id, lora.lora_name), ("/tmp/rl_lora/rank0/step21", 22, "step21"))
        kw = params[0].kw
        self.assertEqual((kw["n"], kw["temperature"], kw["max_tokens"], kw["min_tokens"], kw["stop_token_ids"], kw["top_k"]),
                         (2, 1.0, 64, 16, [0], 0))
        self.assertEqual(kw["extra_args"], {"apply_steering_vectors": [("sv", 2)]})
        self.assertEqual(steer_calls[0], ((8,), 123.0, 2)); self.assertEqual(len({p.kw["seed"] for p in params}), 3)
        rl = assets["last_rollouts"]
        self.assertEqual(rl["ckpt_step"], 20); self.assertEqual(rl["rollouts"][1000], ["100 hot", "100 cold"])
        self.assertEqual(rl["rollouts"][1002], ["102 hot", "102 cold"])
        self.assertAlmostEqual(out["extra/locality/fire_frac"], 0.5); self.assertAlmostEqual(out["extra/locality/feat_fire_frac"], 1.0)
        self.assertAlmostEqual(out["extra/locality/peak_in_last5_frac"], 1.0); self.assertAlmostEqual(out["extra/locality/win5_share"], 1.0)
        self.assertAlmostEqual(out["extra/locality/peak_norm_best_mean"], 0.8); self.assertEqual(out["extra/locality/n_texts"], 6.0)
        self.assertEqual(out["extra/adversarial/n_pending_scored"], 1.0); self.assertIn("time/extra_eval_gpu_s", out)
        res = dict(IE.poll_judge_results())                       # previous checkpoint's adversarial scores, under ITS step
        self.assertEqual(list(res), [10]); m = res[10]
        self.assertEqual(m["extra/adversarial/n_features"], 2.0)
        self.assertAlmostEqual(m["extra/adversarial/holds_frac"], 0.5)   # 1000: 8 > 1 and 8 > 2.5; 1001: 0 < 8
        self.assertAlmostEqual(m["extra/adversarial/true_act_norm_mean"], (0.8 + 0.0) / 2)
        self.assertEqual(IE._broadcast_pending(assets, 0, 1), [])
        self.assertTrue(os.path.exists(os.path.join(assets["out_dir"], "rollouts_ckpt20.json")))
        # a failing rank reports the error as data instead of raising
        bad = IE.run_extra_evals_gpu(llm, "actor", None, Tok(), [1, 2, 3], 2, A(), "cpu", 30, 31, 0, 1, assets,
                                     lambda *a: 1 / 0, lambda *a, **k: 1.0, {0}, trim_fn)
        self.assertIn("error", bad); self.assertIn("ZeroDivisionError", bad["error"])
        self.assertEqual(IE.run_extra_evals_gpu(None, None, None, None, None, None, None, None, 0, 0, 0, 1, None, None, None, None, None), {})


if __name__ == "__main__":
    unittest.main()
