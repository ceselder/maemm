"""CPU tests for sft/pretrain.py's multi-part bank loader (--data-dir a,b,c): load_shard / VecBank / TokRows.

    python -m pytest -q sft/test_multibank.py        (or: PYTHONPATH=. python sft/test_multibank.py)

Two tiny synthetic banks A (37 rows, 40 vectors, vecs.f16, vec_idx != line index) and B (23 rows, vecs.f32) against
their MANUAL concatenation C (records appended, vec_idx shifted by A's vector count, vectors appended):
  (1) a single --data-dir reproduces the pre-refactor inline loader tuple-for-tuple (records loop + tokenize_records, copied
      here verbatim from the old code) for world 1/2/3, with and without --max-examples;
  (2) "A,B" == C: identical global row -> (ids, labels, pos, vec_idx) mapping per rank, identical vectors for every row,
      identical lengths; also "A,B,A" (the same part listed twice, the smoke scenario);
  (3) the materialised micro-batches of step_groups (both composition modes, epochs 0/1) are identical, and so is the
      --skip-steps suffix, i.e. resume lands on the same batches;
  (4) VecBank int / list / ndarray indexing == the concatenated memmap, out-of-range raises; TokRows rows == tokenize_records
      rows, lengths() == len(row[0]); count_lines == the line iterator (trailing newline or not, empty file)."""
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

if True:  # transformers binds fla's Triton kernels at import when importable; no Triton driver on a CPU box
    sys.modules.setdefault("fla", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sft import pretrain  # noqa: E402
from sft.pretrain import D_MODEL, TokRows, VecBank, count_lines, gather_rows, load_shard, parse_data_dirs, step_groups  # noqa: E402

WORDS = [f"w{i}" for i in range(300)] + ["hello", "world"]


def fake_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from mxf.prompts import MARKER

    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    vocab.update({w: i + 3 for i, w in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]",
                                  additional_special_tokens=[MARKER])
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %} assistant"
    return tok


def make_bank(path, n_rows, n_vecs, dtype, seed):
    """records.jsonl (vec_idx a random subset of the rows -> NOT the line index; extra keys like the real banks),
    vecs.<dtype> unit rows, build_stats.json."""
    rng = np.random.default_rng(seed)
    os.makedirs(path)
    vecs = rng.standard_normal((n_vecs, D_MODEL)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    fname = "vecs.f16" if dtype == np.float16 else "vecs.f32"
    vecs.astype(dtype).tofile(f"{path}/{fname}")
    vidx = rng.permutation(n_vecs)[:n_rows]
    with open(f"{path}/records.jsonl", "w") as f:
        for i in range(n_rows):
            n_words = int(rng.integers(1, 13))
            text = " ".join(rng.choice(WORDS + ["zzz_unknown", "héllo", "世界"], size=n_words).tolist())
            f.write(json.dumps({"vec_idx": int(vidx[i]), "target_text": text, "family": "realact",
                                "ctx_len": int(rng.integers(8, 256)), "W": n_words, "src": f"r0_w{i}"}) + "\n")
    json.dump({"n_examples": n_rows, "d": D_MODEL}, open(f"{path}/build_stats.json", "w"))


def concat_banks(parts, out):
    """The manual merge the trainer used to need: records appended in order with vec_idx shifted by the vector-row offset
    of the part, vectors appended (as f32 when dtypes mix)."""
    os.makedirs(out)
    mats, voff, n_ex = [], 0, 0
    with open(f"{out}/records.jsonl", "w") as fo:
        for d in parts:
            fname = "vecs.f32" if os.path.exists(f"{d}/vecs.f32") else "vecs.f16"
            dt = np.float32 if fname == "vecs.f32" else np.float16
            m = np.fromfile(f"{d}/{fname}", dtype=dt).reshape(-1, D_MODEL)
            mats.append(m)
            for line in open(f"{d}/records.jsonl"):
                r = json.loads(line)
                r["vec_idx"] += voff
                fo.write(json.dumps(r) + "\n")
                n_ex += 1
            voff += m.shape[0]
    dts = {m.dtype for m in mats}
    if len(dts) == 1:
        dt = dts.pop()
        np.concatenate(mats).astype(dt).tofile(f"{out}/vecs.f16" if dt == np.float16 else f"{out}/vecs.f32")
    else:
        np.concatenate([m.astype(np.float32) for m in mats]).tofile(f"{out}/vecs.f32")
    json.dump({"n_examples": n_ex, "d": D_MODEL}, open(f"{out}/build_stats.json", "w"))


def legacy_load(data_dir, rank, world, max_examples, max_examples_seed, tok, max_seq):
    """The pre-refactor inline loader (pretrain.main) + tokenize_records, verbatim: list of (ids, labels, pos, vec_idx)."""
    n_lines = sum(1 for _ in open(f"{data_dir}/records.jsonl", "rb"))
    n_use = min(n_lines, max_examples) if max_examples else n_lines
    keep = n_use // world
    if n_use < n_lines:
        chosen = np.sort(np.random.default_rng(max_examples_seed).permutation(n_lines)[:n_use])
        mine = set(chosen[rank::world][:keep].tolist())
        take = mine.__contains__
    else:
        take = lambda i: i % world == rank  # noqa: E731
    records = []
    with open(f"{data_dir}/records.jsonl") as f:
        for i, l in enumerate(f):
            if take(i) and len(records) < keep:
                records.append(json.loads(l))
    vec_path, vec_bytes = next((os.path.join(data_dir, fn), np.dtype(dt).itemsize) for fn, dt in pretrain.VEC_BANK_FILES
                               if os.path.exists(os.path.join(data_dir, fn)))
    n_vecs = os.path.getsize(vec_path) // (D_MODEL * vec_bytes)
    assert all(r["vec_idx"] < n_vecs for r in records)
    vecs, _ = pretrain.open_vec_bank(data_dir, n_vecs)
    from mxf.prompts import build_prompt_ids
    prompt, positions = build_prompt_ids(tok)
    prompt_labels = [-100] * len(prompt)
    rows = []
    for start in range(0, len(records), 4096):
        chunk = records[start:start + 4096]
        targets = tok([r["target_text"] for r in chunk], add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        for record, ids in zip(chunk, targets):
            target = list(ids) + [tok.eos_token_id]
            rows.append(((prompt + target)[:max_seq], (prompt_labels + target)[:max_seq], positions, record["vec_idx"]))
    return rows, vecs


@pytest.fixture(scope="module")
def banks():
    tmp = tempfile.mkdtemp(prefix="multibank_")
    A, B = f"{tmp}/A", f"{tmp}/B"
    make_bank(A, 37, 40, np.float16, seed=1)
    make_bank(B, 23, 23, np.float32, seed=2)
    concat_banks([A, B], f"{tmp}/AB")
    concat_banks([A, B, A], f"{tmp}/ABA")
    return dict(A=A, B=B, AB=f"{tmp}/AB", ABA=f"{tmp}/ABA", tok=fake_tokenizer())


CASES = [(w, r, mx) for w in (1, 2, 3) for r in range(w) for mx in (0, 25)]


def _rows(toks):
    return [toks[i] for i in range(len(toks))]


@pytest.mark.parametrize("world,rank,max_examples", CASES)
def test_single_dir_equals_legacy_loader(banks, world, rank, max_examples):
    tok = banks["tok"]
    for max_seq in (16, 192):
        old_rows, old_vecs = legacy_load(banks["AB"], rank, world, max_examples, 0, tok, max_seq)
        toks, vecs, info = load_shard([banks["AB"]], rank, world, tok, max_seq, max_examples, 0)
        assert _rows(toks) == old_rows and list(toks) == old_rows
        assert len(toks) == len(old_rows) == (min(60, max_examples) if max_examples else 60) // world
        assert np.array_equal(toks.lengths(), np.array([len(r[0]) for r in old_rows], dtype=np.int64))
        idx = [r[3] for r in old_rows]
        assert len(vecs) == old_vecs.shape[0] and len(vecs.parts) == 1
        if idx:
            assert np.array_equal(gather_rows(vecs, idx).numpy(), gather_rows(old_vecs, idx).numpy())
            assert np.array_equal(gather_rows(vecs, idx[0]).numpy(), gather_rows(old_vecs, idx[0]).numpy())


@pytest.mark.parametrize("world,rank,max_examples", CASES)
def test_two_parts_equal_manual_concat(banks, world, rank, max_examples):
    tok = banks["tok"]
    for max_seq in (16, 192):
        t_parts, v_parts, i_parts = load_shard([banks["A"], banks["B"]], rank, world, tok, max_seq, max_examples, 0)
        t_one, v_one, i_one = load_shard([banks["AB"]], rank, world, tok, max_seq, max_examples, 0)
        assert _rows(t_parts) == _rows(t_one)                         # same global row -> (ids, labels, pos, vec_idx)
        assert np.array_equal(t_parts.lengths(), t_one.lengths())
        assert i_parts["n_lines"] == i_one["n_lines"] == 60 and i_parts["n_lines_per_part"] == [37, 23]
        assert i_parts["record_offsets"] == [0, 37, 60] and i_parts["vec_offsets"] == [0, 40, 63]
        assert len(v_parts) == len(v_one) == 63
        idx = t_parts.vidx.tolist()
        if idx:
            assert np.array_equal(gather_rows(v_parts, idx).numpy(), gather_rows(v_one, idx).numpy())
            for j in idx[:5]:
                assert np.array_equal(gather_rows(v_parts, j).numpy(), gather_rows(v_one, j).numpy())


def test_same_part_twice_equals_concat(banks):
    """'A,B,A' (the smoke's realact_short_smoke3 twice) == the manual A+B+A merge."""
    tok = banks["tok"]
    for world, rank in ((1, 0), (2, 1)):
        t3, v3, i3 = load_shard(",".join([banks["A"], banks["B"], banks["A"]]), rank, world, tok, 192)
        t1, v1, _ = load_shard([banks["ABA"]], rank, world, tok, 192)
        assert _rows(t3) == _rows(t1) and i3["n_lines"] == 97 and len(v3) == len(v1) == 103
        idx = t3.vidx.tolist()
        assert np.array_equal(gather_rows(v3, idx).numpy(), gather_rows(v1, idx).numpy())
        assert pretrain.bank_stats_n_examples(i3["dirs"]) == 97


@pytest.mark.parametrize("length_bucket", [False, True])
@pytest.mark.parametrize("world", [1, 2])
def test_batches_and_resume_identical(banks, world, length_bucket):
    """Materialised micro-batches (what the training loop feeds the model) and the --skip-steps suffix are identical."""
    tok = banks["tok"]
    for rank in range(world):
        t_parts, v_parts, _ = load_shard([banks["A"], banks["B"]], rank, world, tok, 192)
        t_one, v_one, _ = load_shard([banks["AB"]], rank, world, tok, 192)
        for ep in (0, 1):
            g_parts = step_groups(t_parts.lengths(), 4, 2, ep, length_bucket)
            g_one = step_groups(t_one.lengths(), 4, 2, ep, length_bucket)
            assert len(g_parts) == len(g_one) == math.ceil(math.ceil(len(t_one) / 4) / 2)

            def materialise(groups, toks, vecs):
                out = []
                for g in groups:
                    for mb in g:
                        rows = [toks[i] for i in mb]
                        out.append((rows, gather_rows(vecs, [r[3] for r in rows]).numpy().tobytes()))
                return out
            m_parts, m_one = materialise(g_parts, t_parts, v_parts), materialise(g_one, t_one, v_one)
            assert m_parts == m_one
            skip = 3
            assert materialise(g_parts[skip:], t_parts, v_parts) == materialise(g_one[skip:], t_one, v_one)


def test_vecbank_indexing_matches_memmap(banks):
    vb = VecBank([banks["A"], banks["B"]])
    ref = np.fromfile(f"{banks['AB']}/vecs.f32", dtype=np.float32).reshape(-1, D_MODEL)
    assert len(vb) == ref.shape[0] == 63 and vb.dtype == np.float32 and vb.files == ["vecs.f16", "vecs.f32"]
    for i in (0, 39, 40, 62):
        assert np.array_equal(np.asarray(vb[i], dtype=np.float32), ref[i])
        assert np.array_equal(np.asarray(vb[np.int64(i)], dtype=np.float32), ref[i])
    idx = [62, 0, 40, 39, 1, 40, 5]
    assert np.array_equal(vb[idx].astype(np.float32), ref[idx])
    assert np.array_equal(vb[np.array(idx)].astype(np.float32), ref[idx])
    assert vb[[]].shape == (0, D_MODEL)
    for bad in (63, -1, [0, 63]):
        with pytest.raises(IndexError):
            vb[bad]
    one = VecBank([banks["B"]])
    assert isinstance(one[[1, 2]], np.ndarray) and one.dtype == np.float32 and one.n == 23
    with pytest.raises(FileNotFoundError):
        VecBank([tempfile.gettempdir() + "/definitely_missing_bank"])


def test_tokrows_rows_equal_tokenize_records(banks):
    tok = banks["tok"]
    from mxf.prompts import build_prompt_ids
    records = [json.loads(l) for l in open(f"{banks['AB']}/records.jsonl")]
    for max_seq in (8, 16, 192):
        expected = pretrain.tokenize_records(records, tok, max_seq, chunk_size=7)
        prompt, positions = build_prompt_ids(tok)
        tr = TokRows(prompt, positions, max_seq)
        texts = [r["target_text"] for r in records]
        for s in range(0, len(records), 11):
            tr.extend(pretrain.tokenize_targets(tok, texts[s:s + 11]), [r["vec_idx"] for r in records[s:s + 11]])
        tr.finalize()
        assert list(tr) == expected and [tr[i] for i in range(len(tr))] == expected and tr[-1] == expected[-1]
        assert tr[np.int64(3)] == expected[3]
        assert np.array_equal(tr.lengths(), np.array([len(r[0]) for r in expected], dtype=np.int64))
        assert tr.nbytes() == tr.flat.nbytes + tr.offs.nbytes + tr.vidx.nbytes
        with pytest.raises(IndexError):
            tr[len(tr)]
        with pytest.raises(AssertionError):
            tr.extend([[1]], [0])
    empty = TokRows(prompt, positions, 192).finalize()
    assert len(empty) == 0 and list(empty) == [] and empty.lengths().shape == (0,)


def test_count_lines_and_parse():
    d = tempfile.mkdtemp()
    for content in (b"", b"a\n", b"a", b"a\nb\n", b"a\nb", b"\n\n", b"x" * (1 << 24) + b"\ny\n"):
        p = f"{d}/f"
        open(p, "wb").write(content)
        assert count_lines(p, chunk=5) == count_lines(p) == sum(1 for _ in open(p, "rb")), content[:20]
    assert parse_data_dirs("a") == ["a"] and parse_data_dirs(" a , b,c,") == ["a", "b", "c"]
    with pytest.raises(AssertionError):
        parse_data_dirs(" , ")


def test_bad_vec_idx_is_rejected(banks):
    tok = banks["tok"]
    d = tempfile.mkdtemp()
    make_bank(f"{d}/X", 5, 5, np.float32, seed=9)
    with open(f"{d}/X/records.jsonl", "a") as f:
        f.write(json.dumps({"vec_idx": 5, "target_text": "hello"}) + "\n")   # row 5 of a 5-row bank
    with pytest.raises(AssertionError, match="beyond the vector bank"):
        load_shard([f"{d}/X"], 0, 1, tok, 192)


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__]))
