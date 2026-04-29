# HSTU Context Parallelism — user guide

**Status (2026-04-29)**: v0 multi-GPU forward+backward verified on PCIe.
Slice 5 perf and Slice 6 training-loop integration in progress.

This guide tells you how to opt your training run into HSTU CP, what
assumptions you need to satisfy, and how to verify it on your data.

---

## 1. When to use CP

Use CP when you hit either:
- **Memory wall**: per-sample seqlen × heads × head_dim is too large to
  fit attention's quadratic intermediate state on a single GPU.
- **Sequence-length scaling**: you want longer history (8K+ tokens) and
  TP/SP alone don't recover memory.

Don't use CP when:
- You can already fit single-GPU at your target seqlen — pure
  data-parallel scales better and has no CP comm overhead.
- Your training uses heterogeneous mask params (`num_contexts`,
  `num_targets`, `target_group_size > 1`). v0 CP rejects these
  (DualChunkSwap is balanced-chunk; heterogeneous mask breaks load
  balance). Track B (MagiAttention-style chunk dispatch) is the
  follow-on; not in v0.

---

## 2. Hardware requirements

- **NVLink/SXM strongly recommended.** On PCIe nodes the NCCL CUMEM
  P2P transport silently hangs (a verified bug, see SPEC §2 quirk
  #1). The workaround `NCCL_P2P_DISABLE=1` works for correctness but
  forces NCCL Socket transport which is CPU-bound and gives **no
  comm/compute overlap benefit** — Slice 5's two-stream pattern
  is a no-op in that regime.
- A100, H100, H20 (SM 8 / SM 9). The CUTLASS HSTU kernel is the only
  CP-wired backend in v0; Triton and Torch fallbacks raise
  `ValueError` if combined with multi-rank CP.
- Driver / NCCL combo: production container's NCCL has a 4-op
  `batch_isend_irecv` bug (see SPEC §2 quirk #2). The wrapper splits
  K and V into two 2-op batches; no user action required.

---

## 3. Enabling CP — step by step

### 3.1 Megatron parallel-state init

```python
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=pp_size,
    context_parallel_size=cp_size,   # NEW: was implicit before
)
```

### 3.2 Build HSTUConfig

`HSTUConfig.context_parallel_size` is now sourced from
`parallel_state.get_context_parallel_world_size()` (T6.1 fix). No user
code change if you're already using `get_hstu_config(...)`.

### 3.3 Pass cp_group into the attention module

Until Slice 6 T6.3 lands, you must do this step manually at module
construction time:

```python
from megatron.core import parallel_state
from modules.hstu_attention import create_hstu_attention

cp_group = parallel_state.get_context_parallel_group()
cp_global_ranks = list(parallel_state.get_context_parallel_global_ranks())

attn = create_hstu_attention(
    kernel_backend=KernelBackend.CUTLASS,
    num_heads=...,
    attention_dim=...,
    linear_dim=...,
    is_causal=True,                       # CP requires pure-causal in v0
    cp_group=cp_group,
    cp_global_ranks=cp_global_ranks,
    cp_stream=None,                       # None ⇒ wrapper allocates one per device
)
```

When `cp_group is None` or its world size is 1, the module is
bit-identical to the pre-CP path (zero overhead, the legacy kernel is
called directly).

### 3.4 Dispatch the batch with DualChunkSwap (Slice 6 T6.3 — pending)

Once T6.3 lands, the trainer does this for you. Until then, do it
manually before calling the model forward:

```python
from context_parallel import get_batch_on_this_cp_rank_for_hstu

q_loc, k_loc, v_loc, cu_loc, l2g_idx, _ = get_batch_on_this_cp_rank_for_hstu(
    q_global, k_global, v_global, cu_global,
    cp_size=cp_size, cp_rank=cp_rank,
)
```

Per-sample requirement: `seqlen % (2 * cp_size) == 0`. If your data
has heterogeneous seqlens, run
`examples/hstu/cp/bench/padding_cost.py --custom <seqlens>` first to
see the padding overhead. Above 30% padding, escalate to Track B
before wiring CP into your run.

### 3.5 Aggregate the output (Slice 6 T6.3 — pending)

Each rank's CP forward returns its local shard. Until the trainer
handles aggregation, do it manually for loss computation:

```python
import torch.distributed as dist

contrib = torch.zeros_like(q_global, dtype=torch.float32)
contrib[l2g_idx] = out_local.float()
dist.all_reduce(contrib, op=dist.ReduceOp.SUM, group=cp_group)
out_global = contrib.to(q_global.dtype)
# now compute loss against out_global
```

---

## 4. Sanity check before scaling out

Run the regression suite end-to-end on your container before bringing
up a real training job:

```sh
unset PYTHONPATH
bash examples/hstu/cp/run_regression.sh           # single-GPU + multi-GPU
bash examples/hstu/cp/run_cp_tests.sh --bwd        # cp ∈ {2,4,8} fwd+bwd
```

108 tests should pass, 12 expected skips (per-cp_size shard tests
that don't match WORLD_SIZE). On the same node, run the perf bench:

```sh
torchrun --standalone --nproc-per-node 4 \
    examples/hstu/cp/bench/bench_cp.py --cp-size 4 \
    --output /tmp/bench_cp_size4.json
```

Compare the cp=4 median against the cp=1 single-GPU baseline:

```sh
python examples/hstu/cp/bench/bench_cp.py --cp-size 1 \
    --output /tmp/bench_cp_size1.json
```

Plan §Phase 4 perf gate: cp=4 step time should be ≤ 1.5× the cp=1
single-GPU per-token time. **On PCIe this gate is not currently met**;
on NVLink/SXM the verification is pending (see tasks/todo.md
"Next milestone").

---

## 5. Common errors

| Error                                                                 | Cause                                                  | Fix                                                                          |
| --                                                                    | --                                                     | --                                                                           |
| `GuardError: window_size != (-1, 0) not supported in v0`              | sliding-window CP                                      | drop CP, or wait for v0.5 (`docs/cp/v0.5_sliding_causal.md`)                  |
| `GuardError: head_dim N not in {32,64,128,256}`                       | unsupported head dim                                   | reshape, or drop CP                                                          |
| `GuardError: per-rank seqlen must be even`                            | `seqlen % (2*cp_size) != 0` after dispatch              | check the dispatcher input; the global rule is `seqlen % (2*cp_size) == 0`   |
| Hang at `r.wait()` on PCIe                                            | NCCL CUMEM bug                                         | `export NCCL_P2P_DISABLE=1`                                                  |
| Hang at `cuda.synchronize()` after a P2P                              | same bug, different surface                            | same fix                                                                     |
| `ValueError: heterogeneous mask params`                               | `num_contextuals != None` or `num_candidates` or `target_group_size > 1` | drop those for the CP path; full support comes via Track B                   |
| `ValueError: Context Parallelism currently requires the CUTLASS …`    | combined CP + Triton/Torch backend                     | switch `kernel_backend=KernelBackend.CUTLASS`                                |

---

## 6. Diagnostics

- `examples/hstu/cp/bench/bench_cp.py` — per-shape cp ∈ {1,2,4,8} medians.
- `examples/hstu/cp/bench/baseline.py` — single-GPU reference numbers.
- `examples/hstu/cp/bench/compare.py` — diff a candidate vs baseline JSON.
- `examples/hstu/cp/bench/padding_cost.py` — DualChunkSwap padding cost
  on a custom or canned seqlen distribution.
- `examples/hstu/cp/run_regression.sh` — one-shot correctness+perf gate.
- `examples/hstu/cp/run_cp_tests.sh --bwd` — multi-GPU correctness matrix.

---

## 7. References

- `docs/cp/SPEC.md` — v0 contract.
- `docs/cp/hstu_cp_design.md` — research / rationale.
- `docs/cp/v0.5_sliding_causal.md` — v0.5 sliding-causal design.
- `tasks/plan.md` — phased implementation plan.
- `tasks/todo.md` — per-task checklist with verification timestamps.
