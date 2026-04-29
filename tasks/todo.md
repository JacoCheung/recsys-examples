# HSTU CP v0 — TODO

Flat checklist. Drives `tasks/plan.md`. Tick as we go; checkpoints are
**owner sign-off** moments — do not start the next task until the prior
checkpoint is signed.

**Global rules** (apply to every task):
- Reference-driven: compare against single-GPU baseline from T0.1.
- No functional regression: `bash examples/hstu/cp/run_regression.sh` exits 0
  (fail-on-skip for required multi-GPU tests).
- No perf regression — tiered:
  - Phase 0: same-commit `bench/baseline.py` re-runs within 5%.
  - Slice 3+: cp=1 passthrough vs unwrapped baseline within +10% (shapes
    < 1ms) / +5% (shapes ≥ 1ms).
- Boundary coverage required: smallest legal shape `[2*cp_size]`, padding-
  heavy varlen, at least one cell at `head_dim=256` (full kernel-supported
  range). **No sliding-causal in v0** (dropped from scope; see SPEC §2).
- Tests are additive: never delete a prior test.
- Runtime authority: installed `hstu` package signature is source of truth.

---

## Phase 0 — Reference infrastructure

- [x] **T0.1**: Reference test harness
  - File: `examples/hstu/test/cp/conftest.py` + `test_reference.py` (new)
  - Acceptance: `pytest examples/hstu/test/cp/test_reference.py -v` PASS; ≥ 12 boundary tuples
- [x] **T0.2**: Reference benchmark harness + commit `tasks/bench_baseline.json`
  - File: `examples/hstu/cp/bench/baseline.py` + `bench/compare.py` (new)
  - Acceptance: same-commit double-run via `compare.py` exits 0; ≥ 6 shapes
- [x] **T0.3**: One-shot regression command
  - File: `examples/hstu/cp/run_regression.sh` (new)
  - Acceptance: exits 0 in < 60s on clean checkout

### ✅ Checkpoint 0 — reference infra ready

- [x] Owner sign-off (verified 2026-04-29: 65/65 reference tests pass; bench_baseline.json committed)

---

## Phase 1 — PoC generalisation (Slice 2)

- [x] **S1**: PoC equal-len cp=2 fwd PASS (max |diff|=1.95e-3, bf16) — done before SPEC
- [x] **T2.0**: Verify or revert in-progress varlen refactor of `poc_dualrank_sim.py` (start from clean base)
- [x] **T2.1**: Generalise PoC to `cp_size > 2` (3-region tile grid for cp_size=4) — equal-len only
  - Acceptance: `--cp-size 4` equal-len PASS; `--cp-size 2` regression PASS
- [x] **T2.2**: PoC supports varlen at cp_size=2
  - Acceptance: varlen `[16,32,48,64]` cp=2 PASS
- [x] **T2.3**: Matrix sweep (cp ∈ {2,4,8}, varlen, causal only; padding-heavy boundaries)
  - Acceptance: all 6 SPEC §3 Slice 2 matrix cells PASS

### ✅ Checkpoint A — math validated; ready for production code

- [x] Owner sign-off (verified 2026-04-29: PoC matrix 6/6 fwd + 6/6
      fwd+bwd PASS on synthetic batches)
- [ ] *(deferred to Slice 6 prereq)* Padding-cost measurement on
      representative recsys seqlens (SPEC §9.1): if `padding/total >
      30 %`, escalate to Track B before wiring into training loop.
      Phase 2/3 are NOT blocked on this; v0 ships on synthetic-batch
      correctness.

---

## Phase 2 — Multi-GPU forward (Slice 3)

- [x] **T3.1**: Public API skeleton + hard guards + cp=1 passthrough
  - File: `examples/hstu/context_parallel/hstu_attn_cp.py` (new)
  - Acceptance: cp=1 path bit-exact match; each rejected input → documented `ValueError`
- [x] **T3.2**: `get_batch_on_this_cp_rank_for_hstu` helper (pure permutation)
  - Acceptance: round-trip identity; per-rank shard size balanced at cp ∈ {2,4,8}
- [x] **T3.3**: Multi-GPU forward, sequential ring P2P, cp_size=2
  - File: `examples/hstu/test/cp/test_cp_forward.py` (new, torchrun)
  - Acceptance: torchrun N=2 cp=2 fwd matches single-GPU (bf16)
- [x] **T3.4**: Scale to cp_size ∈ {4, 8} + varlen (causal only)
  - File: `examples/hstu/cp/run_cp_tests.sh` (new helper)
  - Acceptance: SPEC §3 Slice 3 matrix PASS via torchrun N=2,4,8

### ✅ Checkpoint B — multi-GPU forward correct

- [x] Owner sign-off (verified 2026-04-29 on PCIe; NVLink TBD)

---

## Phase 3 — Multi-GPU backward (Slice 4)

- [x] **T4.1**: PoC backward (autograd in single-rank simulator)
  - Acceptance: SPEC §3 Slice 2 matrix PASS for grads (single-rank oracle)
- [x] **T4.2**: Multi-GPU backward at cp_size=2 (reverse ring)
  - File: `examples/hstu/test/cp/test_cp_backward.py` (new, torchrun)
  - Acceptance: torchrun N=2 cp=2 fwd+bwd grads match single-GPU
- [x] **T4.3**: Scale backward to cp_size ∈ {4, 8} + varlen (causal only)
  - Acceptance: SPEC §3 Slice 4 matrix PASS via torchrun N=2,4,8 with bwd

### ✅ Checkpoint C — v0 correctness done; merge decision

- [x] Owner sign-off (verified 2026-04-29 on PCIe; NVLink TBD)
- [ ] Cut PR(s) — recommended: one per slice (S2/S3/S4)
- [ ] **v0 ships here.** SPEC §8 locks v0 = Slices 1–4. Slice 5 is v0+
      and ships separately; do NOT block v0 merge on Slice 5 perf.

---

## Phase 4 — Overlap + perf (Slice 5; v0+ / v0.5)

- [ ] **T5.1**: Two-stream + double-buffered KV ring
  - Acceptance: Slice 4 matrix still PASS unchanged
- [ ] **T5.2**: `bench_cp.py` perf harness + NSys profile
  - Acceptance: visual check on NSys shows P2P/compute concurrency
- [ ] **T5.3**: Hit perf gate (cp=4 step time ≤ 1.5× single-GPU per-token)
  - Acceptance: `bench_cp.py` numbers meet gate

### ✅ Checkpoint D — perf decision

- [ ] Owner sign-off (Slice 5 not yet started)

---

## Phase 5 — Module / training integration (Slice 6; post-v0, optional)

Re-plan when this phase starts. Sketch:

- [ ] **T6.1**: `HSTUConfig.cp_size` + Megatron parallel-state CP group
- [ ] **T6.2**: `HSTUAttention` module accepts `cp_group`, routes to CP fn
- [ ] **T6.3**: Dataloader DualChunkSwap shuffle
- [ ] **T6.4**: E2E training smoke test

---

## Day-1 next action

Phase 0 in order: T0.1 (reference test harness) → T0.2 (reference
benchmark + commit `bench_baseline.json`) → T0.3 (one-shot regression
command). All three must pass before Checkpoint 0. Owner sign-off at
Checkpoint 0, then T2.0 (PoC cleanup) → Phase 1 → Checkpoint A.

---

## v0 verification status (2026-04-29) ✅

All Phase 0-3 phases verified on a real 8× A100 80GB PCIe node
(`computelab-job-1994669` on `g492-ha0-0004`):

- [x] Phase 0 reference pytest: 65/65 PASS
- [x] T3.1 single-GPU API smoke: 34/34 PASS
- [x] Phase 1 PoC fwd matrix: 6/6 PASS (max |diff| = 2e-3 at bf16)
- [x] Phase 1 PoC fwd+bwd matrix: 6/6 PASS (max |grad| = 2e-3 at bf16)
- [x] Phase 2 multi-GPU forward cp ∈ {2, 4, 8}: 8/8 PASS
- [x] Phase 3 multi-GPU backward cp ∈ {2, 4, 8}: 8/8 PASS
- [x] Bench baseline `tasks/bench_baseline.json` committed
- [x] `bash examples/hstu/cp/run_cp_tests.sh --bwd` ⇒ overall: OK

Deployment requirements documented in SPEC §2 "Deployment quirks":
- `NCCL_P2P_DISABLE=1` on PCIe nodes (CUMEM hang workaround)
- Wrapper imports kernel from installed `hstu` package (FBGEMM-style)
- 4-op `batch_isend_irecv` split to 2× 2-op (NCCL bug)

## Next milestone (post-v0)

Owner picks one of:
- **Slice 5** (perf): two-stream comm/compute overlap + perf gate
- **Slice 6** (post-v0): training-loop integration (HSTUConfig,
  module wiring, dataloader DualChunkSwap)
- **v0.5**: sliding-causal under DualChunkSwap (per-tile window remap)
