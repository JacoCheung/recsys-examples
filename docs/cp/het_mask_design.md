# Heterogeneous mask under CP — design via the kernel's `func` interface

**Status**: design (2026-04-29). v0 currently rejects `num_contexts`,
`num_targets`, `target_group_size > 1` in both the CP wrapper and the
module dispatcher. This doc replaces the v0 reject-with-GuardError
strategy with a positive design that uses the HSTU kernel's *built-in*
arbitrary-mask facility (`func` parameter) as the unifying source of
truth.

**Motivation**: production HSTU ranking uses heterogeneous masks
(history + target + optional contextual prefix). Without CP support
for these, the v0 wrapper is unusable for any real ranking workload.
Sliding-causal can wait — see `docs/cp/v0.5_sliding_causal.md` for
that track.

**Pre-reads**: SPEC §2 (current v0 reject list), `hstu_cp_design.md`
§3 (DualChunkSwap chunk geometry), kernel sources at
`third_party/FBGEMM/fbgemm_gpu/experimental/hstu/src/hstu_ampere/hstu_fwd.h`
and `hstu_bwd.h` (the FBGEMM submodule is the production source of
truth; the in-tree `corelib/hstu/` is deprecated and intentionally
ignored here).

---

## 1. Kernel `func` interface — what it actually is

The FBGEMM HSTU kernel signature already accepts a `func` parameter
(`Optional[torch.Tensor]`). Reading the C++/CUDA implementation
under `third_party/FBGEMM/fbgemm_gpu/experimental/hstu/src/hstu_ampere/`:

- `hstu_ops_gpu.cpp:198-215`: `func` is `int32`, shape
  `(B, H, NFUNC, max_seqlen_q)` where `NFUNC == HSTU_ARBITRARY_NFUNC`
  (compile-time constant; FBGEMM build default needs to be probed at
  runtime — see open question A).
- The Python docstring says `(nheads, total_q + 256)`; that is wrong —
  the C++ side and the FBGEMM tests
  (`third_party/FBGEMM/fbgemm_gpu/experimental/hstu/test/hstu_test.py:401-409`)
  build the tensor as `(B, H, NFUNC, max_seqlen_q)`.
- The kernel's `mMaxFunc`/`mMinFunc` views (`hstu_fwd.h:155-168`)
  interleave the NFUNC axis: stride `2 * func_ids_stride` along that
  dim. Slot `0` is `MaxFunc[0]`, slot `1` is `MinFunc[0]`, slot `2` is
  `MaxFunc[1]`, slot `3` is `MinFunc[1]`, etc.
- For each `(b, h, q_row)` the kernel reads
  `NFUNC // 2 + 1` upper bounds and `NFUNC // 2` lower bounds,
  encoding up to **`NFUNC // 2 + 1` disjoint K intervals**:
    - Interval `0` is implicitly `[0, func[b, h, 0, q_row])`.
    - Interval `k ≥ 1` is
      `[func[b, h, 2k-1, q_row], func[b, h, 2k, q_row])`.
- `hstu_fwd.h:543-558` evaluates `(q_row, col)`:
  ```cpp
  non_mask = (0 <= col) && (col < col_max[0]);
  if (non_mask) continue;            // in window: keep score
  for (j = 0; j < size<0>(gMinFunc); ++j) {
      non_mask = (col_min[j] <= col) && (col < col_max[j+1]);
      if (non_mask) break;
  }
  if (!non_mask) tSrS_view = -INFINITY;  // out of window: drop score
  ```
  Cells inside any of the encoded intervals are kept, others zeroed.
- Backward (`hstu_bwd.h:152-168`) reads the **same** `MaxFunc` /
  `MinFunc` tensor — fwd/bwd "symmetric" (a single mask description
  drives both passes).
- `is_arbitrary_mask = func.defined()` — when `func` is provided,
  the kernel **intersects** the arbitrary intervals with whichever
  structured masks (`Is_causal` / `Is_local` / `Is_target` /
  `Is_context`) are also enabled. Reading `hstu_fwd.h:519-558`:
  structured masks zero out-of-region cells first; the arbitrary
  intervals then zero anything not covered. Net result: a cell is
  kept iff it passes structured AND arbitrary.
- **Practical implication**: callers using `func` to express the
  full mask should pass `window_size=(-1, -1)` (disables both
  `Is_causal` and `Is_local`), set `num_contexts=None`,
  `num_targets=None`, `target_group_size=1`, and encode every
  mask constraint inside the `func` intervals. Otherwise the
  structured layer "double-applies" and over-restricts. Verified
  against FBGEMM tests at
  `third_party/FBGEMM/fbgemm_gpu/experimental/hstu/test/hstu_test.py:401-442`
  where the "emulate causal" / "emulate local" examples set
  `is_causal=False, is_local=False, is_target=False` and put the
  full causal/sliding constraint into `func`.

For `NFUNC = 3`: 2 disjoint intervals max per Q row. Walking the
PT reference (`examples/hstu/ops/pt_ops/pt_hstu_attention.py:46-105`)
shows the standard HSTU mask families never exceed 2 intervals per
Q row — even history+contextual+target_group+sliding combinations
fit. So NFUNC=3 is sufficient for production v1 het-mask under CP.

This is exactly the cell-level mask facility I claimed (incorrectly)
the kernel did not expose. Owner correctly flagged this; the v0.5
sliding design's "remedy 1 — kernel per-row K-mask" is in fact
**already shipping** as `func`.

## 2. Design plan in one sentence

Replace the v0 wrapper's "thread a discrete `(num_contexts,
num_targets, target_group_size, window_size)` tuple" entry with
"thread a `func` tensor (or a `mask_mod`-style callable that
materialises one)". CP integration becomes "build the right `func`
tensor per ring step" — pure host-side work, O(local_seqlen_q ×
cp_size × n_func), trivially cheap vs attention compute.

## 3. Three layers we need

```
                       ┌──────────────────────────────────┐
                       │ Application-layer mask spec      │
                       │   (num_contexts, num_targets,    │
  LAYER 1 (input)      │    target_group_size,            │
                       │    window_size, …)               │
                       └──────────────┬───────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────┐
                       │ mask_mod : (b, q_global, k_global)
  LAYER 2 (predicate)  │     -> bool                      │
                       │ (the "symmetric function")       │
                       └──────────────┬───────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────────┐
  LAYER 3 (kernel)     │ func tensor: per-Q-row union of  │
                       │   ≤ NFUNC K-intervals (LOCAL K)  │
                       └──────────────────────────────────┘
```

Layer 1 is what users (and the existing module call sites) speak.
Layer 3 is what the kernel speaks. **Layer 2 is the bridge**: a pure
predicate over global token positions that does NOT know about CP,
DualChunkSwap, or which rank's K is currently in the ring slot.

CP-awareness lives in the **Layer 2 → Layer 3** materialisation step:
- Each rank knows its local Q rows' global positions
  (`local_to_global` from DualChunkSwap dispatch).
- Each ring step's K is a peer's chunk pair; rank knows that peer's
  global chunk indices `{src, 2cp-1-src}` and therefore each local-K
  position's global index.
- For each `(local_q_row, current_step_local_k_layout)` pair, scan
  the K positions whose `mask_mod(q_global, k_global) == True`,
  group consecutive runs into intervals (≤ NFUNC of them), pack into
  the `func` tensor for this kernel call.

## 4. Open questions (need owner decision before coding)

### A. `HSTU_ARBITRARY_NFUNC` value at runtime — RESOLVED

The production container's pre-installed `hstu` package
(`/usr/local/lib/python3.12/dist-packages/hstu/`) was built with
`HSTU_ARBITRARY_NFUNC=0` and `HSTU_DISABLE_ARBITRARY=FALSE` —
runtime probe (passing a small `func` tensor to
`hstu_attn_varlen_func`) raises:

  > RuntimeError: This hstu attention build does not support
  > arbitrary mask.

(Verified 2026-04-29 against the existing PCIe container
`computelab-job-1994669`.)

**Resolution (chosen 2026-04-29)**: rebuild the FBGEMM hstu kernel
from `third_party/FBGEMM/fbgemm_gpu/experimental/hstu/setup.py`
inside the container with
`HSTU_ARBITRARY_NFUNC=3 HSTU_ARCH_LIST=8.0
 HSTU_DISABLE_ARBITRARY=FALSE` (the default already has
`DISABLE_ARBITRARY=FALSE`; the only required override is
`ARBITRARY_NFUNC=3`). `pip install --user .` overlays
`~/.local/lib/python3.12/site-packages/hstu/` which precedes the
system path → `import hstu` resolves to the rebuilt module. NFUNC
must be odd (`setup.py:243-244`); 3 is the smallest legal value
that supports the 2-interval ceiling HSTU's standard masks need.

**Followup**: a permanent fix is to update the production container
build to enable arbitrary-mask by default; this branch's user-site
overlay is a stopgap for development. Document the rebuild step in
`USER_GUIDE.md` until the production image picks it up.

### B. Layer 2 interface shape — `mask_mod` callable vs explicit `func` tensor

Two viable surfaces:

1. **`mask_mod(b, q_global, k_global) -> bool`**: high-level
   FlexAttention-style. Wrapper traces it on a host-side index grid
   to produce the `func` tensor per ring step. Pros: caller writes
   one Python callable, doesn't have to think about NFUNC encoding.
   Cons: tracing requires either (a) calling Python `mask_mod` per
   `(q_global, k_global)` cell, slow, or (b) symbolically handling
   only known mask families (causal, sliding, target-group, …) —
   then there is no real callable freedom.
2. **`func: Optional[torch.Tensor]`**: low-level. Caller (or a
   helper) builds the int32 tensor directly. CP wrapper takes the
   *full-batch* `func` tensor at the global Q×K geometry, then
   per-step **slices/remaps** it down to local Q × current-step
   local K. Pros: zero tracing cost; user has full control. Cons:
   user has to understand the encoding.

**Recommendation**: ship Layer 3 (explicit `func` tensor) as the
primary CP wrapper input. Provide a `build_hstu_mask_func(...)`
helper that turns the existing 4-tuple into a global `func` tensor
for the common HSTU mask families (causal, contextual prefix,
target groups, sliding). Layer 2 (mask_mod callable) is a future
abstraction layered on top — out of scope here.

**Question for owner**: agree with this layering? If you want
`mask_mod` callable as the user-visible API, we add ~200 LOC
trace-and-encode logic that has to be tested separately.

### C. Backwards compatibility with existing
`(num_contexts, num_targets, target_group_size, window_size)` callers

The wrapper's signature is pinned in
`examples/hstu/test/cp/conftest.py::CANONICAL_HSTU_PARAMS` against
the installed kernel. Adding a `func` arg is fine (kernel already
has one). But the existing 4-tuple needs to keep working for v0
non-CP callers; the v0 hard-guard list rejects it for CP today.
After this work, CP path can also take the 4-tuple (we internally
build the corresponding `func` tensor).

**Plan**: keep the 4-tuple args in the wrapper signature, but
inside the CP path translate them via `build_hstu_mask_func(...)`
into a `func` tensor at call time. If the caller passes BOTH the
4-tuple AND a `func` tensor, prefer `func` and ignore the 4-tuple
(matching the kernel's behavior). 4-tuple-only callers see no
signature change. `func`-only callers get the new path. Mixed
callers get a deterministic resolution.

**Question for owner**: agreed?

### D. "Symmetric" interpretation

Owner's note said "this function is symmetric". My reading: fwd and
bwd kernels read the same `MaxFunc`/`MinFunc` tensor, so a single
mask description drives both passes — no separate bwd mask.
Verified at `hstu_bwd.h:139-148` (same global memory pointer +
stride layout as fwd). Codex round-1 review of any code that
ships will revisit this.

If "symmetric" was meant differently (e.g. FlexAttention-style
`(i, j)` argument symmetry), the design above still holds; only the
recommended Layer 2 interface name changes.

---

## 5. Per-step `func` tensor builder — algorithm sketch

For a single CP layer's forward call at ring step `s` on rank `r`:

Inputs:
- `cp_size`, `r`, `s`
- Per-sample chunk size `c = L_b / (2 * cp_size)`
- `local_to_global_q[2*c]`: global token indices held as Q on rank `r`
  (from `apply_dualchunkswap_to_jagged`). Layout per sample:
  `[chunk_r, chunk_(2cp-1-r)]`.
- `peer = (r - s) % cp_size` (forward ring direction)
- `local_to_global_k[2*c]`: peer's chunks `{peer, 2cp-1-peer}` global
  indices. Same layout convention.
- Global mask spec (from translated 4-tuple or user-supplied):
  `mask_predicate(q_global, k_global) -> bool`

Output: int32 tensor of shape `(B, H, n_func, 2*c_local)` packed
into the `(MaxFunc, MinFunc)` layout the kernel expects.

Pseudocode:

```python
def build_step_func(local_to_global_q, local_to_global_k,
                    mask_predicate, n_func: int = 3,
                    n_heads: int = 1):
    # Fast path: head-uniform mask (HSTU's stock masks all are).
    # We materialise once for h=0 and broadcast the result over heads.
    func = torch.full(
        (B, n_heads, 2 * n_func, max_local_q),
        fill_value=...,      # see encoding below
        dtype=torch.int32,
    )
    for b in range(B):
        for q_idx in range(local_q_b):
            q_global = local_to_global_q[b][q_idx]
            # Walk current-step local K rows, find "in-window" runs
            allowed: list[(int, int)] = []  # (min_local, max_local)
            run_start = None
            for k_idx in range(local_k_b):
                k_global = local_to_global_k[b][k_idx]
                if mask_predicate(b, q_global, k_global):
                    if run_start is None:
                        run_start = k_idx
                else:
                    if run_start is not None:
                        allowed.append((run_start, k_idx))
                        run_start = None
            if run_start is not None:
                allowed.append((run_start, local_k_b))
            if len(allowed) > n_func:
                # Compile-time NFUNC exceeded; fall back to widest-cover.
                # See open question A.
                raise ValueError(...)
            # Encode into MaxFunc / MinFunc per kernel convention.
            ...
    return func
```

The double loop is `O(2c × 2c)` per sample. For `c = 128`, that is
65k cells per sample per step — measurable but small. If it
becomes a bottleneck the inner loop can be vectorised or moved to
a tiny CUDA kernel.

## 6. Test plan

1. **Unit (CPU/single-GPU)**: `build_hstu_mask_func` against the
   `_get_valid_attn_mask` PT reference at
   `examples/hstu/ops/pt_ops/pt_hstu_attention.py` for each
   mask-family combination (causal-only, +contextual, +targets,
   +group_size, +sliding). Assert the produced `func` tensor when
   passed to the kernel reproduces the same output as passing the
   4-tuple.

2. **Per-step builder unit**: round-trip a known global mask
   through `build_step_func` for every ring step at cp ∈ {2, 4, 8}
   and verify the in-window runs match the global predicate.

3. **Multi-GPU correctness**: extend `test_cp_forward.py` and
   `test_cp_backward.py` matrices with cells where `num_targets`,
   `target_group_size`, `num_contexts` are non-trivial. Compare
   against the single-GPU baseline (which passes the 4-tuple
   directly to `hstu_attn_varlen_func`). This is the necessary
   correctness gate.

4. **NFUNC exceeded**: a guard test that constructs a mask
   requiring > NFUNC intervals and asserts the wrapper raises a
   clear `ValueError` rather than silently truncating.

## 7. What this commit set will NOT do

- Sliding-causal (still kernel-blocked unless NFUNC supports it,
  which it does — but per `docs/cp/v0.5_sliding_causal.md` we
  defer that to a separate slice).
- Backwards-incompatible signature changes; the existing 4-tuple
  keeps working.
- FlexAttention-style `mask_mod` callable input. v1 takes the
  explicit `func` tensor + the 4-tuple translator only.

## 8. Rollout sequence

1. **Refactor wrapper entry**: add `func: Optional[torch.Tensor]`
   to `hstu_attn_varlen_cp_func`, plumb through `_HSTUVarlenCPFunc`,
   thread to per-tile kernel calls. cp_size==1 short-circuit just
   passes `func` to the underlying kernel unchanged.
2. **Add `build_hstu_mask_func` translator**: takes
   `(num_contexts, num_targets, target_group_size, window_size,
   cu_seqlens)` → `(B, H, n_func, max_seqlen_q)` int32 tensor.
   Single-GPU only; no CP awareness.
3. **Add per-step builder**: takes the global mask predicate (or
   the global func tensor) + per-step Q/K layouts → per-step local
   func tensor.
4. **Wire into ring**: `_multi_gpu_forward` and `_multi_gpu_backward`
   each per-tile kernel call now uses the per-step func instead of
   the discrete 4-tuple.
5. **Drop the het-mask hard-guards**: `_enforce_v0_contract` no
   longer rejects `num_contexts`/`num_targets`/`target_group_size`;
   instead the wrapper translates them via `build_hstu_mask_func`.
6. **Tests + Codex review** per slice.

The first three steps are non-CP changes (kernel refactor + helper)
that are independently testable. The last three are the CP
integration.
