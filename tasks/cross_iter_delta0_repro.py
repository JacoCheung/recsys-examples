"""GPU repro for cross_iter Δ=0 silent-wrong-sync — using REAL engine.

User's 1:1 setup:
  fwd.la=1, bwd.la=1, update.la=0, all on stream "default".
  fwd.cross_iter_depends_on=(("update", -1)) — Δ=0.
  Single thread, single stream, single nn.Linear module, SGD lr=1.0.

Two pipelines built via the engine's actual Task / Schedule /
SchedulablePipeline machinery; the only difference is task
DECLARATION order:
  CORRECT: [update, fwd, bwd] — topological_sort places update first
                                in steady state. update.la=0 in P_n
                                applies grad-of-B_n (left in
                                param.grad by P_(n-1)'s bwd of la=1
                                which processed B_n). Then fwd reads
                                post-update w on B_(n+1).
  BUGGY:   [fwd, bwd, update] — topological_sort places fwd first
                                (cross_iter contributes no topo edge).
                                fwd reads PRE-update w, bwd OVERWRITES
                                param.grad with grad-of-B_(n+1)
                                (LOSING grad-of-B_n), update applies
                                the wrong grad.

The Δ=0 validator is monkey-patched out so the bad schedule survives
construction; this is purely to expose the runtime symptom that the
validator now prevents.
"""
import copy
import sys

import torch
import torch.nn as nn

# Bypass commons.pipeline.__init__ → train_pipeline's torch import is fine here
# since we're inside the recsys container which has torch.
sys.path.insert(
    0, "/home/scratch.junzhang_sw/workspace/github/recsys-postproc-cleanup/examples"
)
from commons.pipeline.engine import (
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)
from commons.pipeline.engine import deps as deps_mod

# ------------------------------------------------------------------------
# Bypass Δ=0 cross_iter validator so the buggy schedule can be built.
# ------------------------------------------------------------------------
_original_infer = deps_mod.infer_cross_stream_event_deps


def _patched_infer(schedule):
    try:
        return _original_infer(schedule)
    except ValueError as e:
        msg = str(e)
        if "same_progress_sync" in msg:
            # Δ=0 case rejected — return empty deps (no event triples).
            # This is what the engine would emit if cross_iter Δ=0 had
            # been silently accepted (no GPU sync, fall back to topo +
            # CPU thread ordering).
            return {
                name: ()
                for stage in schedule.stages
                for name in (t.name for t in stage.tasks)
            }
        raise


deps_mod.infer_cross_stream_event_deps = _patched_infer

# ------------------------------------------------------------------------
torch.manual_seed(0)
device = torch.device("cuda")
LR = 1.0
N_PROGRESS = 6

# Identical initial weights for both runs
init_model = nn.Linear(2, 2, bias=False).to(device)
torch.manual_seed(42)
batches = [
    (torch.randn(4, 2, device=device), torch.randn(4, 2, device=device))
    for _ in range(N_PROGRESS + 4)
]


def build_pipeline(declaration_order):
    """Build a SchedulablePipeline with the given task declaration order.
    Returns (pipe, model, logits_log)."""
    model = copy.deepcopy(init_model)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    logits_log = []

    def fwd_fn(ctx):
        batch = ctx.slots["batch_cpu"]
        x, _y = batch
        out = model(x)
        ctx.slots.set("logit", out)
        logits_log.append(out.detach().cpu().clone())

    def bwd_fn(ctx):
        out = ctx.slots["logit"]
        _x, y = ctx.slots["batch_cpu"]
        opt.zero_grad()
        loss = ((out - y) ** 2).mean()
        loss.backward()

    def update_fn(ctx):
        opt.step()

    fwd = Task.from_fn(
        "fwd",
        fwd_fn,
        lookahead=1,
        stream="default",
        reads=("batch_cpu",),
        writes=("logit",),
        cross_iter_depends_on=(("update", -1),),  # Δ=0 — would be rejected
    )
    bwd = Task.from_fn(
        "bwd",
        bwd_fn,
        lookahead=1,
        stream="default",
        reads=("logit", "batch_cpu"),
        depends_on=("fwd",),
    )
    update_task = Task.from_fn(
        "update",
        update_fn,
        lookahead=0,
        stream="default",
    )

    by_name = {"fwd": fwd, "bwd": bwd, "update": update_task}
    tasks = tuple(by_name[n] for n in declaration_order)
    schedule = Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=("default",),
    )
    pool = StreamPool({"default": torch.cuda.default_stream(device)})
    pipe = SchedulablePipeline(schedule, pool)
    return pipe, model, logits_log


# ------------------------------------------------------------------------
# Run both
# ------------------------------------------------------------------------
print("Building pipelines...")
pipe_correct, model_C, logits_C = build_pipeline(["update", "fwd", "bwd"])
pipe_buggy, model_B, logits_B = build_pipeline(["fwd", "bwd", "update"])

# Inspect topo orders that the engine actually computed
from commons.pipeline.engine.deps import topological_sort

correct_order = tuple(t.name for t in topological_sort(pipe_correct._schedule))
buggy_order = tuple(t.name for t in topological_sort(pipe_buggy._schedule))
print(f"Correct topological_sort order: {correct_order}")
print(f"Buggy topological_sort order:   {buggy_order}")
print()

# Drive both with same data
iter_C = iter(batches)
iter_B = iter(batches)
for _ in range(N_PROGRESS):
    pipe_correct.progress(iter_C)
    pipe_buggy.progress(iter_B)

# ------------------------------------------------------------------------
# Compare logits
# ------------------------------------------------------------------------
print(f"Captured {len(logits_C)} correct logits, {len(logits_B)} buggy logits")
print()
print(f"{'idx':>4}  {'logit_correct[0,0]':>22}  {'logit_buggy[0,0]':>22}  {'|Δ|':>14}")
max_delta = 0.0
for i in range(min(len(logits_C), len(logits_B))):
    a, b = logits_C[i], logits_B[i]
    a_v, b_v = a[0, 0].item(), b[0, 0].item()
    delta = abs(a_v - b_v)
    max_delta = max(max_delta, delta)
    print(f"{i:>4}  {a_v:>22.6f}  {b_v:>22.6f}  {delta:>14.6f}")

print()
fro_per_iter = [(a - b).pow(2).sum().sqrt().item() for a, b in zip(logits_C, logits_B)]
print(f"Per-iter ||logit_correct - logit_buggy||_F:")
for i, fn in enumerate(fro_per_iter):
    print(f"  idx {i}: {fn:.6f}")

# Also compare final model weights
print()
for (n_c, p_c), (n_b, p_b) in zip(
    model_C.named_parameters(), model_B.named_parameters()
):
    diff = (p_c - p_b).abs().max().item()
    print(f"  param {n_c}: max |Δ w| = {diff:.6f}")

print()
print(f"Max single-element |logit_correct - logit_buggy| = {max_delta:.6f}")
assert max_delta > 1e-3, (
    f"Expected divergence with cross_iter Δ=0 buggy schedule, "
    f"got max_delta={max_delta}"
)
print("OK — engine-driven cross_iter Δ=0 schedule produces divergent logits.")
