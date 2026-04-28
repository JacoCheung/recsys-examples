"""Same as cross_iter_delta0_repro.py but WITHOUT the validator
bypass. Demonstrates the engine REJECTS the bad schedule at
SchedulablePipeline construction time."""
import sys

sys.path.insert(
    0, "/home/scratch.junzhang_sw/workspace/github/recsys-postproc-cleanup/examples"
)

import torch
import torch.nn as nn
from commons.pipeline.engine import (
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)

torch.manual_seed(0)
device = torch.device("cuda")
init_model = nn.Linear(2, 2, bias=False).to(device)

fwd = Task.from_fn(
    "fwd",
    lambda ctx: None,
    lookahead=1,
    stream="default",
    reads=("batch_cpu",),
    writes=("logit",),
    cross_iter_depends_on=(("update", -1),),
)
bwd = Task.from_fn(
    "bwd",
    lambda ctx: None,
    lookahead=1,
    stream="default",
    reads=("logit", "batch_cpu"),
    depends_on=("fwd",),
)
update_task = Task.from_fn(
    "update",
    lambda ctx: None,
    lookahead=0,
    stream="default",
)

schedule = Schedule(
    stages=(Stage(tasks=(fwd, bwd, update_task)),),
    stream_slots=("default",),
)
pool = StreamPool({"default": torch.cuda.default_stream(device)})

print("Constructing SchedulablePipeline (no monkey-patch)...")
try:
    pipe = SchedulablePipeline(schedule, pool)
    print("UNEXPECTED: construction succeeded without raising")
except ValueError as e:
    print(f"REJECTED at construction time:\n  {e}")
