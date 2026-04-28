"""Directly call infer_cross_stream_event_deps on the bad schedule and
trace which validator path runs."""
import sys

sys.path.insert(
    0, "/home/scratch.junzhang_sw/workspace/github/recsys-postproc-cleanup/examples"
)

from commons.pipeline.engine import Schedule, Stage, Task
from commons.pipeline.engine.deps import infer_cross_stream_event_deps

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
update_task = Task.from_fn("update", lambda ctx: None, lookahead=0, stream="default")

schedule = Schedule(
    stages=(Stage(tasks=(fwd, bwd, update_task)),), stream_slots=("default",)
)

print("Calling infer_cross_stream_event_deps directly:")
print(f"  fwd.lookahead = {fwd.batch_offset}")
print(f"  fwd.cross_iter_depends_on = {fwd.cross_iter_depends_on}")
print(f"  update.lookahead = {update_task.batch_offset}")
print(f"  fwd.stream = {fwd.stream}, update.stream = {update_task.stream}")
print()

try:
    deps = infer_cross_stream_event_deps(schedule)
    print(f"Result: {deps}")
    print("(no raise — investigate why)")
except ValueError as e:
    print(f"RAISED: {e}")
