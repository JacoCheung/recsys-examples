"""Probe: for the HSTU prefetch schedule, what does
producers_with_cross_stream_consumers return? Which tasks WOULD have
their events skipped, and is each really only consumed same-stream?
"""
import sys

sys.path.insert(
    0, "/home/scratch.junzhang_sw/workspace/github/recsys-examples/examples"
)

import torch
from commons.pipeline.engine.deps import infer_cross_stream_event_deps
from commons.pipeline.hstu_pipeline import HSTUPipeline


def main():
    pipe = HSTUPipeline(
        model=torch.nn.Linear(4, 4),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        device=torch.device("cpu"),
        prefetch=True,
        prefetch_depth=1,
    )
    schedule, _ = pipe._build_schedule()

    print("=" * 60)
    print("Tasks (declaration order) — HSTU prefetch variant")
    print("=" * 60)
    for stage in schedule.stages:
        for task in stage.tasks:
            reads = ",".join(f"{s.name}@{s.batch_offset}" for s in task.reads)
            writes = ",".join(f"{s.name}@{s.batch_offset}" for s in task.writes)
            deps = ",".join(task.depends_on)
            print(
                f"  {task.name:30s}  stream={task.stream:10s}  "
                f"lookahead={task.batch_offset}  reads=({reads})  "
                f"writes=({writes})  depends_on=({deps})"
            )

    print()
    print("=" * 60)
    print(
        "Cross-stream event_deps (consumer -> [(producer, stream, slot_offset), ...])"
    )
    print("=" * 60)
    deps = infer_cross_stream_event_deps(schedule)
    for consumer, triples in deps.items():
        print(f"  consumer {consumer}:")
        for p, s, off in triples:
            print(f"    wait_event({p} @stream={s} slot[{off}])")

    print()
    print("=" * 60)
    print("producers_to_record set (would record event)")
    print("=" * 60)
    record_set = set()
    for triples in deps.values():
        for p, _, _ in triples:
            record_set.add(p)
    print(f"  {sorted(record_set)}")

    print()
    print("=" * 60)
    print("Tasks that would SKIP event recording with the v3 fix")
    print("=" * 60)
    all_tasks = [t.name for stage in schedule.stages for t in stage.tasks]
    skip_set = [t for t in all_tasks if t not in record_set]
    print(f"  {skip_set}")

    print()
    print("=" * 60)
    print("Sanity: for each skipped task, who consumes its writes/depends?")
    print("=" * 60)
    for stage in schedule.stages:
        for task in stage.tasks:
            if task.name in record_set:
                continue
            consumers = []
            for stage2 in schedule.stages:
                for t2 in stage2.tasks:
                    if t2.name == task.name:
                        continue
                    # slot consumer
                    write_names = {s.name for s in task.writes}
                    read_match = [s.name for s in t2.reads if s.name in write_names]
                    if read_match:
                        consumers.append(
                            (t2.name, t2.stream, "reads " + ",".join(read_match))
                        )
                    # depends_on consumer
                    if task.name in t2.depends_on:
                        consumers.append((t2.name, t2.stream, "depends_on"))
            if not consumers:
                print(f"  {task.name} (stream={task.stream}): NO consumers")
            else:
                same = [c for c in consumers if c[1] == task.stream]
                cross = [c for c in consumers if c[1] != task.stream]
                tag = (
                    "OK same-stream only"
                    if not cross
                    else "BUG: cross-stream consumer present!"
                )
                print(f"  {task.name} (stream={task.stream}): {tag}")
                for name, s, kind in consumers:
                    print(f"    -> {name} (stream={s}, {kind})")


if __name__ == "__main__":
    main()
