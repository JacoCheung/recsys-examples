#!/usr/bin/env python3
"""Export HSTU schedulable-pipeline DAG JSON for a benchmark run.

Stdlib-only by design: this runs before the training container, where torch,
nvtx, and repo imports may be unavailable.
"""

import argparse
import json
import os
from pathlib import Path


def _bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _load_config(value, hstu_root):
    raw = (value or "").strip()
    if not raw:
        return None, None
    if raw.startswith("{"):
        return json.loads(raw), "<inline-json>"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(hstu_root) / path
    return json.loads(path.read_text()), str(path)


def _shuffle_nccl(args):
    if args.gin_config:
        path = Path(args.gin_config)
        if path.is_file():
            text = path.read_text(errors="ignore").lower()
            if "balancedbatchshuffler" in text or "balanced_batch_shuffler" in text:
                return True
    return "--balanced_shuffler" in (
        args.gin_options or ""
    ) or "--balanced-shuffler" in (args.gin_options or "")


def _side_set(raw):
    if raw is None:
        return {"cpu", "gpu"}
    if isinstance(raw, int) and not isinstance(raw, bool):
        out = set()
        if raw & 1:
            out.add("cpu")
        if raw & 2:
            out.add("gpu")
        return out
    text = str(raw).strip().lower().replace("_", "-")
    if text in ("", "none", "off", "0", "-"):
        return set()
    if text in ("both", "cpu+gpu", "gpu+cpu", "cpu,gpu", "gpu,cpu"):
        return {"cpu", "gpu"}
    return {
        part.strip()
        for part in text.replace(",", "+").split("+")
        if part.strip() in ("cpu", "gpu")
    }


def _side_name(sides):
    if sides == {"cpu", "gpu"}:
        return "both"
    if sides == {"cpu"}:
        return "cpu"
    if sides == {"gpu"}:
        return "gpu"
    return "none"


def _sync_edge(raw):
    if isinstance(raw, str):
        return {"task": raw, "sides": "both"}
    if isinstance(raw, dict):
        return {
            "task": raw.get("task", raw.get("name")),
            "sides": _side_name(_side_set(raw.get("sides", "both"))),
        }
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return {"task": raw[0], "sides": _side_name(_side_set(raw[1]))}
    raise TypeError("bad same_progress_sync edge: {!r}".format(raw))


def _sync_map(raw):
    if not isinstance(raw, dict):
        return {}
    out = {}
    for task, spec in raw.items():
        if spec is None or spec is False:
            out[task] = []
        elif isinstance(spec, (str, dict)):
            out[task] = [_sync_edge(spec)]
        elif isinstance(spec, (list, tuple)):
            out[task] = [_sync_edge(edge) for edge in spec]
        else:
            raise TypeError("bad same_progress_sync.{}: {!r}".format(task, spec))
    return out


def _thread(config, task, stream):
    thread_map = config.get("thread_map")
    if isinstance(thread_map, dict):
        return thread_map.get(task, "default")
    if isinstance(thread_map, str) and thread_map:
        return "{}:{}".format(thread_map, stream)
    return stream


def _tasks(config, args):
    lookahead = config.get("lookahead") or {}
    sync = _sync_map(config.get("same_progress_sync"))
    shuffle = _shuffle_nccl(args)

    def t(name, stream, default_la, reads=(), writes=(), deps=(), nccl=False):
        offset = int(lookahead.get(name, default_la))
        return {
            "name": name,
            "stream": stream,
            "thread": _thread(config, name, stream),
            "lookahead": offset,
            "reads": [{"name": r, "offset": offset} for r in reads],
            "writes": [{"name": w, "offset": offset} for w in writes],
            "depends_on": list(deps),
            "same_progress_sync": sync.get(name, []),
            "nccl": bool(nccl),
        }

    tasks = [
        t(
            "h2d",
            "memcpy",
            2,
            reads=("batch_cpu",),
            writes=("batch_gpu", "torchrec_ctx"),
        ),
        t(
            "start_shuffle",
            "memcpy",
            2,
            reads=("batch_gpu",),
            writes=("shuffle_handle",),
            nccl=shuffle,
        ),
        t(
            "finish_shuffle",
            "memcpy",
            2,
            reads=("batch_gpu", "shuffle_handle"),
            writes=("shuffled_batch",),
            nccl=shuffle,
        ),
        t(
            "start_input_dist",
            "data_dist",
            1,
            reads=("shuffled_batch", "torchrec_ctx"),
            nccl=True,
        ),
        t(
            "wait_input_dist",
            "data_dist",
            1,
            reads=("torchrec_ctx",),
            deps=("start_input_dist",),
            nccl=True,
        ),
        t("zero_grad", "default", 0),
        t(
            "global_tokens_allreduce",
            "default",
            0,
            reads=("batch_gpu",),
            writes=("global_tokens",),
            nccl=True,
        ),
        t(
            "prefetch_embeddings",
            "prefetch",
            1,
            reads=("shuffled_batch", "torchrec_ctx"),
            deps=("wait_input_dist",),
        ),
        t(
            "compute_output_dist",
            "default",
            0,
            reads=("torchrec_ctx",),
            deps=("wait_input_dist", "prefetch_embeddings"),
            nccl=True,
        ),
    ]

    tasks.extend(
        [
            t(
                "ranking_embedding_forward",
                "default",
                0,
                reads=("batch_gpu", "torchrec_ctx", "shuffled_batch"),
                writes=("ranking_embeddings",),
                deps=("compute_output_dist", "prefetch_embeddings"),
            ),
            t(
                "dense_forward",
                "default",
                0,
                reads=("batch_gpu", "shuffled_batch", "ranking_embeddings"),
                writes=("losses", "output", "embedding_backward_inputs"),
                deps=("ranking_embedding_forward",),
            ),
        ]
    )

    tasks.extend(
        [
            t(
                "dense_backward",
                "default",
                0,
                reads=("losses", "global_tokens", "embedding_backward_inputs"),
                writes=("local_loss_sum", "embedding_grads"),
                deps=("zero_grad",),
            ),
            t(
                "embedding_backward",
                "default",
                0,
                reads=("embedding_grads",),
                deps=("dense_backward",),
                nccl=True,
            ),
            t(
                "finalize_model_grads",
                "default",
                0,
                deps=("embedding_backward",),
                nccl=True,
            ),
            t(
                "optimizer_step",
                "default",
                0,
                writes=("step_result",),
                deps=("finalize_model_grads",),
            ),
        ]
    )
    if _bool(args.watchdog):
        tasks.append(t("watchdog_step", "default", 0, deps=("optimizer_step",)))
    return tasks


def _edges(tasks):
    by_name = {task["name"]: task for task in tasks}
    writers = {
        (slot["name"], slot["offset"]): task["name"]
        for task in tasks
        for slot in task["writes"]
    }
    edges, seen = [], set()

    def add(producer, consumer, kind, reason="", sides=""):
        if not producer or producer == consumer:
            return
        key = (producer, consumer, kind, reason, sides)
        if key not in seen:
            seen.add(key)
            edge = {"producer": producer, "consumer": consumer, "kind": kind}
            if reason:
                edge["reason"] = reason
            if sides:
                edge["sides"] = sides
            edges.append(edge)

    for task in tasks:
        name = task["name"]
        for slot in task["reads"]:
            producer = writers.get((slot["name"], slot["offset"]))
            if producer:
                add(
                    producer, name, "slot", "{}@{}".format(slot["name"], slot["offset"])
                )
        for dep in task["depends_on"]:
            producer = by_name.get(dep)
            if producer and producer["lookahead"] == task["lookahead"]:
                add(dep, name, "depends_on")
        for sync in task["same_progress_sync"]:
            sides = _side_set(sync.get("sides", "both"))
            if "cpu" in sides:
                add(
                    sync.get("task"),
                    name,
                    "same_progress_sync",
                    sides=_side_name(sides),
                )
    return edges


def _topo(tasks, edges):
    pos = {task["name"]: i for i, task in enumerate(tasks)}
    incoming = {task["name"]: set() for task in tasks}
    outgoing = {task["name"]: set() for task in tasks}
    for edge in edges:
        if edge["producer"] in outgoing and edge["consumer"] in incoming:
            incoming[edge["consumer"]].add(edge["producer"])
            outgoing[edge["producer"]].add(edge["consumer"])

    ready = sorted([name for name in incoming if not incoming[name]], key=pos.get)
    order = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for nxt in sorted(outgoing[name], key=pos.get):
            incoming[nxt].discard(name)
            if not incoming[nxt]:
                ready.append(nxt)
                ready.sort(key=pos.get)
    if len(order) != len(tasks):
        raise RuntimeError(
            "cycle in DAG: {}".format(sorted(set(incoming) - set(order)))
        )
    return order


def _data(args):
    backend = args.backend or os.environ.get("RECSYS_PIPELINE_BACKEND", "legacy")
    config, source = _load_config(
        args.config or os.environ.get("HSTU_PIPELINE_CONFIG", ""), args.hstu_root
    )
    if backend != "new" or config is None:
        return {
            "version": 1,
            "generated_by": "export_pipeline_dag.py",
            "experiment": args.exp_name,
            "backend": backend,
            "note": "legacy backend or missing HSTU_PIPELINE_CONFIG: no schedulable-pipeline DAG/tickets to export",
            "tasks": [],
            "edges": [],
            "topo_order": [],
            "tickets": [],
        }

    tasks = _tasks(config, args)
    edges = _edges(tasks)
    order = _topo(tasks, edges)
    task_by_name = {task["name"]: task for task in tasks}
    tickets = [
        {"ticket": i, "task": name}
        for i, name in enumerate(n for n in order if task_by_name[n]["nccl"])
    ]
    return {
        "version": 1,
        "generated_by": "export_pipeline_dag.py",
        "experiment": args.exp_name,
        "backend": backend,
        "config": {
            "source": args.config,
            "resolved_path": source,
            "raw": config,
            "thread_map": config.get("thread_map"),
            "lookahead": config.get("lookahead", {}),
            "same_progress_sync": config.get("same_progress_sync", {}),
            "shuffle_nccl": _shuffle_nccl(args),
            "watchdog": _bool(args.watchdog),
        },
        "tasks": tasks,
        "edges": edges,
        "topo_order": order,
        "tickets": tickets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default=os.environ.get("RECSYS_PIPELINE_BACKEND", "legacy")
    )
    parser.add_argument("--config", default=os.environ.get("HSTU_PIPELINE_CONFIG", ""))
    parser.add_argument("--hstu-root", default=os.getcwd())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--gin-options", default="")
    parser.add_argument("--gin-config", default="")
    parser.add_argument("--watchdog", default=os.environ.get("CUDA_MEM_WATCHDOG", "0"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = _data(args)
    (out / "dag.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("DAG artifact saved: {}".format(out / "dag.json"))


if __name__ == "__main__":
    main()
