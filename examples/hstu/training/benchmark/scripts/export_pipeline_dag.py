#!/usr/bin/env python3
"""Export HSTU schedulable-pipeline DAG + NCCL tickets for a benchmark run.

Stdlib-only by design: this runs before the training container, where torch,
nvtx, and repo imports may be unavailable.
"""

import argparse
import html
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

    if _bool(config.get("split_ranking_forward", False)):
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
                    "forward",
                    "default",
                    0,
                    reads=("batch_gpu", "shuffled_batch", "ranking_embeddings"),
                    writes=("losses", "output"),
                    deps=("ranking_embedding_forward",),
                ),
            ]
        )
    else:
        tasks.append(
            t(
                "forward",
                "default",
                0,
                reads=("batch_gpu", "torchrec_ctx", "shuffled_batch"),
                writes=("losses", "output"),
                deps=("compute_output_dist", "prefetch_embeddings"),
            )
        )

    tasks.extend(
        [
            t(
                "backward",
                "default",
                0,
                reads=("losses", "global_tokens"),
                writes=("local_loss_sum",),
                deps=("zero_grad",),
                nccl=True,
            ),
            t("finalize_model_grads", "default", 0, deps=("backward",), nccl=True),
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


def _levels(order, edges):
    level = {name: 0 for name in order}
    for name in order:
        for edge in edges:
            if edge["producer"] == name and edge["consumer"] in level:
                level[edge["consumer"]] = max(level[edge["consumer"]], level[name] + 1)
    return level


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
            "split_ranking_forward": _bool(config.get("split_ranking_forward", False)),
            "shuffle_nccl": _shuffle_nccl(args),
            "watchdog": _bool(args.watchdog),
        },
        "tasks": tasks,
        "edges": edges,
        "topo_order": order,
        "tickets": tickets,
    }


def _render(data):
    esc = html.escape
    title = "Pipeline DAG - {}".format(data.get("experiment") or "benchmark")
    if not data["tasks"]:
        return '<!doctype html><meta charset="utf-8"><title>{}</title><body style="font-family:system-ui;background:#0d1117;color:#e6edf3;padding:24px"><h1>{}</h1><p>{}</p><p><a href="dag.json" style="color:#58a6ff">dag.json</a></p></body>'.format(
            esc(title), esc(title), esc(data.get("note", ""))
        )

    tasks = {task["name"]: task for task in data["tasks"]}
    levels = _levels(data["topo_order"], data["edges"])
    groups = {}
    for name in data["topo_order"]:
        groups.setdefault(levels[name], []).append(name)
    width = max(980, max(len(v) for v in groups.values()) * 190 + 80)
    height = (max(groups) + 1) * 92 + 80
    xy = {}
    for level, names in groups.items():
        start = (width - len(names) * 190) / 2
        for i, name in enumerate(names):
            xy[name] = (start + i * 190, 44 + level * 92)

    stream_color = {
        "default": "#58a6ff",
        "memcpy": "#d29922",
        "data_dist": "#bc8cff",
        "prefetch": "#3fb950",
    }
    edge_color = {
        "slot": "#8b949e",
        "depends_on": "#58a6ff",
        "same_progress_sync": "#d29922",
    }
    ticket = {entry["task"]: entry["ticket"] for entry in data["tickets"]}
    svg = [
        '<svg viewBox="0 0 {0} {1}" width="100%" height="{1}"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#8b949e"/></marker></defs>'.format(
            width, height
        )
    ]
    for edge in data["edges"]:
        if edge["producer"] not in xy or edge["consumer"] not in xy:
            continue
        x1, y1 = xy[edge["producer"]]
        x2, y2 = xy[edge["consumer"]]
        dash = ' stroke-dasharray="5 4"' if edge["kind"] == "same_progress_sync" else ""
        svg.append(
            '<path d="M{:.1f},{:.1f} C{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}" fill="none" stroke="{}"{} marker-end="url(#a)"><title>{}</title></path>'.format(
                x1 + 72,
                y1 + 54,
                x1 + 72,
                (y1 + y2) / 2,
                x2 + 72,
                (y1 + y2) / 2,
                x2 + 72,
                y2,
                edge_color.get(edge["kind"], "#8b949e"),
                dash,
                esc(
                    "{} -> {} ({})".format(
                        edge["producer"], edge["consumer"], edge["kind"]
                    )
                ),
            )
        )
    for name in data["topo_order"]:
        task, (x, y) = tasks[name], xy[name]
        tlabel = "" if name not in ticket else "T{}".format(ticket[name])
        svg.append(
            '<rect x="{:.1f}" y="{:.1f}" width="145" height="54" rx="7" fill="#161b22" stroke="{}" stroke-width="2"/>'.format(
                x, y, stream_color.get(task["stream"], "#8b949e")
            )
        )
        svg.append(
            '<text x="{:.1f}" y="{:.1f}" fill="#e6edf3" font-size="12" font-weight="700">{}</text>'.format(
                x + 8, y + 20, esc(name)
            )
        )
        svg.append(
            '<text x="{:.1f}" y="{:.1f}" fill="#8b949e" font-size="10">la={} {}</text>'.format(
                x + 8, y + 38, task["lookahead"], esc(task["stream"])
            )
        )
        if tlabel:
            svg.append(
                '<text x="{:.1f}" y="{:.1f}" fill="#f85149" font-size="11" font-weight="700">{}</text>'.format(
                    x + 112, y + 20, tlabel
                )
            )
    svg.append("</svg>")

    rows = "\n".join(
        "<tr><td>{ticket}</td><td>{task}</td><td>{stream}</td><td>{thread}</td></tr>".format(
            ticket=e["ticket"],
            task=esc(e["task"]),
            stream=esc(tasks[e["task"]]["stream"]),
            thread=esc(tasks[e["task"]]["thread"]),
        )
        for e in data["tickets"]
    )
    return """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;padding:24px;background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif}}a{{color:#58a6ff}}.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:14px 0;overflow:auto}}table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #30363d;padding:6px 10px;font-size:12px;text-align:left}}th,.muted{{color:#8b949e}}code{{color:#bc8cff}}</style></head>
<body><h1>{title}</h1><div class="muted">backend=<code>{backend}</code> config=<code>{config}</code> - <a href="dag.json">dag.json</a></div>
<div class="panel"><span class="muted">gray=slot, blue=depends_on, orange dashed=same_progress_sync CPU, red T#=NCCL ticket</span>{svg}</div>
<div class="panel"><b>Topo order</b><br><code>{topo}</code></div>
<div class="panel"><b>NCCL tickets</b><table><thead><tr><th>ticket</th><th>task</th><th>stream</th><th>thread</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>""".format(
        title=esc(title),
        backend=esc(data.get("backend", "")),
        config=esc((data.get("config") or {}).get("source") or ""),
        svg="\n".join(svg),
        topo=esc(" -> ".join(data["topo_order"])),
        rows=rows or '<tr><td colspan="4">none</td></tr>',
    )


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
    (out / "dag.html").write_text(_render(data))
    print("DAG artifacts saved: {} {}".format(out / "dag.json", out / "dag.html"))


if __name__ == "__main__":
    main()
