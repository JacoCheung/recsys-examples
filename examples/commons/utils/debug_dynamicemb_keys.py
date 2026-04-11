"""Debug: track DynamicEmb key distribution per rank via prefetch monkey-patch.

Enable with DYNAMICEMB_KEY_DEBUG=1.
Prints summary every 100 iters with key range, unique count, hash table occupancy.
"""

import os
from typing import Any

import torch
import torch.nn as nn
from commons.utils.dynamicemb_utils import find_dynamicemb_modules

_REPORT_INTERVAL = 100


def auto_install(model: nn.Module) -> int:
    if os.environ.get("DYNAMICEMB_KEY_DEBUG", "0") != "1":
        return 0

    modules = find_dynamicemb_modules(model)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[KEYDBG] Found {len(modules)} DynamicEmb module(s)", flush=True)

    count = 0
    for m in modules:
        _patch_prefetch(m, rank)
        count += 1
    if rank == 0:
        print(f"[KEYDBG] Patched {count} prefetch(s)", flush=True)
    return count


def _patch_prefetch(module: Any, rank: int) -> None:
    original_prefetch = module.prefetch
    state = {
        "iter": 0,
        "cumul_unique": set(),
        "cumul_total": 0,
        "window_keys": [],
    }

    def patched_prefetch(indices, offsets, *args, **kwargs):
        if indices is not None and indices.numel() > 0:
            keys_cpu = indices.detach().cpu()
            state["cumul_total"] += keys_cpu.numel()
            for k in torch.unique(keys_cpu).tolist():
                state["cumul_unique"].add(k)
            state["window_keys"].extend(keys_cpu.tolist())

        state["iter"] += 1

        if state["iter"] % _REPORT_INTERVAL == 0:
            _print_summary(module, state, rank)

        return original_prefetch(indices, offsets, *args, **kwargs)

    module.prefetch = patched_prefetch


def _print_summary(module: Any, state: dict, rank: int) -> None:
    km = None
    storage = getattr(module, "_storage", None)
    cache = getattr(module, "_cache", None)
    if storage is not None and hasattr(storage, "_state"):
        km = storage._state.key_index_map
    elif cache is not None and hasattr(cache, "_state"):
        km = cache._state.key_index_map

    ht_info = ""
    if km is not None:
        try:
            sz = int(km.size())
            cap = int(km.capacity())
            lf = sz / cap if cap > 0 else 0
            ht_info = f" ht_size={sz} ht_cap={cap} ht_lf={lf:.4f}"
        except Exception:
            pass

    tables_str = ",".join(module.table_names)
    key_info = ""
    if state["window_keys"]:
        t = torch.tensor(state["window_keys"], dtype=torch.long)
        key_info = (
            f" key_min={t.min().item()} key_max={t.max().item()}"
            f" key_p50={int(torch.quantile(t.float(), 0.5).item())}"
            f" key_p99={int(torch.quantile(t.float(), 0.99).item())}"
        )

    ws = state["iter"] - _REPORT_INTERVAL
    print(
        f"[rank{rank}] [KEYDBG iter={ws}-{state['iter'] - 1}]"
        f" tables={tables_str}{ht_info}"
        f" cumul_unique={len(state['cumul_unique'])}"
        f" cumul_total={state['cumul_total']}"
        f" window_keys={len(state['window_keys'])}"
        f"{key_info}",
        flush=True,
    )
    state["window_keys"] = []
