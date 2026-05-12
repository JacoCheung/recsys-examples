# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CostModel and CostProfiler tests."""

import pathlib

import pytest
import torch
from commons.pipeline.engine import SchedulablePipeline
from commons.pipeline.engine.autosched import CostModel, CostProfiler, TaskCost


def test_task_cost_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TaskCost(cpu_us=-1.0, gpu_us=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        TaskCost(cpu_us=0.0, gpu_us=-5.0)


def test_task_cost_total_is_max() -> None:
    c = TaskCost(cpu_us=10.0, gpu_us=100.0)
    assert c.total_us == 100.0
    c = TaskCost(cpu_us=50.0, gpu_us=20.0)
    assert c.total_us == 50.0


def test_cost_model_from_dict_roundtrips_via_json(tmp_path: pathlib.Path) -> None:
    raw = {
        "h2d": {"cpu_us": 12.3, "gpu_us": 456.7},
        "forward": {"cpu_us": 100.0, "gpu_us": 2000.5},
        "backward": {"cpu_us": 150.0, "gpu_us": 3500.0},
    }
    model = CostModel.from_dict(raw)
    out = tmp_path / "cost.json"
    model.save_json(out)
    reloaded = CostModel.from_json(out)
    assert model == reloaded
    assert reloaded.get("forward") == TaskCost(cpu_us=100.0, gpu_us=2000.5)


def test_cost_model_get_unknown_task_raises() -> None:
    model = CostModel.from_dict({"a": {"cpu_us": 1.0, "gpu_us": 2.0}})
    with pytest.raises(KeyError, match="No cost entry"):
        model.get("nonexistent")


def test_cost_profiler_gathers_task_timings() -> None:
    """Profiler records positive CPU time for every preset task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    pipe = SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    batches = [torch.randn(2, 4, device=device) for _ in range(5)]

    profiler = CostProfiler(pipe)
    profiler.run(iter(batches), steps=3)

    cm = profiler.as_cost_model()
    names = set(cm.task_names())
    # Preset schedule produces 5 named tasks:
    expected_names = {
        "h2d",
        "zero_grad",
        "forward",
        "backward",
        "optimizer",
    }
    assert expected_names <= names, (
        f"profiler missed some tasks: expected >= {expected_names}, " f"got {names}"
    )
    for name in expected_names:
        cost = cm.get(name)
        assert cost.cpu_us > 0, f"{name} had zero cpu_us — profiler broken"


def test_cost_profiler_reset_clears_aggregates() -> None:
    """reset() clears prior measurements."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    pipe = SchedulablePipeline.basic(model, opt, loss_fn=lambda o: o.sum())

    profiler = CostProfiler(pipe)
    profiler.run(iter([torch.randn(2, 4, device=device)]), steps=1)
    assert len(list(profiler.as_cost_model().task_names())) > 0

    profiler.reset()
    assert list(profiler.as_cost_model().task_names()) == []
