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

"""V10 — integration tests for the example scripts.

Calls `main()` on each script and asserts it runs to completion. If
an example bitrots (API drift, import break, etc.), this test fails.
"""


def test_example_adopt_existing_loop_runs() -> None:
    """The T1 adoption demo must import cleanly and run to completion."""
    from commons.pipeline.engine.examples import adopt_existing_loop

    adopt_existing_loop.main()


def test_example_minimal_mlp_runs() -> None:
    """The T2 minimal-MLP demo must import cleanly, run the 50-step
    T1 and T2 loops, and confirm numerical parity between them."""
    from commons.pipeline.engine.examples import minimal_mlp

    minimal_mlp.main()


def test_engine_public_api_exact_surface() -> None:
    """SPEC §10 + README contract: `engine/__init__.py` exposes EXACTLY
    the 8 symbols documented in README.md's Public API table. Not
    ≤ N — exact match. Future slices adding public symbols must
    update the README + this test in the same change.

    Also asserts autosched symbols (`CostModel`, `CostProfiler`,
    `schedule_tasks`) are NOT reachable from the top-level engine
    namespace — they MUST be imported via
    `commons.pipeline.engine.autosched` explicitly. This protects
    the documented separation between "engine core" and
    "autosched add-on".
    """
    import commons.pipeline.engine as engine

    expected = {
        "DataSlot",
        "Schedule",
        "SchedulablePipeline",
        "ScheduleValidationError",
        "Stage",
        "StreamPool",
        "Task",
        "TaskContext",
    }
    actual = set(engine.__all__)
    assert actual == expected, (
        f"Public API drift.\n"
        f"  expected: {sorted(expected)}\n"
        f"  actual:   {sorted(actual)}\n"
        f"  added:    {sorted(actual - expected)}\n"
        f"  removed:  {sorted(expected - actual)}\n"
        f"If this is intentional, update both this test and "
        f"README.md's Public API table in the same change."
    )

    # Autosched symbols must NOT be reachable from top-level
    # `commons.pipeline.engine`. Users explicitly opt in via
    # `from commons.pipeline.engine.autosched import ...`.
    for autosched_symbol in ("CostModel", "CostProfiler", "schedule_tasks"):
        assert not hasattr(engine, autosched_symbol), (
            f"Autosched symbol {autosched_symbol!r} is reachable "
            f"from `commons.pipeline.engine` — that breaks the "
            f"documented engine-core vs. autosched-add-on separation "
            f"in README.md. Must be imported via "
            f"`commons.pipeline.engine.autosched` only."
        )
