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

"""T1 adoption demo — ≤ 8-line diff vs vanilla PyTorch loop.

**Vanilla PyTorch (before):**

    model = nn.Linear(10, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for batch in dataloader:                   # <- loop body (6 lines)
        batch = batch.cuda()                   #    starts here
        optimizer.zero_grad()
        out = model(batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()

**Engine-hosted (after):**

    model = nn.Linear(10, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    pipe = SchedulablePipeline.basic(          # <- +1 line (setup)
        model, optimizer, loss_fn=lambda out: out.sum()
    )
    for batch in dataloader:
        pipe.step(batch)                       # <- +1 line (replaces 6)

`git diff`: 6 deletions + 2 insertions = 8 lines changed.

The model class, optimizer, dataloader, and loss convention all stay
untouched. Everything else the engine does (ring-advance, §4.8
prefill/drain, task scheduling) happens inside `pipe.step`.
"""

from typing import List

import torch
from commons.pipeline.engine import SchedulablePipeline


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Linear(10, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # The one net-new line in the training-loop preamble.
    pipe = SchedulablePipeline.basic(model, optimizer, loss_fn=lambda out: out.sum())

    # Synthetic "dataloader" — a list of random batches.
    dataloader: List[torch.Tensor] = [
        torch.randn(4, 10, device=device) for _ in range(5)
    ]

    # Vanilla 6-line loop body replaced by `pipe.step(batch)`.
    for batch in dataloader:
        result = pipe.step(batch)
        print(f"  step_result = {result.detach().sum().item():.4f}")

    print("T1 vanilla training loop — 5 steps completed via engine")


if __name__ == "__main__":
    main()
