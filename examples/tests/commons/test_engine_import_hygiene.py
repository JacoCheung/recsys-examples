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

"""Import hygiene guard for the schedulable-pipeline engine package.

The engine must not import TorchRec, Megatron, FBGEMM-GPU, or repo-local
commons.distributed.*. The check covers every Python file under
engine/, including tests and examples.

Examples under engine/examples/ may use torch.distributed (which does
not match any forbidden prefix below), but they must still avoid
torchrec/megatron/fbgemm_gpu.
"""

import ast
import pathlib
import subprocess
from typing import Iterator, Tuple

import pytest

FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "torchrec",
    "megatron",
    "fbgemm_gpu",
    "commons.distributed",
)


def _repo_root() -> pathlib.Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return pathlib.Path(result.stdout.strip())


ENGINE_ROOT = _repo_root() / "examples" / "commons" / "pipeline" / "engine"


def _python_files() -> Iterator[pathlib.Path]:
    """Every .py under engine/, excluding __pycache__."""
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imports_in_file(path: pathlib.Path) -> Iterator[str]:
    """Yield every absolute import module name referenced by the file.

    Relative imports (from . import X) target engine internals and are
    not subject to the forbidden-prefix check.
    """
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                yield node.module


@pytest.mark.parametrize(
    "path",
    list(_python_files()),
    ids=lambda p: str(p.relative_to(ENGINE_ROOT)),
)
def test_no_forbidden_imports(path: pathlib.Path) -> None:
    for imported in _imports_in_file(path):
        for prefix in FORBIDDEN_PREFIXES:
            if imported == prefix or imported.startswith(prefix + "."):
                pytest.fail(
                    f"{path.relative_to(ENGINE_ROOT)} imports "
                    f"'{imported}' which matches forbidden prefix "
                    f"'{prefix}'. Engine must stay framework-agnostic "
                    f"at the core package boundary."
                )
