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

"""Guard that legacy pipeline files stay byte-identical on this branch.

SPEC §7 'Never do' — the engine package lands alongside legacy
train_pipeline.py / train_pipeline_factory.py / utils.py without
touching them. This test asserts `git diff HEAD` shows no change on
these files, catching accidental edits in the working tree.

CI should additionally compare against the PR base to catch committed
changes; that wrapping is outside this unit test.
"""

import pathlib
import subprocess
from typing import List

import pytest

LEGACY_FILES: List[str] = [
    "examples/commons/pipeline/train_pipeline.py",
    "examples/commons/pipeline/train_pipeline_factory.py",
    "examples/commons/pipeline/utils.py",
]


def _repo_root() -> pathlib.Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return pathlib.Path(result.stdout.strip())


def test_legacy_files_exist() -> None:
    """Files referenced in the guard must actually exist in the repo.

    If any of these files gets renamed or deleted, this test fails
    loudly so the guard is updated explicitly, not silently bypassed.
    """
    repo_root = _repo_root()
    for rel_path in LEGACY_FILES:
        full_path = repo_root / rel_path
        assert full_path.exists(), (
            f"Legacy file missing from repo: {rel_path}. "
            f"If this file was legitimately renamed/deleted, update "
            f"LEGACY_FILES in this test. Do not silently skip."
        )


def test_legacy_files_untouched() -> None:
    """Working tree must have no diff against HEAD on legacy pipeline files."""
    repo_root = _repo_root()
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", *LEGACY_FILES],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(repo_root),
    )
    if result.stdout.strip():
        pytest.fail(
            f"Legacy pipeline files have uncommitted changes:\n\n"
            f"{result.stdout}\n\n"
            f"SPEC §7 'Never do' — these files must stay byte-identical "
            f"on the engine-rework branch. Revert or move your change "
            f"into engine/."
        )
