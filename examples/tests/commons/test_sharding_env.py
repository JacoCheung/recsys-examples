# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from commons.distributed.sharding import _enable_index_dedup


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("0", False), ("1", True)],
)
def test_enable_index_dedup(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ENABLE_DEDUP", raising=False)
    else:
        monkeypatch.setenv("ENABLE_DEDUP", value)

    assert _enable_index_dedup() is expected


@pytest.mark.parametrize("value", ["", "2", "true"])
def test_enable_index_dedup_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("ENABLE_DEDUP", value)

    with pytest.raises(ValueError, match="ENABLE_DEDUP must be 0 or 1"):
        _enable_index_dedup()
