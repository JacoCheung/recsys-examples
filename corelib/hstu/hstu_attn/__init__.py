# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
__version__ = "1.0"
from .hstu_attn_cp import (
    GuardError,
    gather_global_from_cp_rank,
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)

# `hstu_attn_interface` requires `hstu_attn_2_cuda` C-extension, which is built
# only when this corelib is built from source against the local CUDA. In the
# container, the kernel is provided by a separately-installed `hstu` package
# (different shape: FBGEMM-style), so we leave this import optional. Callers
# that need the legacy wrappers can `from hstu_attn.hstu_attn_interface
# import ...` themselves and accept the C-extension dependency.
try:
    from .hstu_attn_interface import (  # noqa: F401
        hstu_attn_qkvpacked_func,
        hstu_attn_varlen_func,
    )

    _has_legacy_interface = True
except ImportError:
    _has_legacy_interface = False

__all__ = [
    "hstu_attn_varlen_cp_func",
    "get_batch_on_this_cp_rank_for_hstu",
    "gather_global_from_cp_rank",
    "GuardError",
]
if _has_legacy_interface:
    __all__ += ["hstu_attn_varlen_func", "hstu_attn_qkvpacked_func"]
