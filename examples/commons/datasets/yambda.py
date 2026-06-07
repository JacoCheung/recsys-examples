# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.utils.logger import print_rank_0
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

LISTEN_TYPE = 0
LIKE_TYPE = 1
LISTEN_PLUS_THRESHOLD = 50

LP_BIT = 1
LIKE_BIT = 2
SKIP_BIT = 4

ITEM_ID_NUM_EMBEDDINGS = 9_390_624
ARTIST_ID_NUM_EMBEDDINGS = 1_293_395
ALBUM_ID_NUM_EMBEDDINGS = 3_367_692
UID_NUM_EMBEDDINGS = 1_000_001

YAMBDA_CONTEXTUAL_FEATURE_NAMES = [
    "uid",
    "user_x_artist",
    "user_x_album",
    "user_x_hour",
    "item_x_hour",
    "artist_x_hour",
    "user_x_is_organic",
    "user_x_artist_x_hour",
]
YAMBDA_CROSS_FEATURE_NAMES = [
    name for name in YAMBDA_CONTEXTUAL_FEATURE_NAMES if name != "uid"
]

YAMBDA_SEQUENCE_FEATURE_NAMES = ["item_id", "artist_id", "album_id"]

YAMBDA_5B_CROSS_SPECS: Tuple[Tuple[str, Tuple[str, ...], int, int], ...] = (
    ("user_x_artist", ("uid", "artist_id"), 100_000_000, 0),
    ("user_x_album", ("uid", "album_id"), 40_000_000, 0),
    ("user_x_hour", ("uid", "hour_of_day"), 24_000_000, 0),
    ("item_x_hour", ("item_id", "hour_of_day"), 40_000_000, 0),
    ("artist_x_hour", ("artist_id", "hour_of_day"), 32_000_000, 0),
    ("user_x_is_organic", ("uid", "is_organic"), 2_000_000, 0),
    ("user_x_artist_x_hour", ("uid", "artist_id", "hour_of_day"), 40_000_000, 0),
)


@dataclass
class YambdaHSTUBatch(HSTUBatch):
    """HSTU ranking batch with multiple item-side features per sequence position."""

    sequence_feature_names: List[str] = field(
        default_factory=lambda: list(YAMBDA_SEQUENCE_FEATURE_NAMES)
    )
    action_weights: Optional[KeyedJaggedTensor] = None
    action_weight_feature_name: str = "action_weight"


def _load_npy_readonly(path: Union[str, Path]) -> np.ndarray:
    """Load .npy through MAP_SHARED so all ranks share physical pages."""
    path = Path(path)
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()

    fd = os.open(str(path), os.O_RDONLY)
    try:
        buf = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    arr = np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
    arr.flags.writeable = False
    return arr


class _FlatEventStore:
    flat_uid: np.ndarray
    flat_item_ids: np.ndarray
    flat_timestamps: np.ndarray
    flat_event_types: np.ndarray
    flat_played_ratio: np.ndarray
    flat_is_listen_plus: np.ndarray
    flat_is_like: np.ndarray
    flat_is_skip: np.ndarray
    flat_is_organic: np.ndarray
    user_start: np.ndarray
    user_end: np.ndarray
    unique_uids: np.ndarray
    num_users: int
    total_events: int

    _MMAP_COLS: Tuple[str, ...] = (
        "flat_uid",
        "flat_item_ids",
        "flat_timestamps",
        "flat_event_types",
        "flat_played_ratio",
        "flat_is_listen_plus",
        "flat_is_like",
        "flat_is_skip",
        "flat_is_organic",
        "user_start",
        "user_end",
        "unique_uids",
    )

    @classmethod
    def load_mmap(cls, cache_dir: Union[str, Path]) -> "_FlatEventStore":
        import json

        cache_dir = Path(cache_dir)
        ready_path = cache_dir / "_READY"
        if not ready_path.exists():
            raise FileNotFoundError(
                f"Yambda cache sentinel is missing: {ready_path}. "
                "Build the cache from the reference recommendation_v4 preprocessing first."
            )
        with open(cache_dir / "store_meta.json") as f:
            meta = json.load(f)

        store = object.__new__(cls)
        missing = [
            str(cache_dir / f"{name}.npy")
            for name in cls._MMAP_COLS
            if not (cache_dir / f"{name}.npy").exists()
        ]
        if missing:
            raise FileNotFoundError(f"Yambda cache is missing files: {missing}")
        for name in cls._MMAP_COLS:
            setattr(store, name, _load_npy_readonly(cache_dir / f"{name}.npy"))
        store.num_users = int(meta["num_users"])
        store.total_events = int(meta["total_events"])
        return store


def _resolve_yambda_paths(
    dataset_path: Optional[str],
    history_length: int,
    cache_dir: Optional[str],
    metadata_path: Optional[str],
) -> Tuple[Path, Path, Path]:
    if dataset_path is None:
        dataset_path = "/home/scratch.junzhang_sw/workspace/datasets/yambda"
    root = Path(dataset_path)
    processed_dir = root / "processed_5b" if (root / "processed_5b").exists() else root
    metadata_dir = (
        Path(metadata_path)
        if metadata_path is not None
        else (
            root / "shared_metadata"
            if (root / "shared_metadata").exists()
            else root.parent / "shared_metadata"
        )
    )
    cache_path = (
        Path(cache_dir)
        if cache_dir is not None
        else processed_dir / f"hstu_cache_L{history_length}"
    )
    return processed_dir, metadata_dir, cache_path


def _read_parquet_columns(path: Path, columns: Sequence[str]) -> Dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=list(columns))
        return {name: table[name].to_numpy(zero_copy_only=False) for name in columns}
    except ImportError:
        import polars as pl

        frame = pl.read_parquet(path, columns=list(columns))
        return {name: frame[name].to_numpy() for name in columns}


def _load_item_metadata(metadata_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    artist_cols = _read_parquet_columns(
        metadata_dir / "artist_item_mapping.parquet", ["item_id", "artist_id"]
    )
    album_cols = _read_parquet_columns(
        metadata_dir / "album_item_mapping.parquet", ["item_id", "album_id"]
    )

    item_to_artist = np.zeros(ITEM_ID_NUM_EMBEDDINGS, dtype=np.int64)
    artist_items = artist_cols["item_id"].astype(np.int64, copy=False)
    artist_values = artist_cols["artist_id"].astype(np.int64, copy=False)
    valid = (artist_items >= 0) & (artist_items < ITEM_ID_NUM_EMBEDDINGS)
    item_to_artist[artist_items[valid]] = np.clip(
        artist_values[valid], 0, ARTIST_ID_NUM_EMBEDDINGS - 1
    )

    item_to_album = np.zeros(ITEM_ID_NUM_EMBEDDINGS, dtype=np.int64)
    album_items = album_cols["item_id"].astype(np.int64, copy=False)
    album_values = album_cols["album_id"].astype(np.int64, copy=False)
    valid = (album_items >= 0) & (album_items < ITEM_ID_NUM_EMBEDDINGS)
    item_to_album[album_items[valid]] = np.clip(
        album_values[valid], 0, ALBUM_ID_NUM_EMBEDDINGS - 1
    )
    return item_to_artist, item_to_album


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & 0xFFFFFFFFFFFFFFFF


def _xxh64(data: bytes, seed: int = 0) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    prime1 = 11400714785074694791
    prime2 = 14029467366897019727
    prime3 = 1609587929392839161
    prime4 = 9650029242287828579
    prime5 = 2870177450012600261
    length = len(data)
    idx = 0

    if length >= 32:
        v1 = (seed + prime1 + prime2) & mask
        v2 = (seed + prime2) & mask
        v3 = seed & mask
        v4 = (seed - prime1) & mask
        limit = length - 32
        while idx <= limit:
            lane = struct.unpack_from("<Q", data, idx)[0]
            v1 = _rotl64((v1 + lane * prime2) & mask, 31)
            v1 = (v1 * prime1) & mask
            lane = struct.unpack_from("<Q", data, idx + 8)[0]
            v2 = _rotl64((v2 + lane * prime2) & mask, 31)
            v2 = (v2 * prime1) & mask
            lane = struct.unpack_from("<Q", data, idx + 16)[0]
            v3 = _rotl64((v3 + lane * prime2) & mask, 31)
            v3 = (v3 * prime1) & mask
            lane = struct.unpack_from("<Q", data, idx + 24)[0]
            v4 = _rotl64((v4 + lane * prime2) & mask, 31)
            v4 = (v4 * prime1) & mask
            idx += 32
        h64 = (
            _rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)
        ) & mask
        for v in (v1, v2, v3, v4):
            v = (_rotl64((v * prime2) & mask, 31) * prime1) & mask
            h64 ^= v
            h64 = ((h64 * prime1) + prime4) & mask
    else:
        h64 = (seed + prime5) & mask

    h64 = (h64 + length) & mask
    while idx + 8 <= length:
        k1 = struct.unpack_from("<Q", data, idx)[0]
        k1 = (_rotl64((k1 * prime2) & mask, 31) * prime1) & mask
        h64 ^= k1
        h64 = ((_rotl64(h64, 27) * prime1) + prime4) & mask
        idx += 8
    if idx + 4 <= length:
        k1 = struct.unpack_from("<I", data, idx)[0]
        h64 ^= (k1 * prime1) & mask
        h64 = ((_rotl64(h64, 23) * prime2) + prime3) & mask
        idx += 4
    while idx < length:
        h64 ^= (data[idx] * prime5) & mask
        h64 = (_rotl64(h64, 11) * prime1) & mask
        idx += 1

    h64 ^= h64 >> 33
    h64 = (h64 * prime2) & mask
    h64 ^= h64 >> 29
    h64 = (h64 * prime3) & mask
    h64 ^= h64 >> 32
    return h64 & mask


def xxhash_cross(
    anchor: Dict[str, int], keys: Sequence[str], table_size: int, salt: int = 0
) -> int:
    packed = struct.Struct(f"<{len(keys)}q").pack(*(int(anchor[k]) for k in keys))
    try:
        import xxhash

        digest = xxhash.xxh64(seed=salt)
        digest.update(packed)
        return digest.intdigest() % table_size
    except ImportError:
        pass
    return _xxh64(packed, seed=salt) % table_size


class YambdaDataset(IterableDataset[YambdaHSTUBatch]):
    def __init__(
        self,
        dataset_path: Optional[str],
        batch_size: int,
        max_history_seqlen: int,
        max_num_candidates: int,
        num_tasks: int,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        random_seed: int,
        history_length: int = 2039,
        scan_window: int = 20000,
        max_samples: Optional[int] = None,
        sample_start: int = 0,
        window_ts: Optional[int] = None,
        streaming_window_seconds: int = 86400,
        streaming_sort_within_window: bool = False,
        cache_dir: Optional[str] = None,
        metadata_path: Optional[str] = None,
        disable_cross_features: bool = False,
        drop_last: bool = False,
        rank_split: str = "contiguous",
    ) -> None:
        super().__init__()
        if max_num_candidates != 1:
            raise ValueError("Yambda ranking currently expects max_num_candidates=1")
        if num_tasks != 1:
            raise ValueError("Yambda listen-plus ranking expects num_tasks=1")
        if rank_split not in ("contiguous", "round_robin"):
            raise ValueError(
                "rank_split must be either 'contiguous' or 'round_robin', "
                f"got {rank_split}"
            )
        if rank_split == "round_robin" and not drop_last:
            raise ValueError("round_robin rank_split requires drop_last=True")

        (
            self._processed_dir,
            self._metadata_dir,
            self._cache_dir,
        ) = _resolve_yambda_paths(
            dataset_path, history_length, cache_dir, metadata_path
        )
        self._store = _FlatEventStore.load_mmap(self._cache_dir)
        self._positions = _load_npy_readonly(
            self._cache_dir / f"positions_L{history_length}.npy"
        )
        self._item_to_artist, self._item_to_album = _load_item_metadata(
            self._metadata_dir
        )

        self._batch_size = batch_size
        self._global_batch_size = batch_size * world_size
        self._max_history_seqlen = max_history_seqlen
        self._max_num_candidates = max_num_candidates
        self._rank = rank
        self._world_size = world_size
        self._history_length = history_length
        self._scan_window = scan_window
        self._random_seed = random_seed
        self._shuffle = shuffle
        self._streaming_window_seconds = streaming_window_seconds
        self._streaming_sort_within_window = streaming_sort_within_window
        self._drop_last = drop_last
        self._rank_split = rank_split
        self._device = torch.device("cpu")
        self._contextual_feature_names = (
            ["uid"] if disable_cross_features else list(YAMBDA_CONTEXTUAL_FEATURE_NAMES)
        )
        self._disable_cross_features = disable_cross_features
        self._sequence_feature_names = list(YAMBDA_SEQUENCE_FEATURE_NAMES)
        self._feature_to_max_seqlen: Dict[str, int] = {
            **{name: 1 for name in self._contextual_feature_names},
            **{
                name: max_history_seqlen + max_num_candidates
                for name in self._sequence_feature_names
            },
            "action_weight": max_history_seqlen,
        }

        self._set_sample_ids(
            sample_ids=self._load_sample_ids(
                window_ts=window_ts,
                streaming_window_seconds=streaming_window_seconds,
                streaming_sort_within_window=streaming_sort_within_window,
            ),
            max_samples=max_samples,
            sample_start=sample_start,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        print_rank_0(
            "[YambdaDataset] "
            f"processed={self._processed_dir}, cache={self._cache_dir}, "
            f"metadata={self._metadata_dir}, samples={self._num_samples:,}, "
            f"rank={rank}/{world_size}, batch={batch_size}, window_ts={window_ts}, "
            f"shuffle={shuffle}, drop_last={drop_last}, rank_split={rank_split}, "
            f"disable_cross_features={disable_cross_features}"
        )

    def _set_sample_ids(
        self,
        sample_ids: np.ndarray,
        max_samples: Optional[int],
        sample_start: int,
        shuffle: bool,
        random_seed: int,
    ) -> None:
        total_samples = int(sample_ids.shape[0])
        if sample_start < 0:
            raise ValueError(f"sample_start must be >= 0, got {sample_start}")
        end = (
            total_samples
            if max_samples is None
            else min(total_samples, sample_start + max_samples)
        )
        self._sample_ids = np.array(sample_ids[sample_start:end], dtype=np.int64)
        self._shuffle = shuffle
        self._random_seed = random_seed
        if shuffle:
            rng = np.random.default_rng(random_seed)
            self._sample_ids = rng.permutation(self._sample_ids)
        self._num_samples = int(self._sample_ids.shape[0])

    def set_window(
        self,
        window_ts: int,
        *,
        max_samples: Optional[int] = None,
        sample_start: int = 0,
        shuffle: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """Switch the active anchor set to one streaming time window.

        This mirrors the reference Yambda streaming path: a train window T is
        followed by evaluating window T+1. Window indices are anchor positions
        whose target timestamp falls inside the fixed-duration window.
        """
        self._set_sample_ids(
            sample_ids=self._load_sample_ids(
                window_ts=window_ts,
                streaming_window_seconds=self._streaming_window_seconds,
                streaming_sort_within_window=self._streaming_sort_within_window,
            ),
            max_samples=max_samples,
            sample_start=sample_start,
            shuffle=shuffle,
            random_seed=self._random_seed if random_seed is None else random_seed,
        )
        print_rank_0(
            "[YambdaDataset] "
            f"set_window={window_ts}, samples={self._num_samples:,}, "
            f"rank={self._rank}/{self._world_size}, batch={self._batch_size}, "
            f"shuffle={shuffle}, drop_last={self._drop_last}, "
            f"rank_split={self._rank_split}"
        )

    def _load_sample_ids(
        self,
        window_ts: Optional[int],
        streaming_window_seconds: int,
        streaming_sort_within_window: bool,
    ) -> np.ndarray:
        if window_ts is None:
            return np.arange(self._positions.shape[0], dtype=np.int64)

        cache_path = self._cache_dir / (
            f"window_indices_L{self._history_length}_"
            f"W{streaming_window_seconds}_ts{window_ts}_"
            f"sort{int(streaming_sort_within_window)}.npy"
        )
        if cache_path.exists():
            return _load_npy_readonly(cache_path)

        anchor_ts_path = self._cache_dir / f"anchor_ts_L{self._history_length}.npy"
        meta_path = self._cache_dir / f"anchor_ts_L{self._history_length}.meta.json"
        if not anchor_ts_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Yambda window index {cache_path} is missing, and anchor timestamp "
                f"cache {anchor_ts_path} / {meta_path} is not available to rebuild it."
            )
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        anchor_ts = _load_npy_readonly(anchor_ts_path)
        lo = int(meta["t_min"]) + window_ts * streaming_window_seconds
        hi = lo + streaming_window_seconds
        import fcntl

        lock_path = str(cache_path) + ".lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            if cache_path.exists():
                return _load_npy_readonly(cache_path)
            sample_ids = np.where((anchor_ts >= lo) & (anchor_ts < hi))[0].astype(
                np.int64
            )
            if streaming_sort_within_window and sample_ids.size > 0:
                sample_ids = sample_ids[
                    np.argsort(anchor_ts[sample_ids], kind="stable")
                ]
            tmp_path = str(cache_path) + ".tmp.npy"
            np.save(tmp_path, sample_ids)
            os.replace(tmp_path, cache_path)
        print_rank_0(
            "[YambdaDataset] "
            f"window_ts={window_ts}, range=[{lo}, {hi}), "
            f"samples={int(sample_ids.size):,}, cache={cache_path}"
        )
        return sample_ids

    def __len__(self) -> int:
        if self._rank_split == "round_robin":
            rank_samples = self._num_samples // self._world_size
            if self._drop_last:
                return rank_samples // self._batch_size
            return math.ceil(rank_samples / self._batch_size)
        if self._drop_last:
            return self._num_samples // self._global_batch_size
        return math.ceil(self._num_samples / self._global_batch_size)

    def num_windows(self) -> int:
        """Number of fixed-duration streaming windows in the anchor cache."""
        meta_path = self._cache_dir / f"anchor_ts_L{self._history_length}.meta.json"
        if not meta_path.exists():
            return 0
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        span = int(meta["t_max"]) - int(meta["t_min"]) + 1
        return math.ceil(span / self._streaming_window_seconds)

    def _flat_pos(self, sample_id: int) -> int:
        return int(self._positions[int(sample_id)])

    def _gather_history(
        self, flat_pos: int, user_start: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        per_pool = max(1, self._history_length // 3)
        scan_start = max(int(user_start), int(flat_pos) - self._scan_window)
        scan_end = int(flat_pos)
        if scan_end <= scan_start:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, empty

        item_ids = self._store.flat_item_ids[scan_start:scan_end]
        is_lp = self._store.flat_is_listen_plus[scan_start:scan_end]
        is_like = self._store.flat_is_like[scan_start:scan_end]
        is_skip = self._store.flat_is_skip[scan_start:scan_end]

        idx_all = np.arange(item_ids.shape[0], dtype=np.int64)
        keep_local = np.concatenate(
            [
                idx_all[is_lp][-per_pool:],
                idx_all[is_like][-per_pool:],
                idx_all[is_skip][-per_pool:],
            ]
        )
        if keep_local.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, empty
        keep_local = keep_local[np.argsort(keep_local, kind="stable")]

        items = np.clip(item_ids[keep_local], 0, ITEM_ID_NUM_EMBEDDINGS - 1).astype(
            np.int64, copy=False
        )
        weights = np.zeros(keep_local.shape[0], dtype=np.int64)
        weights[is_lp[keep_local]] |= LP_BIT
        weights[is_like[keep_local]] |= LIKE_BIT
        weights[is_skip[keep_local]] |= SKIP_BIT
        if items.shape[0] > self._max_history_seqlen:
            items = items[-self._max_history_seqlen :]
            weights = weights[-self._max_history_seqlen :]
        artists = self._item_to_artist[items]
        albums = self._item_to_album[items]
        return items, artists, albums, weights

    def _build_sample(
        self, sample_id: int
    ) -> Tuple[Dict[str, int], Dict[str, List[int]], List[int], int]:
        flat_pos = self._flat_pos(sample_id)
        uid = int(self._store.flat_uid[flat_pos])
        uid = max(0, min(uid, UID_NUM_EMBEDDINGS - 1))
        user_start = int(self._store.user_start[uid])
        items, artists, albums, action_weights = self._gather_history(
            flat_pos, user_start
        )

        target_item = int(
            np.clip(self._store.flat_item_ids[flat_pos], 0, ITEM_ID_NUM_EMBEDDINGS - 1)
        )
        target_artist = int(self._item_to_artist[target_item])
        target_album = int(self._item_to_album[target_item])
        target_ts = int(self._store.flat_timestamps[flat_pos])
        event_type = int(self._store.flat_event_types[flat_pos])
        if event_type != LISTEN_TYPE:
            raise ValueError(
                "Yambda positions cache must contain listen-event anchors only, "
                f"but flat position {flat_pos} has event_type={event_type}"
            )
        played_ratio = float(self._store.flat_played_ratio[flat_pos])
        label = int(played_ratio >= LISTEN_PLUS_THRESHOLD)

        anchor: Dict[str, int] = {
            "uid": uid,
            "item_id": target_item,
            "artist_id": target_artist,
            "album_id": target_album,
            "hour_of_day": int((target_ts // 3600) % 24),
            "is_organic": int(self._store.flat_is_organic[flat_pos]),
        }
        contextual = {"uid": uid}
        if not self._disable_cross_features:
            for name, keys, table_size, salt in YAMBDA_5B_CROSS_SPECS:
                contextual[name] = xxhash_cross(anchor, keys, table_size, salt)

        sequence_features = {
            "item_id": items.tolist() + [target_item],
            "artist_id": artists.tolist() + [target_artist],
            "album_id": albums.tolist() + [target_album],
        }
        return contextual, sequence_features, action_weights.tolist(), label

    def __iter__(self) -> Iterator[YambdaHSTUBatch]:
        rank_sample_ids = None
        if self._rank_split == "round_robin":
            usable_samples = (
                (self._num_samples // self._world_size) * self._world_size
                if self._drop_last
                else self._num_samples
            )
            rank_sample_ids = self._sample_ids[:usable_samples][
                self._rank :: self._world_size
            ]
        for global_batch_idx in range(len(self)):
            if rank_sample_ids is not None:
                local_batch_start = global_batch_idx * self._batch_size
                local_batch_end = min(
                    local_batch_start + self._batch_size,
                    int(rank_sample_ids.shape[0]),
                )
                sample_ids = rank_sample_ids[local_batch_start:local_batch_end]
            else:
                local_batch_start = min(
                    global_batch_idx * self._global_batch_size
                    + self._rank * self._batch_size,
                    self._num_samples,
                )
                local_batch_end = min(
                    global_batch_idx * self._global_batch_size
                    + (self._rank + 1) * self._batch_size,
                    self._num_samples,
                )
                sample_ids = self._sample_ids[local_batch_start:local_batch_end]
            actual_batch_size = int(sample_ids.shape[0])
            pad_size = self._batch_size - actual_batch_size

            values_by_key: Dict[str, List[int]] = {
                name: []
                for name in self._contextual_feature_names
                + self._sequence_feature_names
            }
            lengths_by_key: Dict[str, List[int]] = {
                name: []
                for name in self._contextual_feature_names
                + self._sequence_feature_names
            }
            labels: List[int] = []
            num_candidates: List[int] = []
            action_weight_values: List[int] = []
            action_weight_lengths: List[int] = []

            for sample_id in sample_ids:
                (
                    contextual,
                    sequence_features,
                    action_weights,
                    label,
                ) = self._build_sample(int(sample_id))
                for name in self._contextual_feature_names:
                    values_by_key[name].append(int(contextual[name]))
                    lengths_by_key[name].append(1)
                for name in self._sequence_feature_names:
                    seq = sequence_features[name]
                    values_by_key[name].extend(seq)
                    lengths_by_key[name].append(len(seq))
                action_weight_values.extend(action_weights)
                action_weight_lengths.append(len(action_weights))
                labels.append(label)
                num_candidates.append(1)

            for name in values_by_key:
                if pad_size > 0:
                    lengths_by_key[name].extend([0] * pad_size)
            if pad_size > 0:
                action_weight_lengths.extend([0] * pad_size)

            keys = self._contextual_feature_names + self._sequence_feature_names
            feature_values = torch.tensor(
                [v for name in keys for v in values_by_key[name]],
                device=self._device,
                dtype=torch.int64,
            )
            feature_lengths = torch.tensor(
                [v for name in keys for v in lengths_by_key[name]],
                device=self._device,
                dtype=torch.int64,
            )
            features = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=feature_values,
                lengths=feature_lengths,
            )

            if pad_size > 0:
                num_candidates.extend([0] * pad_size)
            num_candidates_tensor = torch.tensor(
                num_candidates, device=self._device, dtype=torch.int64
            )
            labels_kjt = KeyedJaggedTensor.from_lengths_sync(
                keys=["label"],
                values=torch.tensor(labels, device=self._device, dtype=torch.int64),
                lengths=num_candidates_tensor,
            )
            action_weights_kjt = KeyedJaggedTensor.from_lengths_sync(
                keys=["action_weight"],
                values=torch.tensor(
                    action_weight_values, device=self._device, dtype=torch.int64
                ),
                lengths=torch.tensor(
                    action_weight_lengths, device=self._device, dtype=torch.int64
                ),
            )

            yield YambdaHSTUBatch(
                features=features,
                batch_size=self._batch_size,
                feature_to_max_seqlen=dict(self._feature_to_max_seqlen),
                contextual_feature_names=list(self._contextual_feature_names),
                labels=labels_kjt,
                actual_batch_size=actual_batch_size,
                item_feature_name="item_id",
                action_feature_name=None,
                max_num_candidates=self._max_num_candidates,
                num_candidates=num_candidates_tensor,
                sequence_feature_names=list(self._sequence_feature_names),
                action_weights=action_weights_kjt,
                action_weight_feature_name="action_weight",
            )


def get_dataset(
    dataset_path: Optional[str],
    max_history_seqlen: int,
    max_num_candidates: int,
    num_tasks: int,
    train_batch_size: int,
    eval_batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    random_seed: int,
    history_length: int = 2039,
    scan_window: int = 20000,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    train_sample_start: int = 0,
    eval_sample_start: int = 0,
    train_window_ts: Optional[int] = None,
    eval_window_ts: Optional[int] = None,
    eval_shuffle: bool = False,
    streaming_window_seconds: int = 86400,
    streaming_sort_within_window: bool = False,
    cache_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    disable_cross_features: bool = False,
) -> Tuple[YambdaDataset, YambdaDataset]:
    train_dataset = YambdaDataset(
        dataset_path=dataset_path,
        batch_size=train_batch_size,
        max_history_seqlen=max_history_seqlen,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        random_seed=random_seed,
        history_length=history_length,
        scan_window=scan_window,
        max_samples=max_train_samples,
        sample_start=train_sample_start,
        window_ts=train_window_ts,
        streaming_window_seconds=streaming_window_seconds,
        streaming_sort_within_window=streaming_sort_within_window,
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        disable_cross_features=disable_cross_features,
    )
    eval_dataset = YambdaDataset(
        dataset_path=dataset_path,
        batch_size=eval_batch_size,
        max_history_seqlen=max_history_seqlen,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        rank=rank,
        world_size=world_size,
        shuffle=eval_shuffle,
        random_seed=random_seed + 1,
        history_length=history_length,
        scan_window=scan_window,
        max_samples=max_eval_samples,
        sample_start=eval_sample_start,
        window_ts=eval_window_ts,
        streaming_window_seconds=streaming_window_seconds,
        streaming_sort_within_window=streaming_sort_within_window,
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        disable_cross_features=disable_cross_features,
    )
    return train_dataset, eval_dataset
