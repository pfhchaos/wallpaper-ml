"""Schema-driven packed training corpus — mmap-able .npy files with
named-channel access.

Training corpora are commonly stored as per-example .npz files (one
file per audio clip, image, etc.) with multiple named arrays. At
training time, reading each .npz means decompressing zlib chunks for
every batch — fine for small datasets, expensive for large ones where
the same files get re-read many times per epoch.

This module packs a corpus of .npz files into per-example .npy files
with a fixed column layout, then mmap-loads them at training time so
slicing is a free view (no decompression, no copy). A schema declares
which keys to extract and how many columns each occupies; the schema
is written as `schema.json` alongside the packed files so trainers can
recover the column layout without hardcoding column indices.

Pattern:

    schema = CorpusSchema(channels=[
        ChannelSpec(name='spectrum',    npz_key='spectrum',    columns=108),
        ChannelSpec(name='diff',        npz_key='diff',        columns=108),
        ChannelSpec(name='labels_hier', npz_key='labels_hier', columns=3),
    ])
    pack_corpus(src_dir, dst_dir, schema)
    # ... later, in trainer:
    schema = load_schema(dst_dir)
    packed = load_packed(dst_dir / 'foo.npy')   # (T, 219) memmap
    spec   = packed[:, schema.column_range('spectrum')]     # (T, 108) view
    target = packed[:, schema.column_range('labels_hier')]  # (T, 3) view

Built for time-series corpora (rows = frames) but is shape-agnostic
beyond requiring (T, *) inputs — first axis must be aligned across
channels. Per-example fields with no row axis (a scalar score per
clip, say) don't fit; promote them to (1, N) at pack time.

Disk cost: roughly source-corpus-size × decompression_ratio (3-5x for
zlib float arrays). Disk is amortized across training runs that don't
change the schema; new design = new schema = new pack.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ChannelSpec:
    """One field in the packed row layout."""
    name: str               # how trainers refer to it
    npz_key: str            # which key to extract from each source .npz
    columns: int            # how many columns this field occupies in
                            # the packed row (1 for scalar-per-frame,
                            # N for (T, N) array)
    src_slice: tuple[int, int] | None = None
    # optional (start, stop) column slice to extract from npz_key. Use
    # when the source key holds more columns than you want packed:
    # e.g. extract just labels[:, 0] from a labels array with shape (T, 3).
    # When set, columns must equal stop - start.


@dataclass
class CorpusSchema:
    """Ordered list of channels = the packed row layout."""
    channels: list[ChannelSpec] = field(default_factory=list)

    @property
    def total_columns(self) -> int:
        return sum(c.columns for c in self.channels)

    def column_range(self, name: str) -> slice:
        """Column slice for a named channel. Raises KeyError if missing."""
        offset = 0
        for c in self.channels:
            if c.name == name:
                return slice(offset, offset + c.columns)
            offset += c.columns
        raise KeyError(f'channel {name!r} not in schema '
                       f'(have: {[c.name for c in self.channels]})')

    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'channels': [
                {
                    'name': c.name,
                    'npz_key': c.npz_key,
                    'columns': c.columns,
                    'src_slice': list(c.src_slice) if c.src_slice else None,
                }
                for c in self.channels
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CorpusSchema':
        if d.get('version') != 1:
            raise ValueError(f'unsupported schema version: {d.get("version")}')
        channels = []
        for c in d['channels']:
            channels.append(ChannelSpec(
                name=c['name'],
                npz_key=c['npz_key'],
                columns=c['columns'],
                src_slice=tuple(c['src_slice']) if c.get('src_slice') else None,
            ))
        return cls(channels=channels)


# Manifest filename in a packed corpus dir
SCHEMA_FILENAME = 'schema.json'


def write_schema(dst_dir: Path, schema: CorpusSchema) -> Path:
    """Write the schema manifest to dst_dir/schema.json. Returns the path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    p = dst_dir / SCHEMA_FILENAME
    p.write_text(json.dumps(schema.to_dict(), indent=2))
    return p


def load_schema(packed_dir: Path) -> CorpusSchema:
    """Load schema from a packed corpus dir."""
    p = packed_dir / SCHEMA_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f'no schema manifest at {p} — this dir does not look like a '
            f'packed corpus written by wallpaper_ml.packed_corpus')
    return CorpusSchema.from_dict(json.loads(p.read_text()))


def pack_one(src: Path, dst: Path, schema: CorpusSchema) -> tuple[str, str]:
    """Pack one source .npz file to a .npy at dst per schema.

    Returns a (status, message) tuple:
      ('ok',      '')          on success
      ('skip',    reason)      if file is missing keys / not a label file
      ('exists',  '')          if dst already exists
      ('error',   detail)      if pack failed for an unexpected reason

    Atomic write: writes to dst.tmp then renames.
    """
    if dst.exists():
        return ('exists', '')
    try:
        with np.load(src) as d:
            files = set(d.files)
            # All required keys must be present. If any is missing, skip
            # (caller can decide if missing-keys are errors or expected).
            missing = [c.npz_key for c in schema.channels if c.npz_key not in files]
            if missing:
                return ('skip', f'{src.name}: missing keys {missing}')
            arrays = []
            T = None
            for ch in schema.channels:
                arr = np.asarray(d[ch.npz_key], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]  # (T,) → (T, 1)
                if arr.ndim != 2:
                    return ('error', f'{src.name}/{ch.npz_key}: '
                                     f'shape {arr.shape}; need (T,) or (T, N)')
                if ch.src_slice is not None:
                    start, stop = ch.src_slice
                    if stop - start != ch.columns:
                        return ('error', f'{src.name}/{ch.npz_key}: '
                                         f'src_slice width {stop-start} != '
                                         f'declared columns {ch.columns}')
                    if stop > arr.shape[1]:
                        return ('error', f'{src.name}/{ch.npz_key}: '
                                         f'src_slice stop {stop} > '
                                         f'available cols {arr.shape[1]}')
                    arr = arr[:, start:stop]
                elif arr.shape[1] != ch.columns:
                    return ('error', f'{src.name}/{ch.npz_key}: '
                                     f'{arr.shape[1]} cols, '
                                     f'expected {ch.columns}')
                if T is None:
                    T = arr.shape[0]
                elif arr.shape[0] != T:
                    return ('error', f'{src.name}/{ch.npz_key}: T='
                                     f'{arr.shape[0]} != {T} (earlier channel)')
                arrays.append(arr)
        packed = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
        tmp = Path(str(dst) + '.tmp')
        np.save(tmp, packed)
        # np.save may append .npy if the path didn't already end in it.
        actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + '.npy')
        actual_dst = dst if dst.suffix == '.npy' else dst.with_suffix('.npy')
        actual_tmp.replace(actual_dst)
        return ('ok', '')
    except (KeyError, OSError, ValueError) as e:
        return ('error', f'{src.name}: {type(e).__name__}: {e}')


def _pack_worker(args):
    src, dst, schema = args
    return src, pack_one(src, dst, schema)


def pack_corpus(src_dir: Path, dst_dir: Path, schema: CorpusSchema,
                pattern: str = '*.npz',
                n_jobs: int | None = None,
                progress_every: int = 100,
                log=print) -> dict:
    """Pack all matching files from src_dir into dst_dir per schema.

    Writes schema.json into dst_dir, then walks src_dir for files
    matching `pattern`, packs each via pack_one in parallel.

    Args:
        src_dir, dst_dir: source and destination directories
        schema: CorpusSchema describing the row layout
        pattern: glob pattern for source files (default *.npz)
        n_jobs: worker process count (default: cpu_count())
        progress_every: print a progress line every N files (0 = silent)
        log: where to write progress (callable accepting str)

    Returns a summary dict: {'ok': N, 'skip': N, 'exists': N, 'error': N,
                              'errors': [(src_name, msg), ...]}
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    write_schema(dst_dir, schema)

    src_files = sorted(p for p in src_dir.glob(pattern)
                       if not p.name.endswith('.tmp.npz')
                       and not p.name.endswith('.npz.tmp'))
    if not src_files:
        return {'ok': 0, 'skip': 0, 'exists': 0, 'error': 0, 'errors': []}

    pairs = [(f, dst_dir / (f.stem + '.npy'), schema) for f in src_files]
    n_jobs = n_jobs or mp.cpu_count()
    counts = {'ok': 0, 'skip': 0, 'exists': 0, 'error': 0}
    errors = []
    t0 = time.monotonic()
    done = 0

    with mp.Pool(processes=n_jobs) as pool:
        for src, (status, msg) in pool.imap_unordered(_pack_worker, pairs,
                                                       chunksize=8):
            counts[status] = counts.get(status, 0) + 1
            if status in ('skip', 'error') and msg:
                errors.append((src.name, msg))
                if status == 'error':
                    log(f'  ERROR {msg}')
            done += 1
            if progress_every and done % progress_every == 0:
                rate = done / (time.monotonic() - t0)
                eta = (len(pairs) - done) / rate
                log(f'  packed {done}/{len(pairs)} '
                    f'({rate:.0f}/s, ETA {eta/60:.1f} min)  '
                    f'ok={counts["ok"]} skip={counts["skip"]} '
                    f'exists={counts["exists"]} error={counts["error"]}')

    summary = dict(counts, errors=errors)
    elapsed = time.monotonic() - t0
    log(f'Packed {done} files in {elapsed/60:.1f} min: {counts}')
    return summary


def load_packed(path: Path) -> np.memmap:
    """mmap-load a packed file. Returns (T, total_cols) memmap."""
    return np.load(path, mmap_mode='r')


def packed_frames(path: Path) -> int:
    """Read frame count from a packed .npy file's header without
    loading data. Faster than np.load when you only need shape.
    """
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape[0]
