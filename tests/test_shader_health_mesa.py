"""Mesa-stat shader health tests — slow (one Vulkan compile per shader).

Compiles each `.comp` under `INTEL_DEBUG=cs` and parses Mesa's emitted
stats: SIMD variant, spill ops, per-basic-block cycle estimates. Catches
what the source-level scanner (test_shader_health.py) can't see:

- Register spillage (Mesa narrows from SIMD16 → SIMD8 + emits spill/fill
  ops when a shader exceeds the register file). The chaos shader pre-trim
  saga (2026-05-31 daytime) was this class — universal flame_chaos.comp
  at 127 variations: 1929:3326 spills, SIMD8, 61M cycles.
- Worst-case basic-block cycle estimates above 1M. Mesa assigns very
  large cycle counts to blocks containing contended atomic-spin loops
  (the GRU backward saga, 2026-05-31 evening) and to over-long inner
  loops generally. A single block over the threshold is a smell even
  if total spillage is fine.

Slow because each subprocess Vulkan compile takes ~1-2s; the full
shader tree (~20 files) is ~30-40s. Run via:

    pytest wallpaper_ml/tests/test_shader_health_mesa.py -v
    pytest -m slow                # all slow tests
    pytest -m "not slow"          # skip slow tests

The SOFTWARE_ATOMIC_FLOAT_DEBT allowlist (defined in test_shader_health)
is honored here too — debt shaders won't fail the basic-block cycle
test until they get ported to workgroup-shared accumulation.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

# Import the sibling test module directly by file path rather than via a
# rootdir-relative dotted name (e.g. `tests.wallpaper_ml.test_shader_health`
# or `wallpaper_ml.tests.test_shader_health`) — that name changes whenever
# this file's location relative to the pytest rootdir changes (it did, when
# this moved from tests/wallpaper_ml/ to wallpaper_ml/tests/, and it will
# again once wallpaper_ml becomes its own repo). A same-directory sys.path
# insert is invariant to both.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shader_health import (  # noqa: E402
    SHADERS_ROOT,
    SOFTWARE_ATOMIC_FLOAT_DEBT,
    _all_shader_files,
    _relative_key,
)

# Thresholds — tunable. Set conservatively so they catch real
# pathologies without flagging every busy shader.
MAX_SPILL_OPS = 100
MAX_BLOCK_CYCLES = 1_000_000


# ---------------------------------------------------------------------------
# Shader introspection (so we can build a matching pipeline)
# ---------------------------------------------------------------------------

def _parse_binding_count(source: str) -> int:
    """Highest `binding = N` declared + 1. Returns 1 if shader has no
    explicit bindings (won't happen for compute shaders, but defensive)."""
    bindings = [int(b) for b in re.findall(r'binding\s*=\s*(\d+)', source)]
    return max(bindings) + 1 if bindings else 1


def _has_push_constants(source: str) -> bool:
    return 'push_constant' in source


# ---------------------------------------------------------------------------
# Subprocess Vulkan compile + Mesa stat capture
# ---------------------------------------------------------------------------

VK_COMPILE_RUNNER = r"""
import sys
from pathlib import Path
sys.path.insert(0, '{project}/wallpaper_ml/src')

from wallpaper_ml.vk_compute import VkCompute

gpu = VkCompute()

bufs = [gpu.create_buffer(64 * 1024, usage='storage')
        for _ in range({n_bindings})]

p = gpu.create_pipeline(
    '{shader_path}',
    buffers=bufs,
    push_constant_size={push_size},
)
# No dispatch needed — pipeline create already triggered Mesa compile.
"""


STAT_LINE_RE = re.compile(
    r'^SIMD(?P<simd>\d+) shader:\s+(?P<instr>\d+) instructions\.\s+'
    r'(?P<loops>\d+) loops?\.\s+(?P<cycles>\d+) cycles\.\s+'
    r'(?P<spills>\d+):(?P<fills>\d+) spills:fills,\s+'
    r'(?P<sends>\d+) sends',
    re.MULTILINE)

# `   START B7 <-B6 (1360 cycles)` — Mesa basic-block annotation
BLOCK_LINE_RE = re.compile(
    r'^\s*START\s+B(?P<id>\d+)(?:\s+<-B\d+)*\s+\((?P<cycles>\d+)\s+cycles\)',
    re.MULTILINE)


def _compile_shader_capture_stats(shader_path: Path) -> dict:
    """Run the shader through Mesa-Xe compile under INTEL_DEBUG=cs.

    Returns a dict per SIMD variant:
        {simd_width: {'instr', 'cycles', 'spills', 'fills', 'sends',
                       'loops', 'max_block_cycles', 'all_blocks': [(id, cyc)]}}
    """
    src = shader_path.read_text()
    n_bindings = _parse_binding_count(src)
    push_size = 128 if _has_push_constants(src) else 0

    project = Path(__file__).resolve().parents[2]
    code = VK_COMPILE_RUNNER.format(
        project=project, shader_path=shader_path,
        n_bindings=n_bindings, push_size=push_size,
    )
    env = {**os.environ,
           'MESA_SHADER_CACHE_DISABLE': '1',
           'INTEL_DEBUG': 'cs'}
    result = subprocess.run([sys.executable, '-c', code],
                             env=env, capture_output=True,
                             text=True, timeout=60)
    if result.returncode != 0:
        # Surface stderr for debugging — usually a shader bind/pipeline
        # mismatch we can fix by tweaking the introspection.
        raise RuntimeError(
            f'compile failed for {shader_path.name}:\n'
            f'stdout: {result.stdout[-500:]}\n'
            f'stderr tail: {result.stderr[-1000:]}')

    out: dict[int, dict] = {}
    for m in STAT_LINE_RE.finditer(result.stderr):
        simd = int(m.group('simd'))
        out[simd] = {
            'instr':  int(m.group('instr')),
            'cycles': int(m.group('cycles')),
            'spills': int(m.group('spills')),
            'fills':  int(m.group('fills')),
            'sends':  int(m.group('sends')),
            'loops':  int(m.group('loops')),
            'all_blocks': [],
        }

    # Per-block annotations appear AFTER the stat header for each
    # variant. We don't strictly need to disambiguate variant ownership
    # — the test is "does the shader have any block over threshold,
    # in any compiled variant", and Mesa's estimates are similar
    # across variants. So aggregate to all variants found.
    blocks = [(int(m.group('id')), int(m.group('cycles')))
              for m in BLOCK_LINE_RE.finditer(result.stderr)]
    max_block = max((c for _, c in blocks), default=0)
    for variant in out.values():
        variant['all_blocks'] = blocks
        variant['max_block_cycles'] = max_block

    return out


# ---------------------------------------------------------------------------
# Fixture — compile every shader once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def all_shader_stats() -> dict[str, dict]:
    """Compile every shader once and cache results for the session.
    Keyed by the same relative-path string as the debt allowlist."""
    return {_relative_key(p): _compile_shader_capture_stats(p)
            for p in _all_shader_files()}


# ---------------------------------------------------------------------------
# Per-shader assertions
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('shader_path', _all_shader_files(),
                          ids=lambda p: _relative_key(p))
def test_no_excessive_spillage(shader_path: Path, all_shader_stats):
    """Spill ops over MAX_SPILL_OPS in any SIMD variant means Mesa
    couldn't fit the shader in the register file and is roundtripping
    through memory. This is a hard perf cliff regardless of the rest of
    the shader's behavior.

    The chaos shader regression that motivated this test:
      N=127 (full untrimmed) → SIMD8, 1929:3326 spills, 61M cycles
    Per-genome trim brought N down to ~10-50 → 0 spills.
    """
    key = _relative_key(shader_path)
    variants = all_shader_stats.get(key, {})
    assert variants, f'no Mesa stats parsed for {key}'

    over_threshold = [(simd, v['spills'], v['fills'])
                      for simd, v in variants.items()
                      if v['spills'] > MAX_SPILL_OPS]
    if over_threshold:
        msg = [f'{key}: register spillage over {MAX_SPILL_OPS} ops:']
        for simd, spills, fills in over_threshold:
            msg.append(f'  SIMD{simd}: {spills} spills, {fills} fills')
        msg.append('')
        msg.append('Fix: reduce live-register pressure. Common shapes — '
                   'split large switch dispatchers (chaos shader pattern: '
                   'spec-const trim), reduce shared state held across '
                   'loop iterations, or split the shader into smaller '
                   'dispatches each doing less work.')
        pytest.fail('\n'.join(msg))


@pytest.mark.slow
@pytest.mark.parametrize('shader_path', _all_shader_files(),
                          ids=lambda p: _relative_key(p))
def test_no_extreme_basic_blocks(shader_path: Path, all_shader_stats):
    """Mesa annotates each basic block with a worst-case cycle estimate.
    Blocks over MAX_BLOCK_CYCLES are almost always one of two
    pathologies:

      1. Contended atomic-CAS spin loops (the gru_seq_backward saga
         2026-05-31 evening: million-cycle blocks for grad accumulation).
         Tracked structurally by SOFTWARE_ATOMIC_FLOAT_DEBT — those
         shaders are exempted here pending the Path B port (task #69).

      2. Runaway inner loops where Mesa couldn't bound the trip count
         and assumed worst-case. Not always a bug, but worth a
         second look.
    """
    key = _relative_key(shader_path)
    variants = all_shader_stats.get(key, {})
    assert variants, f'no Mesa stats parsed for {key}'

    # All variants share `max_block_cycles` (per-shader aggregate).
    max_block = next(iter(variants.values()))['max_block_cycles']

    if max_block <= MAX_BLOCK_CYCLES:
        return  # clean

    if key in SOFTWARE_ATOMIC_FLOAT_DEBT:
        # Debt acknowledged — these will exceed the threshold until
        # the CAS spin loops get replaced with workgroup-shared
        # accumulation (task #69). Pass.
        return

    blocks = next(iter(variants.values()))['all_blocks']
    hot = sorted([(c, bid) for bid, c in blocks if c > MAX_BLOCK_CYCLES],
                 reverse=True)
    msg = [f'{key}: {len(hot)} basic blocks over {MAX_BLOCK_CYCLES:,} '
           f'cycles (worst: {hot[0][0]:,}):']
    for cyc, bid in hot[:5]:
        msg.append(f'  B{bid}: {cyc:,} cycles')
    if len(hot) > 5:
        msg.append(f'  ... and {len(hot) - 5} more')
    msg.append('')
    msg.append('Common causes: contended atomic spin loops (use '
               'workgroup-shared accumulation), unbounded inner loops, '
               'or per-iteration heavy work that Mesa estimates '
               'pessimistically. If this is a known CAS-in-loop case, '
               'add to SOFTWARE_ATOMIC_FLOAT_DEBT in test_shader_health.py '
               'with a remediation reference.')
    pytest.fail('\n'.join(msg))


# ---------------------------------------------------------------------------
# Informational test (always passes, prints summary)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_shader_stats_summary(all_shader_stats, capsys):
    """Prints a human-readable summary table. Always passes; for
    eyeballing the shader inventory's perf shape during dev."""
    with capsys.disabled():
        print()
        print(f'{"shader":<48}  {"SIMD":>5}  {"inst":>5}  '
              f'{"spill":>5}  {"max-block":>12}')
        print('-' * 85)
        for key in sorted(all_shader_stats):
            variants = all_shader_stats[key]
            for simd in sorted(variants):
                v = variants[simd]
                tag = ' *DEBT' if key in SOFTWARE_ATOMIC_FLOAT_DEBT else ''
                print(f'{key:<48}  '
                      f'SIMD{simd:>2}  '
                      f'{v["instr"]:>5}  '
                      f'{v["spills"]:>5}  '
                      f'{v["max_block_cycles"]:>12,}{tag}')
