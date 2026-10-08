"""Static analysis tests for ML compute shaders.

Catches the software-atomic-float anti-pattern that triggered the GRU
backward GuC wedge on 2026-05-31: `atomicCompSwap` inside a HOT loop —
a for/while iteration where multiple inner iterations target the same
parameter address. Under heavy contention every CAS retries and threads
thrash.

The CAS-spin idiom itself (`do { ... atomicCompSwap ... } while (...)`)
is the standard way to emulate `atomicAdd<float>` on backends that lack
the extension (Mesa-Xe in particular). When it's the *only* enclosing
loop — i.e. a helper like:

    void atomicAddFloat(uint idx, float val) {
        ...
        do {
            ...
            atomicCompSwap(...);
        } while (...);
    }

— it's NOT the anti-pattern. Each invocation is 1-iteration-typical;
the retries are proportional to actual contention, not loop trip count.
This is OK practice and the scanner ignores it.

What the scanner flags is CAS reached from inside a for/while iteration,
where the same address gets repeatedly accumulated:

    for (int t = 0; t < T; t++) {            ← outer for-loop
        ...
        while (true) {                       ← inner CAS-spin
            atomicCompSwap(grad[tgt], ...);
            ...
        }
    }

Across T iterations, each thread hammers the same `grad[tgt]` T times.
That's the storm shape — fixed via workgroup-private accumulation
(Path B, task #69).

Shaders that still hold a HOT CAS-in-loop pattern after Path B are
listed in SOFTWARE_ATOMIC_FLOAT_DEBT with a per-shader note describing
why each one is still there (smaller blast radius, restructure planned,
etc.). The list is documentation of remaining work, not a permanent
allowlist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


SHADERS_ROOT = (Path(__file__).resolve().parents[1]
                / 'src' / 'wallpaper_ml' / 'shaders')


# Shaders that still flag the HOT CAS-in-loop pattern after the
# emulation-idiom carveout (see scanner docstring). The 3 helper-using
# shaders (conv2d_backward_weights, gap_linear_backward, gap_mlp_backward)
# were removed when the scanner learned that pure do-while spins are
# the standard atomicAdd<float> emulation. The remaining 3 entries are
# actual residual hot-pattern instances:
#
# - gru_seq_backward.comp: post-Path-B (commit 748851c), the inner-loop
#   storm is gone but the end-of-shader flush still walks (3 gates × I)
#   for W, (3 × H) for U, and 6 bias slots with one CAS-spin per slot.
#   Per-CAS contention is now B-thread instead of B×T, so the impact
#   dropped ~256× without eliminating the syntactic pattern. The true
#   structural fix would be Mesa-Xe exposing VK_EXT_shader_atomic_float
#   (so we'd use hardware atomicAdd instead of CAS-spin) — tracked as
#   task #37 (Mesa upstream report).
#
# - gru_backward.comp: per-timestep version of gru_seq_backward, called
#   T times from Python. Not on the production training path (we use
#   backward_sequence). Same shape as gru_seq_backward pre-Path-B
#   inside one dispatch (B×H simultaneous CAS), but only one timestep
#   per dispatch so the total per-step contention is bounded. Low
#   priority — would only get refactored if we ever shift back to
#   per-step training.
#
# - linear_backward.comp: contention is B (batch) threads racing the
#   same grad_W[i,j] address across workgroups. Modest (B typically 8).
#   Restructure would change dispatch shape (one thread per (i,j)
#   accumulating across batches in private memory instead of one per
#   (b,i)). Tracked but not urgent.
SOFTWARE_ATOMIC_FLOAT_DEBT: set[str] = {
    'rnn/gru_backward.comp',
    'rnn/gru_seq_backward.comp',
    'rnn/linear_backward.comp',
}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    """Remove // line and /* block */ comments so loop/CAS tokens inside
    them don't confuse the parser."""
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)
    return src


def find_cas_in_loops(source: str) -> list[tuple[int, str]]:
    """Walk shader source tracking brace depth + currently-open loop
    bodies. For each `atomicCompSwap` token encountered while inside a
    HOT loop body, return (line_number, line_text).

    A "hot" loop is any `for` or `while`. A `do-while` enclosing only
    the CAS itself is the standard `atomicAdd<float>` emulation idiom —
    NOT flagged when it's the sole enclosing loop, because each call
    is 1-iteration-typical (retries scale with contention, not with
    loop trip count).

    Each open loop's frame records the brace depth and the kind
    ('for' / 'while' / 'do'). The CAS is flagged iff at least one
    enclosing frame is 'for' or 'while'. Pure 'do' chains pass.

    Loop bodies close when brace depth drops below their open depth.
    Single-statement loop bodies (`for(...) stmt;`) don't push a frame.

    The parser is pragmatic, not a full GLSL frontend. It doesn't
    handle preprocessor directives that hide braces or other esoterica.
    Good enough for production shader hygiene.
    """
    src = _strip_comments(source)
    findings: list[tuple[int, str]] = []
    lines = source.split('\n')

    brace_depth = 0
    # Stack of (brace_depth, kind) where kind ∈ {'for', 'while', 'do'}
    loop_frames: list[tuple[int, str]] = []
    pending_loop_kind: str | None = None
    line = 1
    i = 0
    N = len(src)

    while i < N:
        c = src[i]
        if c == '\n':
            line += 1
            i += 1
            continue
        if c == '{':
            brace_depth += 1
            if pending_loop_kind is not None:
                loop_frames.append((brace_depth, pending_loop_kind))
                pending_loop_kind = None
            i += 1
            continue
        if c == '}':
            while loop_frames and loop_frames[-1][0] == brace_depth:
                loop_frames.pop()
            brace_depth -= 1
            i += 1
            continue
        if c == ';':
            if pending_loop_kind is not None:
                # Single-statement loop body — no frame to track.
                pending_loop_kind = None
            i += 1
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < N and (src[j].isalnum() or src[j] == '_'):
                j += 1
            tok = src[i:j]
            if tok in ('for', 'while'):
                # Skip the condition paren to avoid spurious tokens.
                k = j
                while k < N and src[k] != '(':
                    if src[k] == '\n':
                        line += 1
                    k += 1
                if k < N:
                    depth = 1
                    k += 1
                    while k < N and depth > 0:
                        if src[k] == '(':
                            depth += 1
                        elif src[k] == ')':
                            depth -= 1
                        elif src[k] == '\n':
                            line += 1
                        k += 1
                pending_loop_kind = tok
                i = k
                continue
            if tok == 'do':
                pending_loop_kind = 'do'
                i = j
                continue
            if tok == 'atomicCompSwap' and loop_frames:
                # Flag only if at least one enclosing loop is for/while.
                # Pure do-while chains are the CAS-emulation idiom.
                if any(kind != 'do' for _depth, kind in loop_frames):
                    text = lines[line - 1].rstrip() if 0 < line <= len(lines) else tok
                    findings.append((line, text))
            i = j
            continue
        i += 1

    return findings


# ---------------------------------------------------------------------------
# Self-tests for the scanner (so the test that uses it is itself trusted)
# ---------------------------------------------------------------------------

def test_scanner_finds_cas_in_while_loop():
    src = """
    void main() {
        while (true) {
            atomicCompSwap(buf[0], 0u, 1u);
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1
    assert findings[0][0] == 4


def test_scanner_finds_cas_in_for_loop():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            atomicCompSwap(buf[i], 0u, 1u);
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1


def test_scanner_finds_cas_in_nested_loops():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                while (true) {
                    atomicCompSwap(buf[0], 0u, 1u);
                }
            }
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1


def test_scanner_does_not_flag_cas_outside_any_loop():
    src = """
    void main() {
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_does_not_flag_cas_after_loop_closes():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            // unrelated work
            buf[i] += 1u;
        }
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_ignores_cas_in_comment():
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            // atomicCompSwap(buf[0], 0u, 1u);
            /* atomicCompSwap(buf[0], 0u, 1u); */
            buf[i] += 1u;
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_handles_for_keyword_in_identifier():
    """Words like 'before' or 'format' contain 'for' but aren't loops."""
    src = """
    void main() {
        int format = 0;
        atomicCompSwap(buf[0], 0u, 1u);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_skips_emulation_do_while_idiom():
    """The CAS-spin do-while is the standard atomicAdd<float>
    emulation. When it's the only enclosing loop, the retries scale
    with contention (typically 1), not with loop trip count — that's
    OK practice, not the anti-pattern. Skip it."""
    src = """
    void atomicAddFloat(uint idx, float val) {
        uint old_val = buf[idx];
        do {
            uint assumed = old_val;
            uint new_val = floatBitsToUint(uintBitsToFloat(assumed) + val);
            old_val = atomicCompSwap(buf[idx], assumed, new_val);
        } while (old_val != assumed);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


def test_scanner_flags_cas_in_for_even_with_inner_do_while():
    """When an outer for/while wraps a CAS-spin do-while, that IS the
    hot anti-pattern — the for-loop iterations all target the same
    address, accumulating contention. Flag it."""
    src = """
    void main() {
        for (int i = 0; i < 10; i++) {
            uint old_val = buf[0];
            do {
                uint assumed = old_val;
                old_val = atomicCompSwap(buf[0], assumed, assumed + 1u);
            } while (old_val != assumed);
        }
    }
    """
    findings = find_cas_in_loops(src)
    assert len(findings) == 1


def test_scanner_skips_nested_do_while_chain():
    """Pure do-while chains (no outer for/while) are still emulation
    idioms — synthetic case but ensures we check kind not count."""
    src = """
    void main() {
        do {
            do {
                atomicCompSwap(buf[0], 0u, 1u);
            } while (false);
        } while (false);
    }
    """
    findings = find_cas_in_loops(src)
    assert findings == []


# ---------------------------------------------------------------------------
# Whole-shader-tree scan
# ---------------------------------------------------------------------------

def _all_shader_files() -> list[Path]:
    if not SHADERS_ROOT.exists():
        pytest.skip(f'shader root not found: {SHADERS_ROOT}')
    return sorted(SHADERS_ROOT.rglob('*.comp'))


def _relative_key(path: Path) -> str:
    """Stable name for the debt allowlist: 'cnn/foo.comp'."""
    return str(path.relative_to(SHADERS_ROOT))


@pytest.mark.parametrize('shader_path', _all_shader_files(),
                          ids=lambda p: _relative_key(p))
def test_no_new_software_atomic_float(shader_path: Path):
    """Every .comp must either be free of the anti-pattern, or be
    listed in SOFTWARE_ATOMIC_FLOAT_DEBT with an explanatory entry
    while task #69 is in flight.

    If you wrote a new shader and this test fails, DO NOT add it to
    the debt list — use the workgroup-shared accumulation pattern
    instead. See `wallpaper_ml/tests/test_shader_health.py` docstring
    for the canonical fix.

    If you ported a debt shader to the new pattern, REMOVE it from
    SOFTWARE_ATOMIC_FLOAT_DEBT and this test will then enforce that
    it stays clean.
    """
    key = _relative_key(shader_path)
    findings = find_cas_in_loops(shader_path.read_text())

    if key in SOFTWARE_ATOMIC_FLOAT_DEBT:
        if not findings:
            pytest.fail(
                f'{key} is listed in SOFTWARE_ATOMIC_FLOAT_DEBT but no '
                f'atomicCompSwap-in-loop was found. The debt has been '
                f'resolved — remove this shader from the allowlist.')
        # Debt acknowledged; pass.
        return

    if findings:
        msg = [
            f'{key} contains atomicCompSwap inside a loop body '
            f'(software-atomic-float anti-pattern):'
        ]
        for line_num, text in findings:
            msg.append(f'  line {line_num}: {text}')
        msg.append('')
        msg.append('Fix: workgroup-local shared-memory accumulation + '
                   'single atomicAdd per workgroup. See task #69 and '
                   'the gru_seq_backward.comp port for the reference '
                   'implementation pattern.')
        pytest.fail('\n'.join(msg))


def test_debt_list_matches_reality():
    """Every entry in SOFTWARE_ATOMIC_FLOAT_DEBT must correspond to a
    real shader file. Stale entries make the debt look bigger than it
    is and break the resolution check above."""
    for key in SOFTWARE_ATOMIC_FLOAT_DEBT:
        path = SHADERS_ROOT / key
        assert path.exists(), (
            f'SOFTWARE_ATOMIC_FLOAT_DEBT references {key!r} but no '
            f'such file exists. Was the shader renamed or deleted?')
