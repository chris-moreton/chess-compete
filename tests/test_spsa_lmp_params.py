"""LMP depth thresholds must be declared, patchable, and match the engine.

Three separate failures, so three separate tests:

1. PR #7 claimed "all eight LMP depths tunable". `build.py` could patch all
   eight, but `DEFAULT_PARAMS` declared only 1-3, so new runs created no values
   for 4-8 and SPSA could never perturb them.
2. `DEFAULT_PARAMS` then seeded depths 2-3 as 5/8 while the engine and
   `build.py`'s own fallback both say 6/9 - a defaults run would silently start
   from a different baseline than the shipped engine, before iteration 1.
3. An earlier version of this file read the sibling ../rusty-rival checkout, so
   it passed here and failed in an isolated chess-compete clone.

Hermetic: the engine constant below is an inline fixture, not a file read.
"""
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# Minimal stand-in for engine_constants.rs. Only the line under test needs to be
# real; the values match the shipped engine at time of writing.
ENGINE_CONSTANTS_FIXTURE = """
pub const LMP_MAX_DEPTH: u8 = 8;
pub const LMP_MOVE_THRESHOLDS: [u8; 9] = [0, 9, 6, 9, 19, 28, 39, 52, 67];
"""

# The shipped engine's values, duplicated here deliberately: if the engine
# changes, this test should fail and force a conscious decision rather than
# letting the tuner's baseline drift away from what actually ships.
SHIPPED = {1: 9, 2: 6, 3: 9, 4: 19, 5: 28, 6: 39, 7: 52, 8: 67}


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("spsabuild", REPO / "compete/spsa/build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_params_src():
    return (REPO / "compete/spsa/master.py").read_text()


def test_default_params_declare_all_eight_lmp_depths():
    """Declared, or a new run cannot tune them however good the patcher is."""
    declared = set(re.findall(r"'(lmp_threshold_depth\d)':", _default_params_src()))
    expected = {f"lmp_threshold_depth{d}" for d in range(1, 9)}
    assert expected <= declared, f"missing from DEFAULT_PARAMS: {sorted(expected - declared)}"


def test_default_params_seed_the_shipped_engine_values():
    """A defaults run must start from what the engine actually ships.

    Seeding anything else silently moves the baseline before iteration 1, so
    iteration 1 is not measuring the shipped engine.
    """
    src = _default_params_src()
    for depth, expected in SHIPPED.items():
        m = re.search(rf"'lmp_threshold_depth{depth}': {{'value': (\d+)", src)
        assert m, f"lmp_threshold_depth{depth} not found in DEFAULT_PARAMS"
        assert int(m.group(1)) == expected, (
            f"depth {depth} seeded {m.group(1)}, engine ships {expected}"
        )


def test_build_fallbacks_agree_with_the_shipped_engine(build):
    """build.py's own fallback table is a third copy of these numbers."""
    assert build.LMP_DEFAULTS == SHIPPED


def test_every_lmp_depth_lands_in_its_own_slot(build):
    """Patchable, and each depth reaches its own index.

    Asserts position, not presence: a mapping that wrote every value to the same
    index would pass a weaker "the number appears somewhere" check.
    """
    sentinels = {f"lmp_threshold_depth{d}": 3 + d for d in range(1, 9)}
    out = build.apply_parameters(ENGINE_CONSTANTS_FIXTURE, dict(sentinels))

    m = re.search(r"pub const LMP_MOVE_THRESHOLDS: \[u8; 9\] = \[([^\]]+)\];", out)
    assert m, "LMP_MOVE_THRESHOLDS missing or wrong shape after patching"
    values = [int(v.strip()) for v in m.group(1).split(",")]
    assert values == [0] + [sentinels[f"lmp_threshold_depth{d}"] for d in range(1, 9)], values


def test_stale_array_mapping_raises_rather_than_silently_doing_nothing(build):
    """The defect this whole change exists to prevent."""
    broken = ENGINE_CONSTANTS_FIXTURE.replace("[u8; 9]", "[u8; 7]")  # array grew again
    with pytest.raises(ValueError, match="did not match"):
        build.apply_parameters(broken, {"lmp_threshold_depth1": 5})
