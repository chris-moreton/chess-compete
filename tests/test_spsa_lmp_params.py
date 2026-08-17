"""LMP depth thresholds must be BOTH declared and patchable.

PR #7 claimed "all eight LMP depths tunable". `build.py` could patch all eight,
but `DEFAULT_PARAMS` declared only 1-3, so new runs never created values for
4-8 and SPSA could not perturb them. The capability existed and was not wired
up. Caught in review; these tests exist so it cannot regress silently.
"""
import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent


def _load_build():
    spec = importlib.util.spec_from_file_location("spsabuild", REPO / "compete/spsa/build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_declare_all_eight_lmp_depths():
    """Declared, or a new run cannot tune them however good the patcher is."""
    src = (REPO / "compete/spsa/master.py").read_text()
    declared = set(re.findall(r"'(lmp_threshold_depth\d)':", src))
    expected = {f"lmp_threshold_depth{d}" for d in range(1, 9)}
    assert expected <= declared, f"missing from DEFAULT_PARAMS: {sorted(expected - declared)}"


def test_every_lmp_depth_actually_changes_the_array():
    """Patchable, and each depth lands in its own slot.

    Asserts position as well as presence: a mapping that wrote every value to
    the same index would pass a weaker 'the number appears' check.
    """
    build = _load_build()
    constants = (REPO.parent / "rusty-rival/src/engine_constants.rs").read_text()
    sentinels = {f"lmp_threshold_depth{d}": 3 + d for d in range(1, 9)}

    out = build.apply_parameters(constants, dict(sentinels))
    m = re.search(r"pub const LMP_MOVE_THRESHOLDS: \[u8; 9\] = \[([^\]]+)\];", out)
    assert m, "LMP_MOVE_THRESHOLDS not found or wrong shape after patching"

    values = [int(v.strip()) for v in m.group(1).split(",")]
    assert values == [0] + [sentinels[f"lmp_threshold_depth{d}"] for d in range(1, 9)], values


def test_stale_array_mapping_raises_rather_than_silently_doing_nothing():
    """The defect this whole change exists to prevent."""
    build = _load_build()
    constants = (REPO.parent / "rusty-rival/src/engine_constants.rs").read_text()
    broken = constants.replace("[u8; 9]", "[u8; 7]")  # simulate the array growing again

    try:
        build.apply_parameters(broken, {"lmp_threshold_depth1": 5})
    except ValueError as e:
        assert "did not match" in str(e)
    else:
        raise AssertionError("a stale mapping patched nothing and did not raise")
