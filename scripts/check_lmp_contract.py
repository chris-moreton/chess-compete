#!/usr/bin/env python3
"""Verify that SPSA LMP seed values match a live rusty-rival checkout.

This is intentionally a standalone, required check rather than a sibling-path
unit test. Rusty Rival CI checks out chess-compete's default branch and invokes
this script with its own event checkout, so an engine-only threshold change
cannot silently leave the SPSA baseline stale.
"""

import argparse
import ast
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEPTHS = range(1, 9)


def assigned_literal(path: Path, name: str):
    """Return a module-level literal assignment without importing application code."""
    module = ast.parse(path.read_text(), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def engine_source(engine_root: Path) -> str:
    constants = engine_root / "src" / "engine_constants.rs"
    if not constants.is_file():
        raise ValueError(f"engine constants not found: {constants}")
    return constants.read_text()


def engine_thresholds(engine_root: Path) -> dict[int, int]:
    source = engine_source(engine_root)
    match = re.search(
        r"pub const LMP_MOVE_THRESHOLDS: \[u8; (\d+)\] = \[([^]]+)\];",
        source,
    )
    if not match:
        raise ValueError("LMP_MOVE_THRESHOLDS declaration not found or has an unsupported shape")
    declared_len = int(match.group(1))
    values = [int(value.strip()) for value in match.group(2).split(",") if value.strip()]
    if declared_len != len(values) or values[0] != 0 or len(values) != len(DEPTHS) + 1:
        raise ValueError(f"unexpected LMP_MOVE_THRESHOLDS shape: {values}")
    return dict(zip(DEPTHS, values[1:]))


def contract_errors(engine_root: Path, spsa_root: Path = REPO) -> list[str]:
    expected = engine_thresholds(engine_root)
    source = engine_source(engine_root)
    master = spsa_root / "compete" / "spsa" / "master.py"
    build = spsa_root / "compete" / "spsa" / "build.py"
    defaults = assigned_literal(master, "DEFAULT_PARAMS")
    fallbacks = assigned_literal(build, "LMP_DEFAULTS")
    mappings = assigned_literal(build, "ARRAY_PARAM_MAPPINGS")
    errors = []

    lmp_mapping = mappings.get("lmp_move_thresholds")
    if not lmp_mapping or "pattern" not in lmp_mapping:
        errors.append("ARRAY_PARAM_MAPPINGS is missing lmp_move_thresholds.pattern")
    else:
        matches = list(re.finditer(lmp_mapping["pattern"], source))
        if len(matches) != 1:
            errors.append(
                "ARRAY_PARAM_MAPPINGS lmp_move_thresholds.pattern must match "
                f"the live engine exactly once, matched {len(matches)} times"
            )

    for depth, value in expected.items():
        key = f"lmp_threshold_depth{depth}"
        cfg = defaults.get(key)
        if cfg is None:
            errors.append(f"DEFAULT_PARAMS is missing {key}")
        elif cfg.get("value") != value:
            errors.append(f"DEFAULT_PARAMS {key}={cfg.get('value')} but engine ships {value}")
        if fallbacks.get(depth) != value:
            errors.append(f"LMP_DEFAULTS depth {depth}={fallbacks.get(depth)} but engine ships {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True,
                        help="Rusty Rival checkout containing src/engine_constants.rs")
    parser.add_argument("--spsa-root", type=Path, default=REPO,
                        help="chess-compete checkout containing SPSA defaults (defaults to this checkout)")
    args = parser.parse_args()
    try:
        errors = contract_errors(args.engine_root, args.spsa_root)
    except (OSError, SyntaxError, ValueError) as error:
        print(f"LMP SPSA contract check could not run: {error}", file=sys.stderr)
        return 2
    if errors:
        print("LMP SPSA contract failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("LMP SPSA contract matches live engine constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
