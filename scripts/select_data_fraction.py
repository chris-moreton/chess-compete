#!/usr/bin/env python3
"""Reduce ~/data/*.data to a target fraction of total bytes.

Lives in its own file deliberately. It was previously a heredoc inlined inside
the `su - ubuntu -c '...'` block of launch-nnue-training.sh, which is a
single-quoted string containing a `<<'PYEOF'` marker -- itself quoted. Editing
that block programmatically corrupted it (a stray quote closed the su string and
the remainder of the Python was executed as shell), and the failure was invisible
to `bash -n` because user-data is opaque text to the outer script. Two GPU
instances then idled for six hours having done nothing.

Usage:  select_data_fraction.py <fraction> [data_dir]
"""

import glob
import os
import random
import sys

# bulletformat record size. Truncation must land on a record boundary or the
# trainer reads a partial final record.
REC = 32

# Fixed seed: the kept subset must be reproducible across runs, otherwise two
# "equal-sized" corpora are not comparable.
SEED = 20260728


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: select_data_fraction.py <fraction> [data_dir]", file=sys.stderr)
        return 2

    frac = float(sys.argv[1])
    data_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/data")

    files = sorted(glob.glob(os.path.join(data_dir, "*.data")))
    if not files:
        print(f"no .data files in {data_dir}", file=sys.stderr)
        return 1

    total = sum(os.path.getsize(f) for f in files)
    if frac >= 1.0:
        print(f"  fraction {frac} >= 1.0, keeping all {len(files)} shards ({total / 1e9:.3f} GB)")
        return 0

    target = int(total * frac)
    random.Random(SEED).shuffle(files)

    # Take whole shards up to the target, then truncate the last one to land on
    # it exactly. Whole-shard selection alone is far too coarse at small
    # fractions: shards range 364MB-2.7GB, so an 0.89GB target could take a
    # single 2.7GB shard and overshoot 3x. That silently biases any experiment
    # whose premise is equal-sized corpora -- and since data volume is worth
    # ~24 Elo per doubling, an 18% overshoot is worth ~6 Elo to whichever side
    # gets it (NET-326).
    keep, acc = [], 0
    for f in files:
        if acc >= target:
            break
        keep.append(f)
        acc += os.path.getsize(f)

    removed = 0
    for f in files:
        if f not in keep:
            os.remove(f)
            removed += 1

    if acc > target and keep:
        last = keep[-1]
        newsize = os.path.getsize(last) - (acc - target)
        newsize -= newsize % REC
        if newsize >= REC:
            with open(last, "r+b") as fh:
                fh.truncate(newsize)
            print(f"  truncated {os.path.basename(last)} to {newsize / 1e6:.0f} MB")
        else:
            os.remove(last)
            keep.pop()

    final = sum(os.path.getsize(f) for f in keep)
    print(
        f"  kept {len(keep)} shards, {final / 1e9:.3f} GB of {total / 1e9:.3f} GB "
        f"= {final / total:.2%} (target {frac:.2%}), removed {removed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
