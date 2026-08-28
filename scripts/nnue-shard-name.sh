#!/bin/bash
# Map a relative raw-data path to a collision-resistant converted shard name.

set -euo pipefail

if [ "$#" -ne 1 ] || [[ "$1" = /* ]]; then
    echo "Usage: $0 <relative .txt/.zst path>" >&2
    exit 1
fi

relative=$1
name=${relative##*/}
case "$name" in
    *.txt) stem=${name%.txt} ;;
    *.zst) stem=${name%.zst} ;;
    *) echo "Error: expected a .txt or .zst path" >&2; exit 1 ;;
esac

digest=$(printf '%s' "$relative" | sha256sum)
digest=${digest%% *}
printf '%s-%s\n' "$stem" "$digest"
